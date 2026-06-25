import triton
import triton.language as tl
"""
    Triton operates on pointers of data.
    Input -- Use pointer to load.
    Output -- Initiate an output data, use pointer to write.

    Example below: y = X @ w, where x -- (..., n), w -- (n, )

"""

@triton.jit
def weighted_sum_fwd(
    x_ptr, weight_ptr,  # Input pointers for x and weight; x -- (..., n); w -- (n,)
    output_ptr, # Output pointer
    x_stride_row, x_stride_dim, # Strides -- how to move one element in each axis of a tensor.
    weight_stride_dim, 
    output_stride_dim,
    NUM_ROWS, D,
    ROWS_TILE_SIZE: tl.constexpr, D_TILE_SIZE: tl.constexpr # Tile shapes must be known at compile time
):
    # Each instance will compute the weighted sum of a tile of rows of x.
    
    # "tl.program_id" gives us a way to check which thread block we're running in
    row_tile_idx = tl.program_id(0)  # 0 is the axis of tha launch grid

    # Block pointers give us a way toselect from an ND region of memory and move our selection around.
    # The block pointer must know:
    #  - The pointer to the fist element of the tensor
    #  - The overall shape of the tensor to handle out-of-bounds access
    #  - The strides of each dimension to use the memory layout properly
    #  - The ND coordinates of the starting block, i.e., "offsets"
    #  - The block share to load/store at a time
    #  - The order of the dimensions in memory from major to minor
    #      axes (= np.argsort(strides)) for optimizations, needed for TMA support on >= Hopper

    # Below does not load anything; only builds the descriptor for the compiler.
    x_block_ptr = tl.make_block_ptr(
        x_ptr,
        shape=(NUM_ROWS, D,), 
        strides=(x_stride_row, x_stride_dim),
        offsets=(row_tile_idx * ROWS_TILE_SIZE, 0),
        block_shape=(ROWS_TILE_SIZE, D_TILE_SIZE),
        order=(1,0) # The memory order. (1, 0) indicates row-major layout.
    )

    weight_block_ptr = tl.make_block_ptr(
        weight_ptr,
        shape=(D,),
        strides=(weight_stride_dim,),
        offsets=(0,),
        block_shape=(D_TILE_SIZE,),
        order=(0,),
    )

    output_block_ptr = tl.make_block_ptr(
        output_ptr,
        shape=(NUM_ROWS,),
        strides=(output_stride_dim,),
        offsets=(row_tile_idx * ROWS_TILE_SIZE,),
        block_shape=(ROWS_TILE_SIZE,),
        order=(0,),
    )

    # Init a buffer for write
    output = tl.zeros((ROWS_TILE_SIZE,), dtype=tl.float32)

    for i in range(tl.cdiv(D, D_TILE_SIZE)): # cdiv -- ceiling divide (round up)
        # Load the current block pointer
        # Since ROW_TILE_SIZE might not devide NUM_ROWS, and D_TILE_SIZE might not divide D,
        # We need boundary checks for both dimensions
        row = tl.load(x_block_ptr, boundary_check=(0, 1), padding_option="zero")
        weight = tl.load(weight_block_ptr, boundary_check=(0,), padding_option="zero")
        
        # Compute weighted sum (of the tile)
        output += tl.sum(row * weight[None, :], axis=1)

        # Move the pointers to the next tile
        # These are (rows, cols) cordinate deltas
        x_block_ptr = x_block_ptr.advance((0, D_TILE_SIZE))
        weight_block_ptr = weight_block_ptr.advance((D_TILE_SIZE,))
    
    # Write the output to the output block pointer
    # Again, ROWS_TILE_SIZE might not divide NUM_ROWS, a boundary check is necessary
    tl.store(output_block_ptr, output, boundary_check=(0,))


@triton.jit
def weighted_sum_backward(
    x_ptr, weight_ptr, # input
    grad_output_ptr, # Grad input
    grad_x_ptr, partial_grad_weight_ptr, # Grad outputs
    stride_xr, stride_xd,
    stride_wd,
    stride_gr,
    stride_gxr, stride_gxd,
    stride_gwr, stride_gwd,
    NUM_ROWS, D,
    ROWS_TILE_SIZE: tl.constexpr, D_TILE_SIZE: tl.constexpr,
):
    row_tile_idx = tl.program_id(0)
    n_row_tiles = tl.num_programs(0)

    # Inputs
    grad_output_block_ptr = tl.make_block_ptr(
        grad_output_ptr,
        shape=(NUM_ROWS,), strides=(stride_gr,),
        offsets=(row_tile_idx * ROWS_TILE_SIZE,),
        block_shape=(ROWS_TILE_SIZE,),
        order=(0,)
    )
    
    x_block_ptr = tl.make_block_ptr(
        x_ptr,
        shape=(NUM_ROWS, D), strides=(stride_xr, stride_xd),
        offsets=(row_tile_idx * ROWS_TILE_SIZE, 0),
        block_shape=(ROWS_TILE_SIZE, D_TILE_SIZE),
        order=(1, 0)
    )

    weight_block_ptr = tl.make_block_ptr(
        weight_ptr, 
        shape=(D,), strides=(stride_wd,),
        offsets=(0,), block_shape=(D_TILE_SIZE, ),
        order=(0,)
    )

    grad_x_block_ptr = tl.make_block_ptr(
        grad_x_ptr,
        shape=(NUM_ROWS, D,), strides=(stride_gxr, stride_gxd),
        offsets=(row_tile_idx * ROWS_TILE_SIZE, 0),
        block_shape=(ROWS_TILE_SIZE, D_TILE_SIZE),
        order=(1, 0)
    )

    partial_grad_weight_block_ptr = tl.make_block_ptr(
        partial_grad_weight_ptr,
        shape=(n_row_tiles, D), strides=(stride_gwr, stride_gwd),
        offsets=(row_tile_idx, 0),
        block_shape=(1, D_TILE_SIZE),
        order=(1, 0)
    )

    grad_output = tl.load(grad_output_block_ptr, boundary_check=(0,), padding_option="zero") # (ROWS_TILE_SIZE)
    for i in range(tl.cdiv(D, D_TILE_SIZE)):
        
        # Outer product for grad_x
        weight = tl.load(weight_block_ptr, boundary_check=(0,), padding_option="zero") # (D_TILE_SIZE)
        grad_x_row = grad_output[:, None] * weight[None, :]
        tl.store(grad_x_block_ptr, grad_x_row, boundary_check=(0, 1))

        # Reduce as many rows as possible for the grad_weight_result
        row = tl.load(x_block_ptr, boundary_check=(0, 1), padding_option="zero") # (ROWS_TILE_SIZE, D_TILE_SIZE)
        grad_weight_row = tl.sum(row * grad_output[:, None], axis=0, keep_dims=True)
        tl.store(partial_grad_weight_block_ptr, grad_weight_row, boundary_check=(1,)) # Dim 0 is reduced and can never be out of bound

        # Move the pointers to the next tile along D
        x_block_ptr = x_block_ptr.advance((0, D_TILE_SIZE))
        weight_block_ptr = weight_block_ptr.advance((D_TILE_SIZE,))
        partial_grad_weight_block_ptr = partial_grad_weight_block_ptr.advance((0, D_TILE_SIZE))
        grad_x_block_ptr = grad_x_block_ptr.advance((0, D_TILE_SIZE))


import torch
from einops import rearrange
class WeightedSumFunc(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight):
        # Cache x and weight to be used in the backward pass, when we 
        # only receive the gradient w.r.t the output tensor,
        # and need to compute the gradient w.r.t. x and weight
        D, output_dims = x.shape[-1], x.shape[:-1]

        # Reshape input tensor to 2D
        input_shape = x.shape
        x = rearrange(x, "... d -> (...) d")

        ctx.save_for_backward(x, weight)

        # Shape check
        assert len(weight.shape) == 1 and weight.shape[0] == D, f"Dimension mismatch, {weight.shape}"
        assert x.is_cuda and weight.is_cuda, "Expected CUDA tensors"
        assert x.is_contiguous(), "Our pointer arithmetric will assume contiguous x."

        ctx.D_TILE_SIZE = max(triton.next_power_of_2(D) // 16, 1)  # Roughly 16 loops through the embedding dimension (>= 1 for small D)
        ctx.ROWS_TILE_SIZE = 16  # Each thread processes 16 batch elements at a time
        ctx.input_shape = input_shape

        # Need to initialize empty result tensor. 
        # Note: these elements are not necessarily 0 (Garbage OK.)
        # Launch our kernel with n instances in our 1D grid
        n_rows = output_dims.numel()
        y = torch.empty(n_rows, device=x.device, dtype=x.dtype)

        # Launching 1D grid
        weighted_sum_fwd[(triton.cdiv(n_rows, ctx.ROWS_TILE_SIZE),)](
            x_ptr=x, weight_ptr=weight, output_ptr=y,
            x_stride_row=x.stride(0), x_stride_dim=x.stride(1),
            weight_stride_dim=weight.stride(0),
            output_stride_dim=y.stride(0),
            NUM_ROWS=n_rows,
            D=D,
            ROWS_TILE_SIZE=ctx.ROWS_TILE_SIZE,
            D_TILE_SIZE=ctx.D_TILE_SIZE,
        )

        return y.view(input_shape[:-1])

    @staticmethod
    def backward(ctx, grad_out):
        x, weight = ctx.saved_tensors # Load checkpoint
        grad_out = grad_out.contiguous().view(-1)

        ROWS_TILE_SIZE, D_TILE_SIZE = ctx.ROWS_TILE_SIZE, ctx.D_TILE_SIZE
        n_rows, D = x.shape

        # Strategy: for each thread block, first write to a partial buffer
        #  Thenm we reduce over this buffer to get the final gradient.
        partial_grad_weight = torch.empty(
            (triton.cdiv(n_rows, ROWS_TILE_SIZE), D),
            device=x.device, dtype=x.dtype
        )
        grad_x = torch.empty_like(x)

        weighted_sum_backward[(triton.cdiv(n_rows, ROWS_TILE_SIZE),)](
            x_ptr=x, weight_ptr=weight, grad_output_ptr=grad_out,
            grad_x_ptr=grad_x, partial_grad_weight_ptr=partial_grad_weight,
            stride_xr=x.stride(0), stride_xd=x.stride(1),
            stride_wd=weight.stride(0),
            stride_gr=grad_out.stride(0),
            stride_gxr=grad_x.stride(0), stride_gxd=grad_x.stride(1),
            stride_gwr=partial_grad_weight.stride(0), stride_gwd=partial_grad_weight.stride(1),
            NUM_ROWS=n_rows, D=D,
            ROWS_TILE_SIZE=ROWS_TILE_SIZE, D_TILE_SIZE=D_TILE_SIZE,
        )
        grad_weight = partial_grad_weight.sum(axis=0)
        
        return grad_x.view(ctx.input_shape), grad_weight


f_weightedsum = WeightedSumFunc.apply

device = torch.device("cuda")
B, S, D = 4, 32, 8
x = torch.randn(size=(B, S, D)).to(device=device)
w = torch.randn(size=(D,)).to(device=device)

y = f_weightedsum(x, w)
print(y)