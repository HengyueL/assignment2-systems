import triton
import triton.language as tl


@triton.jit
def flash_fwd_kernel(
    Q_ptr, K_ptr, V_ptr,
    O_ptr, L_ptr,
    stride_qb, stride_qq, stride_qd,
    stride_kb, stride_kk, stride_kd,
    stride_vb, stride_vk, stride_vd,
    stride_ob, stride_oq, stride_od,
    stride_lb, stride_lq,
    N_QUERIES, N_KEYS,
    scale,                     # 1/sqrt(d)
    D: tl.constexpr,           # d
    Q_TILE_SIZE: tl.constexpr, # Bq
    K_TILE_SIZE: tl.constexpr  # Bk
):
    # Program indices (2D grid)
    query_tile_index = tl.program_id(0)
    batch_index = tl.program_id(1)

    # Offset example - By batch
    Q_block_ptr = tl.make_block_ptr(
        Q_ptr + batch_index * stride_qb,
        shape=(N_QUERIES, D),
        strides=(stride_qq, stride_qd),
        offsets=(query_tile_index * Q_TILE_SIZE, 0),
        block_shape=(Q_TILE_SIZE, D),
        order=(1, 0),
    )

    K_block_ptr = tl.make_block_ptr(
        K_ptr + batch_index * stride_kb,
        shape=(N_KEYS, D),
        strides=(stride_kk, stride_kd),
        offsets=(0, 0),
        block_shape=(K_TILE_SIZE, D),
        order=(1, 0)
    )

    V_block_ptr = tl.make_block_ptr(
        V_ptr + batch_index * stride_vb,
        shape=(N_KEYS, D),
        strides=(stride_vk, stride_vd),
        offsets=(0, 0),
        block_shape=(K_TILE_SIZE, D),
        order=(1, 0),
    )

    O_block_ptr = tl.make_block_ptr(
        O_ptr + batch_index * stride_ob,
        shape=(N_QUERIES, D),
        strides=(stride_oq, stride_od),
        offsets=(query_tile_index * Q_TILE_SIZE, 0),
        block_shape=(Q_TILE_SIZE, D),
        order=(1, 0)
    )

    L_block_ptr = tl.make_block_ptr(
        L_ptr + batch_index * stride_lb,
        shape=(N_QUERIES,),
        strides=(stride_lq,),
        offsets=(query_tile_index * Q_TILE_SIZE,),
        block_shape=(Q_TILE_SIZE,),
        order=(0,)
    )

    Qi = tl.load(Q_block_ptr, boundary_check=(0, 1), padding_option="zero")

    # Init a buffer for output
    Oi = tl.zeros(shape=(Q_TILE_SIZE, D), dtype=tl.float32)
    Li = tl.zeros(shape=(Q_TILE_SIZE,), dtype=tl.float32)
    mi = tl.full(shape=(Q_TILE_SIZE,), value=-float("inf"), dtype=tl.float32)
    
    for j in range(tl.cdiv(N_KEYS, K_TILE_SIZE)):
        Kj = tl.load(K_block_ptr, boundary_check=(0, 1), padding_option="zero")
        Vj = tl.load(V_block_ptr, boundary_check=(0, 1), padding_option="zero")

        Sij = tl.dot(Qi.to(Kj.dtype), tl.trans(Kj)) * scale
        # == Mask the padded keys in the last tile ==
        key_idx = j * K_TILE_SIZE + tl.arange(0, K_TILE_SIZE)
        Sij = tl.where(key_idx[None, :] < N_KEYS, Sij, -float("inf"))
        # === 

        mij = tl.maximum(mi, tl.max(Sij, axis=-1, keep_dims=False))
        Pij = tl.exp(Sij - mij[:, None])

        lij = tl.exp(mi - mij) * Li + tl.sum(Pij, axis=-1)
        Oij = tl.exp(mi - mij)[:, None] * Oi + tl.dot(Pij.to(dtype=Vj.dtype), Vj)

        # Update buffer
        Oi = Oij
        Li = lij
        mi = mij

        # Move Pointer
        K_block_ptr = K_block_ptr.advance((K_TILE_SIZE, 0))
        V_block_ptr = V_block_ptr.advance((K_TILE_SIZE, 0))

    Oi = Oi / Li[:, None]
    Li = mi + tl.log(Li)

    tl.store(O_block_ptr, Oi.to(O_ptr.dtype.element_ty), boundary_check=(0, 1))
    tl.store(L_block_ptr, Li, boundary_check=(0,))


import torch
import math
class FlashattentionTriton(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx, Q, K, V,
        is_causal:bool = False
    ):
        assert len(Q.shape) == 3, "Requires Q shape to be [batch, n_seq, D]"
        assert len(K.shape) == 3, "Requires K shape to be [batch, n_k, D]"
        batch_size, N_QUERIES, D = Q.shape
        _, N_KEYS, _ = K.shape
        
        ctx.Q_TILE_SIZE = 16  # Pick a fixed size and let triton handle the bundary check
        ctx.K_TILE_SIZE = 16
        
        launch_grid = (triton.cdiv(N_QUERIES, ctx.Q_TILE_SIZE), batch_size)

        # Initialize Empty result tensor
        O = torch.empty_like(Q)
        L = torch.empty(size=(batch_size, N_QUERIES)).to(device=Q.device, dtype=torch.float32)

        # Launch Kernel Grid
        flash_fwd_kernel[launch_grid](
            Q, K ,V,
            O, L,
            stride_qb=Q.stride(0), stride_qq=Q.stride(1), stride_qd=Q.stride(2),
            stride_kb=K.stride(0), stride_kk=K.stride(1), stride_kd=K.stride(2),
            stride_vb=V.stride(0), stride_vk=V.stride(1), stride_vd=V.stride(2),
            stride_ob=O.stride(0), stride_oq=O.stride(1), stride_od=O.stride(2),
            stride_lb=L.stride(0), stride_lq=L.stride(1),
            N_QUERIES=N_QUERIES, N_KEYS=N_KEYS,
            scale=1./math.sqrt(D),
            D=D,
            Q_TILE_SIZE=ctx.Q_TILE_SIZE, K_TILE_SIZE=ctx.K_TILE_SIZE
        )
        ctx.save_for_backward(Q, K, V, O, L)
        return O
    

    @staticmethod
    def backward(ctx, grad_output):
        raise NotImplementedError

