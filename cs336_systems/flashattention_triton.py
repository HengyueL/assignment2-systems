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
    K_TILE_SIZE: tl.constexpr,  # Bk
    is_causal: tl.constexpr
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
        # == Causal mask: query at position q only attends to keys k <= q ==
        if is_causal:
            query_idx = query_tile_index * Q_TILE_SIZE + tl.arange(0, Q_TILE_SIZE)
            Sij = tl.where(query_idx[:, None] >= key_idx[None, :], Sij, -1e6)
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


@triton.jit
def flash_bwd_kernel_KV(
    Q_ptr, K_ptr, V_ptr,
    dO_ptr, L_ptr, D_ptr,
    dK_ptr, dV_ptr,
    stride_qb, stride_qq, stride_qd,
    stride_kb, stride_kk, stride_kd,
    stride_vb, stride_vk, stride_vd,
    stride_ob, stride_oq, stride_od,
    stride_lb, stride_lq,
    stride_db, stride_dq,
    N_QUERIES, N_KEYS,
    scale,
    D: tl.constexpr,           # d
    Q_TILE_SIZE: tl.constexpr, # Bq
    K_TILE_SIZE: tl.constexpr,  # Bk
    is_causal: tl.constexpr
):
    key_tile_index = tl.program_id(0)
    batch_tile_index = tl.program_id(1)

    # Make blocks
    Q_block_ptr = tl.make_block_ptr(
        Q_ptr + batch_tile_index * stride_qb,
        shape=(N_QUERIES, D),
        strides=(stride_qq, stride_qd),
        offsets=(0, 0),
        block_shape=(Q_TILE_SIZE, D),
        order=(1, 0),
    )
    K_block_ptr = tl.make_block_ptr(
        K_ptr + batch_tile_index * stride_kb,
        shape=(N_KEYS, D),
        strides=(stride_kk, stride_kd),
        offsets=(key_tile_index * K_TILE_SIZE, 0),
        block_shape=(K_TILE_SIZE, D),
        order=(1, 0)
    )
    V_block_ptr = tl.make_block_ptr(
        V_ptr + batch_tile_index * stride_vb,
        shape=(N_KEYS, D),
        strides=(stride_vk, stride_vd),
        offsets=(key_tile_index * K_TILE_SIZE, 0),
        block_shape=(K_TILE_SIZE, D),
        order=(1, 0)
    )
    L_block_ptr = tl.make_block_ptr(
        L_ptr + batch_tile_index * stride_lb,
        shape=(N_QUERIES,),
        strides=(stride_lq,),
        offsets=(0,),
        block_shape=(Q_TILE_SIZE,),
        order=(0,)
    )
    dO_block_ptr = tl.make_block_ptr(
        dO_ptr + batch_tile_index * stride_ob,
        shape=(N_QUERIES, D),
        strides=(stride_oq, stride_od,),
        offsets=(0, 0),
        block_shape=(Q_TILE_SIZE, D),
        order=(1, 0)
    )
    D_block_ptr = tl.make_block_ptr(
        D_ptr + batch_tile_index * stride_db,
        shape=(N_QUERIES,),
        strides=(stride_dq,),
        offsets=(0,),
        block_shape=(Q_TILE_SIZE,),
        order=(0,)
    )

    # Output block
    dK_block_ptr = tl.make_block_ptr(
        dK_ptr + batch_tile_index * stride_kb,
        shape=(N_KEYS, D,),
        strides=(stride_kk, stride_kd),
        offsets=(key_tile_index * K_TILE_SIZE, 0),
        block_shape=(K_TILE_SIZE, D,),
        order=(1, 0)
    )
    dV_block_ptr = tl.make_block_ptr(
        dV_ptr + batch_tile_index * stride_vb,
        shape=(N_KEYS, D,),
        strides=(stride_vk, stride_vd),
        offsets=(key_tile_index * K_TILE_SIZE, 0),
        block_shape=(K_TILE_SIZE, D,),
        order=(1, 0)
    )


    Kj = tl.load(K_block_ptr, boundary_check=(0, 1), padding_option="zero")
    Vj = tl.load(V_block_ptr, boundary_check=(0, 1), padding_option="zero")
    dKij = tl.zeros((K_TILE_SIZE, D), dtype=tl.float32)
    dVij = tl.zeros((K_TILE_SIZE, D), dtype=tl.float32)
    
    for i in range(tl.cdiv(N_QUERIES, Q_TILE_SIZE)):
        Qi = tl.load(Q_block_ptr, boundary_check=(0, 1), padding_option="zero")
        dOi = tl.load(dO_block_ptr, boundary_check=(0, 1), padding_option="zero")
        Li = tl.load(L_block_ptr, boundary_check=(0,), padding_option="zero")
        Di = tl.load(D_block_ptr, boundary_check=(0,), padding_option="zero")

        # Recompute attention scores and softmaxes
        Sij = tl.dot(Qi, tl.trans(Kj)) * scale
        # == Causal masking
        if is_causal:
            q_idx = i * Q_TILE_SIZE + tl.arange(0, Q_TILE_SIZE)
            k_idx = key_tile_index * K_TILE_SIZE + tl.arange(0, K_TILE_SIZE)
            Sij = tl.where(q_idx[:, None] >= k_idx[None, :], Sij, -1e6)

        Pij = tl.exp(Sij - Li[:, None])

        # Compute differentiations
        dVij = dVij + tl.dot(tl.trans(Pij).to(dOi.dtype), dOi)
        dPij = tl.dot(dOi, tl.trans(Vj))
        dSij = Pij * (dPij - Di[:, None])
        dKij = dKij + tl.dot(tl.trans(dSij).to(Qi.dtype), Qi) * scale

        Q_block_ptr = Q_block_ptr.advance((Q_TILE_SIZE, 0))
        dO_block_ptr = dO_block_ptr.advance((Q_TILE_SIZE, 0))
        L_block_ptr = L_block_ptr.advance((Q_TILE_SIZE,))
        D_block_ptr = D_block_ptr.advance((Q_TILE_SIZE,))
    
    tl.store(dK_block_ptr, dKij.to(dK_ptr.dtype.element_ty), boundary_check=(0, 1))
    tl.store(dV_block_ptr, dVij.to(dV_ptr.dtype.element_ty), boundary_check=(0, 1))


@triton.jit
def flash_bwd_kernel_Q(
    Q_ptr, K_ptr, V_ptr,
    dO_ptr, L_ptr, D_ptr,
    dQ_ptr,
    stride_qb, stride_qq, stride_qd,
    stride_kb, stride_kk, stride_kd,
    stride_vb, stride_vk, stride_vd,
    stride_ob, stride_oq, stride_od,
    stride_lb, stride_lq,
    stride_db, stride_dq,
    N_QUERIES, N_KEYS,
    scale,
    D: tl.constexpr,           # d
    Q_TILE_SIZE: tl.constexpr, # Bq
    K_TILE_SIZE: tl.constexpr,  # Bk
    is_causal: tl.constexpr
):
    q_tile_index = tl.program_id(0)
    batch_tile_index = tl.program_id(1)

    # Make blocks
    Q_block_ptr = tl.make_block_ptr(
        Q_ptr + batch_tile_index * stride_qb,
        shape=(N_QUERIES, D),
        strides=(stride_qq, stride_qd),
        offsets=(q_tile_index * Q_TILE_SIZE, 0),
        block_shape=(Q_TILE_SIZE, D),
        order=(1, 0),
    )
    K_block_ptr = tl.make_block_ptr(
        K_ptr + batch_tile_index * stride_kb,
        shape=(N_KEYS, D),
        strides=(stride_kk, stride_kd),
        offsets=(0, 0),
        block_shape=(K_TILE_SIZE, D),
        order=(1, 0)
    )
    V_block_ptr = tl.make_block_ptr(
        V_ptr + batch_tile_index * stride_vb,
        shape=(N_KEYS, D),
        strides=(stride_vk, stride_vd),
        offsets=(0, 0),
        block_shape=(K_TILE_SIZE, D),
        order=(1, 0)
    )
    L_block_ptr = tl.make_block_ptr(
        L_ptr + batch_tile_index * stride_lb,
        shape=(N_QUERIES,),
        strides=(stride_lq,),
        offsets=(q_tile_index * Q_TILE_SIZE,),
        block_shape=(Q_TILE_SIZE,),
        order=(0,)
    )
    dO_block_ptr = tl.make_block_ptr(
        dO_ptr + batch_tile_index * stride_ob,
        shape=(N_QUERIES, D),
        strides=(stride_oq, stride_od,),
        offsets=(q_tile_index * Q_TILE_SIZE, 0),
        block_shape=(Q_TILE_SIZE, D),
        order=(1, 0)
    )
    D_block_ptr = tl.make_block_ptr(
        D_ptr + batch_tile_index * stride_db,
        shape=(N_QUERIES,),
        strides=(stride_dq,),
        offsets=(q_tile_index * Q_TILE_SIZE,),
        block_shape=(Q_TILE_SIZE,),
        order=(0,)
    )
    
    # Init output block 
    dQ_block_ptr = tl.make_block_ptr(
        dQ_ptr + batch_tile_index * stride_qb,
        shape=(N_QUERIES, D),
        strides=(stride_qq, stride_qd),
        offsets=(q_tile_index * Q_TILE_SIZE, 0),
        block_shape=(Q_TILE_SIZE, D),
        order=(1, 0)
    )

    Qi = tl.load(Q_block_ptr, boundary_check=(0, 1), padding_option="zero")
    dOi = tl.load(dO_block_ptr, boundary_check=(0, 1), padding_option="zero")
    Li = tl.load(L_block_ptr, boundary_check=(0,), padding_option="zero")
    Di = tl.load(D_block_ptr, boundary_check=(0,), padding_option="zero")

    # Init Output
    dQij = tl.zeros((Q_TILE_SIZE, D), dtype=tl.float32)

    for j in range(tl.cdiv(N_KEYS, K_TILE_SIZE)):
        Kj = tl.load(K_block_ptr, boundary_check=(0, 1), padding_option="zero")
        Vj = tl.load(V_block_ptr, boundary_check=(0, 1), padding_option="zero")

        # Recomputation
        Sij = tl.dot(Qi, tl.trans(Kj)) * scale
        # == Causal masking
        if is_causal:
            q_index = q_tile_index * Q_TILE_SIZE + tl.arange(0, Q_TILE_SIZE)
            k_index = j * K_TILE_SIZE + tl.arange(0, K_TILE_SIZE)
            Sij = tl.where(q_index[:, None] >= k_index[None, :], Sij, -1e6)

        Pij = tl.exp(Sij - Li[:, None])  # Casted to float 32

        dPij = tl.dot(dOi, tl.trans(Vj))
        dSij = Pij * (dPij - Di[:, None])
        dQij += tl.dot(dSij.to(Kj.dtype), Kj) * scale

        K_block_ptr = K_block_ptr.advance((K_TILE_SIZE, 0))
        V_block_ptr = V_block_ptr.advance((K_TILE_SIZE, 0))
    
    tl.store(dQ_block_ptr, dQij.to(dQ_ptr.dtype.element_ty), boundary_check=(0, 1))


import torch
import math
class FlashattentionTriton(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx, Q, K, V,
        is_causal: bool = False
    ):
        assert len(Q.shape) == 3, "Requires Q shape to be [batch, n_seq, D]"
        assert len(K.shape) == 3, "Requires K shape to be [batch, n_k, D]"
        batch_size, N_QUERIES, D = Q.shape
        _, N_KEYS, _ = K.shape
        
        ctx.Q_TILE_SIZE = 16  # Pick a fixed size and let triton handle the bundary check
        ctx.K_TILE_SIZE = 16
        ctx.is_causal = is_causal
        
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
            Q_TILE_SIZE=ctx.Q_TILE_SIZE, K_TILE_SIZE=ctx.K_TILE_SIZE,
            is_causal=is_causal
        )
        ctx.save_for_backward(Q, K, V, O, L)
        return O
    

    @staticmethod
    def backward(ctx, grad_output):
        raise NotImplementedError

