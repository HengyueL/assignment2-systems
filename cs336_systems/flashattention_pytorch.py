import torch
import math
from einops import einsum


class FlashAttentionTorch(torch.autograd.Function):
    @staticmethod
    def forward(ctx, Q, K, V, is_causal=False):
        n_batch, Nq, d = Q.shape[0], Q.shape[1], Q.shape[2]
        Nk = K.shape[-2]
        Bq, Bk = 16, 16
        Tq, Tk = math.ceil(Nq / Bq), math.ceil(Nk / Bk)

        # Create output variables
        O_out = torch.empty_like(Q).to(device=Q.device)
        L_out = torch.empty(size=(n_batch, Nq)).to(device=Q.device)

        for i in range(Tq):
            Qi = Q[:, i*Bq:(i+1)*Bq, :]

            # Init State
            Oi = torch.zeros_like(Qi)
            li = torch.zeros(size=(n_batch, Bq)).to(device=Qi.device)
            mi = torch.empty(size=(n_batch, Bq)).fill_(-float("inf")).to(device=Qi.device)

            for j in range(Tk):
                Kj = K[:, j*Bk:(j+1)*Bk, :]
                Vj = V[:, j*Bk:(j+1)*Bk, :]

                Sij = einsum(Qi, Kj, "b bq d, b bk d -> b bq bk") / math.sqrt(d)
                mij = torch.maximum(mi, torch.amax(Sij, dim=-1, keepdim=False))
                Pij = torch.exp(Sij - mij[..., None])
                lij = (torch.exp(mi - mij) * li) + torch.sum(Pij, dim=-1, keepdim=False)
                Oij = einsum(torch.exp(mi - mij), Oi, "b bq, b bq d -> b bq d") \
                        + einsum(Pij, Vj, "b bq bk, b bk d -> b bq d")

                # Update Oi, li, mi
                Oi = Oij
                mi = mij
                li = lij
            
            Oi = Oi / li[..., None]
            Li = mi + torch.log(li)

            O_out[:, i*Bq:(i+1)*Bq, :] = Oi
            L_out[:,  i*Bq:(i+1)*Bq] = Li
        ctx.save_for_backward(L_out, Q, K, V, O_out)
        return O_out
    
    @staticmethod
    def backward(ctx, grad_out):
        raise NotImplementedError