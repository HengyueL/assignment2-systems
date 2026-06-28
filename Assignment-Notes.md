# Kernel dtype casting

The core idea: "load low, compute high, store low"

Data lives in global memory (HBM) at low precision (fp16/bf16) to save bandwidth. But arithmetic — especially summation
and transcendentals — needs high precision (fp32) or it loses accuracy. So a kernel has three zones:

| Zone | Dtype | Why |
| --- | --- | --- |
| Storage (loads/stores, global memory) | fp16/bf16 (whatever the tensor is) | bandwidth |
| Matmul inputs (`tl.dot` operands) | low precision, but both must match | tensor cores |
| Accumulators & precision-sensitive math (running sums, max, exp, log, normalizers) | fp32 | numerical accuracy |

Every cast you write is moving a value between two of these zones.

The three places you actually need a cast

1. Before tl.dot, when the two operands have different dtypes. This is the one that errors if you skip it. tl.dot requires
both operands to be the same (supported) dtype. The classic case: one operand is a freshly-computed fp32 value, the other
is loaded storage data.

In your forward, line 96:
tl.dot(Pij.to(dtype=Vj.dtype), Vj)
Pij came out of tl.exp(...) → it's fp32. Vj was loaded → it's fp16/bf16. They don't match, so you cast Pij down to Vj's
dtype. (Line 82 does the same: Qi.to(Kj.dtype).)

Key fact: tl.dot always accumulates and returns fp32, even with fp16 inputs. So the output of a matmul is already
high-precision — you only manage the inputs.

2. Accumulators must be fp32 — declare them fp32 and keep them there. Your forward gets this right at lines 74–76:
Oi = tl.zeros((Q_TILE_SIZE, D), dtype=tl.float32)
Li = tl.zeros((Q_TILE_SIZE,), dtype=tl.float32)
mi = tl.full((Q_TILE_SIZE,), -float("inf"), dtype=tl.float32)
Anything you sum/max over a loop (online-softmax stats, the output accumulator, your dKij/dVij) belongs here. fp16 has ~3
decimal digits and a tiny range; accumulating hundreds of terms in fp16 bleeds precision and can overflow. The matmul's
internal fp32 accumulation doesn't protect your own loop additions.

3. On tl.store, cast the fp32 accumulator down to the buffer's dtype. Line 110:
tl.store(O_block_ptr, Oi.to(O_ptr.dtype.element_ty), ...)
Oi is fp32, the output buffer O is fp16/bf16 → cast on the way out. (ptr.dtype.element_ty is how you read a buffer's
element dtype in Triton — handy so you don't hardcode it.) Contrast line 111: L is an fp32 buffer and Li is fp32, so no 
cast — they already match.

A decision checklist

For any value in a kernel, ask:

1. Is it an operand of tl.dot? → make both operands the same dtype (usually cast the fp32 one down to the loaded
low-precision one to hit tensor cores). Result comes back fp32 for free.
2. Is it accumulated/reduced across a loop (sum, max, output buffer, dK/dV)? → fp32.
3. Is it a precision-sensitive elementwise op (exp, log, dividing by the softmax normalizer, the S − L subtraction)? →
fp32.
4. Am I storing it to global memory? → cast to that buffer's element_ty.
5. Is everything already fp32? → you need none of the above. (This is why your fp32 path "just works.")

# Customized backward pass

PyTorch `grad_output` can give a non-contiguous tensor. It is safer to call:
```
grad_output = grad_output.continuous()
```
for the safe implementation of kernel backward pass.