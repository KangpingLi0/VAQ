
import math
import time
import tqdm
import torch
import torch.nn as nn
import logging
from qmllm.utils.device import empty_cache, synchronize

class GPTQ:
    def __init__(self, layer):
        self.layer = layer
        self.dev = self.layer.weight.device
        W = layer.weight.data.clone()
        self.rows = W.shape[0]
        self.columns = W.shape[1]
        self.H = torch.zeros((self.columns, self.columns), device=self.dev)
        self.nsamples = 0

    def add_batch(self, inp, out):
        inp = inp.to(self.dev)
        if len(inp.shape) == 2:
            inp = inp.unsqueeze(0)
        tmp = inp.shape[0]
        if len(inp.shape) == 3:
            inp = inp.reshape((-1, inp.shape[-1]))
        inp = inp.t()
        self.H *= self.nsamples / (self.nsamples + tmp)
        self.nsamples += tmp
        # inp = inp.float()
        inp = inp.float()
        if not torch.isfinite(inp).all():
            count = int((~torch.isfinite(inp)).sum().item())
            logging.warning(
                "GPTQ calibration input contains %s NaN/Inf values; replacing them with zero",
                count,
            )
            inp = torch.nan_to_num(inp, nan=0.0, posinf=0.0, neginf=0.0)
        inp = math.sqrt(2 / self.nsamples) * inp
        # self.H += 2 / self.nsamples * inp.matmul(inp.t())
        self.H += inp.matmul(inp.t())

    def fasterquant(
        self, blocksize=128, percdamp=.01, groupsize=-1, actorder=False, static_groups=False
    ):
        W = self.layer.weight.data.clone()
        W = W.float()

        tick = time.time()

        if not self.quantizer.ready():
            self.quantizer.find_params(W)

        H = self.H
        del self.H
        dead = torch.diag(H) == 0
        H[dead, dead] = 1
        W[:, dead] = 0

        if static_groups:
            import copy
            groups = []
            for i in range(0, self.columns, groupsize):
                quantizer = copy.deepcopy(self.quantizer)
                quantizer.find_params(W[:, i:(i + groupsize)])
                groups.append(quantizer)

        if actorder:
            perm = torch.argsort(torch.diag(H), descending=True)
            W = W[:, perm]
            H = H[perm][:, perm]
            invperm = torch.argsort(perm)

        Losses = torch.zeros_like(W)
        Q = torch.zeros_like(W)

        mean_diag = torch.mean(torch.diag(H)).abs()
        if not torch.isfinite(mean_diag) or mean_diag.item() == 0:
            mean_diag = torch.ones((), device=self.dev, dtype=H.dtype)
        damp = percdamp * mean_diag
        diag = torch.arange(self.columns, device=self.dev)
        H[diag, diag] += damp
        ori_type = H.dtype
        # Ascend does not implement float64 Cholesky natively.  Letting the
        # backend fall back implicitly can cast the matrix back to float32 and
        # has produced a non-finite inverse for otherwise valid PSD Hessians.
        # Do the small linear-algebra step explicitly on CPU in float64, then
        # move only the resulting factor back to the accelerator.
        H = H.detach().to(device="cpu", dtype=torch.float64)
        H = (H + H.transpose(0, 1)) * 0.5
        if not torch.isfinite(H).all():
            count = int((~torch.isfinite(H)).sum().item())
            logging.warning(
                "GPTQ Hessian contains %s NaN/Inf values; replacing them with zero",
                count,
            )
            H = torch.nan_to_num(H, nan=0.0, posinf=0.0, neginf=0.0)
        cpu_diag = torch.arange(self.columns, device=H.device)
        base_jitter = max(float(damp.detach().cpu()), float(mean_diag.detach().cpu()) * 1e-6, 1e-6)
        last_err = None
        for attempt in range(8):
            jitter = base_jitter * (10 ** attempt)
            trial = H.clone()
            trial[cpu_diag, cpu_diag] += jitter
            try:
                chol = torch.linalg.cholesky(trial)
                candidate = torch.cholesky_inverse(chol)
                candidate = torch.linalg.cholesky(candidate, upper=True)
                if torch.isfinite(candidate).all():
                    Hinv = candidate
                    break
                last_err = ValueError("inverse-Hessian factor contains NaN or Inf")
            except (torch._C._LinAlgError, RuntimeError) as err:
                last_err = err
            logging.warning(
                "GPTQ Hessian factorization failed on attempt %s; retrying with diagonal jitter %.6g (%s)",
                attempt + 1,
                jitter,
                last_err,
            )
        else:
            raise last_err
        Hinv = Hinv.to(device=self.dev, dtype=ori_type)

        for i1 in range(0, self.columns, blocksize):
            i2 = min(i1 + blocksize, self.columns)
            count = i2 - i1

            W1 = W[:, i1:i2].clone()
            Q1 = torch.zeros_like(W1)
            Err1 = torch.zeros_like(W1)
            Losses1 = torch.zeros_like(W1)
            Hinv1 = Hinv[i1:i2, i1:i2]

            for i in range(count):
                w = W1[:, i]
                d = Hinv1[i, i]
                if not torch.isfinite(d) or d.abs().item() < 1e-12:
                    raise ValueError(
                        f"GPTQ inverse-Hessian diagonal is invalid at column {i1 + i}: {d.item()}"
                    )

                if groupsize != -1:
                    if not static_groups:
                        if (i1 + i) % groupsize == 0:
                            self.quantizer.find_params(W[:, (i1 + i):(i1 + i + groupsize)])
                    else:
                        idx = i1 + i
                        if actorder:
                            idx = perm[idx]
                        self.quantizer = groups[idx // groupsize]

                q = self.quantizer.quantize(w.unsqueeze(1)).flatten()
                Q1[:, i] = q
                Losses1[:, i] = (w - q) ** 2 / d ** 2

                err1 = (w - q) / d
                W1[:, i:] -= err1.unsqueeze(1).matmul(Hinv1[i, i:].unsqueeze(0))
                Err1[:, i] = err1

            Q[:, i1:i2] = Q1
            Losses[:, i1:i2] = Losses1 / 2

            W[:, i2:] -= Err1.matmul(Hinv[i1:i2, i2:])

        synchronize(self.dev)

        if actorder:
            Q = Q[:, invperm]

        self.layer.weight.data = Q.reshape(self.layer.weight.shape).to(self.layer.weight.data.dtype)
        if torch.any(torch.isnan(self.layer.weight.data)):
            logging.warning('NaN in weights')
            logging.warning(
                "GPTQ quantizer state: bits=%s scale_finite=%s zero_finite=%s",
                self.quantizer.bits,
                bool(torch.isfinite(self.quantizer.scale).all()),
                bool(torch.isfinite(self.quantizer.zero).all()),
            )
            raise ValueError('NaN in weights')

    def free(self):
        self.H = None
        self.Losses = None
        self.Trace = None
        empty_cache(self.dev)
        cleanup_memory(verbos=False, device=self.dev)


def cleanup_memory(verbos=True, device=None) -> None:
    """Run GC and clear GPU memory."""
    import gc
    import inspect
    caller_name = ''
    try:
        caller_name = f' (from {inspect.stack()[1].function})'
    except (ValueError, KeyError):
        pass

    resolved = torch.device(device) if device is not None else None

    def total_reserved_mem() -> int:
        if resolved is not None and resolved.type == "cuda" and torch.cuda.is_available():
            return sum(torch.cuda.memory_reserved(device=i) for i in range(torch.cuda.device_count()))
        if resolved is not None and resolved.type == "npu" and hasattr(torch, "npu") and hasattr(torch.npu, "memory_reserved"):
            return sum(torch.npu.memory_reserved(device=i) for i in range(torch.npu.device_count()))
        return 0

    memory_before = total_reserved_mem()

    # gc.collect and empty cache are necessary to clean up GPU memory if the model was distributed
    gc.collect()

    empty_cache(resolved)
    memory_after = total_reserved_mem()
    if verbos and resolved is not None and resolved.type in {"cuda", "npu"}:
        logging.info(
            f"{resolved.type.upper()} memory{caller_name}: {memory_before / (1024 ** 3):.2f} -> {memory_after / (1024 ** 3):.2f} GB"
            f" ({(memory_after - memory_before) / (1024 ** 3):.2f} GB)"
        )
