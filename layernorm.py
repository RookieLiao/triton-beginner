import torch

import triton
import triton.language as tl


class NaiveLayerNorm:
    @staticmethod
    def forward(x, w, b):
        # x is the input activations, of shape B,T,C
        # w are the weights, of shape C
        # b are the biases, of shape C
        C = x.size(-1)
        # calculate the mean
        mean = x.sum(-1, keepdim=True) / C  # B,T,1
        # calculate the variance
        xshift = x - mean  # B,T,C
        var = (xshift**2).sum(-1, keepdim=True) / C  # B,T,1
        # calculate the inverse standard deviation: **0.5 is sqrt, **-0.5 is 1/sqrt
        rstd = (var + 1e-5) ** -0.5  # B,T,1
        # normalize the input activations
        norm = xshift * rstd  # B,T,C
        # scale and shift the normalized activations at the end
        out = norm * w + b  # B,T,C

        # return the output and the cache, of variables needed later during the backward pass
        cache = (x, w, mean, rstd)
        return out, cache

    @staticmethod
    def backward(dout, cache):
        x, w, mean, rstd = cache
        # recompute the norm (save memory at the cost of compute)
        norm = (x - mean) * rstd
        dw = (dout * norm).sum(dim=(0, 1))
        db = dout.sum(dim=(0, 1))
        dnorm = dout * w
        dx = (
            dnorm
            - dnorm.mean(dim=-1, keepdim=True)
            - norm * (dnorm * norm).mean(dim=-1, keepdim=True)
        )
        dx *= rstd
        return dw, db, dx

@triton.jit
def _layer_norm_fwd():
    pass
