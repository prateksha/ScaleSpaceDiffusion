######################################################################

import torch
import torch.nn.functional as F

class BilinearOp:
    """
    M: bilinear resize from (H_in, W_in) -> (H_out, W_out)
    MT: adjoint (transpose) operator computed via a single VJP
    """
    def __init__(self, H_in, W_in, H_out, W_out, *, align_corners=False, antialias=True):
        self.H_in, self.W_in = H_in, W_in
        self.size_out = (H_out, W_out)
        self.align_corners = align_corners
        self.antialias = antialias

    def M(self, x: torch.Tensor) -> torch.Tensor:
        # x: (N,C,H_in,W_in)

        assert x.shape[-2] == self.H_in and x.shape[-1] == self.W_in, \
            f"Input shape H,W ({x.shape[-2]},{x.shape[-1]}) does not match expected ({self.H_in},{self.W_in})"
        
        H_out, W_out = self.size_out
        if self.H_in ==H_out and self.W_in == W_out:
            return x  # Identity 
        
        return F.interpolate(
            x, size=self.size_out, mode="bilinear",
            align_corners=self.align_corners, antialias=self.antialias
        )

    def MT(self, y: torch.Tensor) -> torch.Tensor:

        assert y.shape[-2] == self.H_out and y.shape[-1] == self.W_out, \
            f"Input shape H,W ({y.shape[-2]},{y.shape[-1]}) does not match expected ({self.H_out},{self.W_out})"

        H_out, W_out = self.size_out
        if self.H_in ==H_out and self.W_in == W_out:
            return y  # Identity
        
        # y: (N,C,H_out,W_out)
        N, C = y.shape[:2]
        with torch.enable_grad():  # <-- ensures autograd is on even if caller used no_grad()
            x = torch.zeros((N, C, self.H_in, self.W_in),
                            device=y.device, dtype=y.dtype, requires_grad=True)
            out = F.interpolate(
                x, size=self.size_out, mode="bilinear",
                align_corners=self.align_corners, antialias=self.antialias
            )
            # VJP: returns (∂⟨out, y⟩/∂x)
            (g,) = torch.autograd.grad(out, x, grad_outputs=y, retain_graph=False)
        return g

from typing import Callable, Optional, Tuple
# -------------------------------------------------
# Lanczos: y ≈ f(A) b   (all 1-D; A is SPD)
# -------------------------------------------------
@torch.no_grad()
def lanczos_fAb_1d(
    A_mv_1d: Callable[[torch.Tensor], torch.Tensor],
    b1d: torch.Tensor,
    f: Callable[[torch.Tensor], torch.Tensor],
    iters: int = 40,
    reorth: bool = False,
) -> torch.Tensor:
    """
    A_mv_1d: function v -> A v   (both 1-D)
    b1d:     1-D tensor (shape [D])
    f:       spectral func (e.g., lambda x: x.sqrt() for A^{1/2})
    """
    device, dtype = b1d.device, b1d.dtype
    q_prev = torch.zeros_like(b1d)
    nrm = b1d.norm().clamp_min(1e-32)
    q = b1d / nrm

    alphas, betas, Q = [], [], [q.clone()]
    beta = torch.zeros(1, device=device, dtype=dtype)

    for _ in range(iters):
        z = A_mv_1d(q)
        alpha = (q * z).sum()
        z = z - alpha * q - beta * q_prev
        if reorth:
            for qi in Q:
                z = z - (z * qi).sum() * qi
        beta = z.norm()
        alphas.append(alpha); betas.append(beta)
        if float(beta) <= 1e-30:
            break
        q_prev, q = q, z / beta
        Q.append(q.clone())

    m = len(alphas)
    T = torch.zeros((m, m), device=device, dtype=dtype)
    T[range(m), range(m)] = torch.stack(alphas)
    if m > 1:
        off = torch.stack(betas[:m-1])
        T[range(m-1), range(1, m)] = off
        T[range(1, m), range(m-1)] = off

    evals, V = torch.linalg.eigh(T)          # tiny EVD
    f_e = f(evals)
    e1 = torch.zeros(m, device=device, dtype=dtype); e1[0] = 1.
    coeffs = V @ (f_e * (V.T @ e1))          # (m,)

    # combine only the first m Lanczos vectors
    Qmat = torch.stack(Q[:m], dim=1)         # (D, m)
    y1d  = nrm * (Qmat @ coeffs)             # (D,)
    return y1d


# -------------------------------------------------
# Power iteration (1-D) to estimate λ_max(M^T M)
# -------------------------------------------------
@torch.no_grad()
def estimate_lambda_max_MtM_1d(
    M_apply: Callable[[torch.Tensor], torch.Tensor],    # (1,C,H,W)->(1,c,h,w)
    MT_apply: Callable[[torch.Tensor], torch.Tensor],   # (1,c,h,w)->(1,C,H,W)
    hi_shape: Tuple[int, int, int, int],                # (1,C,H,W)  **N must be 1**
    iters: int = 20,
) -> float:
    assert hi_shape[0] == 1, "Pass a single sample shape (N=1). Loop outside for batches."
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype  = torch.float32

    D = int(torch.tensor(hi_shape).prod().item())
    v = torch.randn(D, device=device, dtype=dtype)
    v = v / v.norm().clamp_min(1e-32)

    def MtM_1d(x1d: torch.Tensor) -> torch.Tensor:
        x = x1d.view(*hi_shape)
        y = M_apply(x)
        z = MT_apply(y)
        return z.reshape(-1)

    lam = 0.0
    for _ in range(iters):
        w = MtM_1d(v)
        lam = float((v * w).sum().item())      # Rayleigh quotient
        nrm = w.norm().clamp_min(1e-32)
        v = w / nrm
    return max(lam, 0.0)


# ------------------------------------------------------------
# Sample z ~ N(0, Σ) with Σ = σ_s^2 (I - ρ M^T M), via Lanczos
# ------------------------------------------------------------
@torch.no_grad()
def sample_from_simplified_sigma_batched(
    M_apply: Callable[[torch.Tensor], torch.Tensor],    # (N,C,H,W)->(N,c,h,w)
    MT_apply: Callable[[torch.Tensor], torch.Tensor],   # (N,c,h,w)->(N,C,H,W)
    sigma_s: float,
    sigma_t: float,
    hi_shape: Tuple[int, int, int, int],                # (N,C,H,W), N >= 1
    *,
    xi: Optional[torch.Tensor] = None,                  # optional standard normal, shape (N,C,H,W) or (N, D)
    lanczos_iters: int = 40,
    estimate_lmax_iters: int = 20,
    safety_eps: float = 1e-3,
    device: Optional[torch.device] = None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """
    Returns z ~ N(0, Σ) with Σ = σ_s^2 (I - ρ M^T M),  ρ = σ_s^2 / σ_t^2,
    for a whole batch. Internally uses the 1-D Lanczos routine per sample.
    """
    N, C, H, W = hi_shape
    assert N >= 1, "Batch size N must be >= 1."
    device = device or (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))

    sig_s = torch.as_tensor(sigma_s, device=device, dtype=dtype)
    sig_t = torch.as_tensor(sigma_t, device=device, dtype=dtype)
    rho   = (sig_s ** 2) / (sig_t ** 2)

    D = C * H * W

    # Prepare noise ξ ~ N(0, I) per sample (flattened or NCHW both accepted)
    if xi is None:
        xi_flat = torch.randn(N, D, device=device, dtype=dtype)
    else:
        if xi.dim() == 4:
            assert xi.shape[:3] == (N, C, H), "xi must match (N,C,H,W) if 4D"
            xi_flat = xi.reshape(N, D).to(device=device, dtype=dtype)
        elif xi.dim() == 2:
            assert xi.shape == (N, D), "xi 2D must be (N, D=C*H*W)"
            xi_flat = xi.to(device=device, dtype=dtype)
        else:
            raise ValueError("xi must be None, (N,C,H,W), or (N,D).")

    # --- Build a 1-sample operator wrapper (1,C,H,W) -> (1,C,H,W) ---
    one_shape = (1, C, H, W)

    def A_mv_1d_factory(rho_eff: torch.Tensor):
        # Returns a closure A_mv_1d(v1d) that uses current rho_eff
        def A_mv_1d(v1d: torch.Tensor) -> torch.Tensor:
            x = v1d.view(*one_shape)                   # (1,C,H,W)
            y = M_apply(x)                             # (1,c,h,w)
            z = MT_apply(y)                            # (1,C,H,W)
            return v1d - rho_eff * z.reshape(-1)       # (D,)
        return A_mv_1d

    # Estimate λ_max(M^T M) ONCE (on a (1,C,H,W) shape) to clamp ρ safely
    lam_max = estimate_lambda_max_MtM_1d(M_apply, MT_apply, one_shape, iters=estimate_lmax_iters)
    if lam_max <= 0:
        lam_max = 1e-12
    rho_eff = torch.minimum(rho, (1.0 - safety_eps) / torch.as_tensor(lam_max, device=device, dtype=dtype))

    # Build the 1-D matvec once with this rho_eff
    A_mv_1d = A_mv_1d_factory(rho_eff)

    # Allocate output
    out = torch.empty(N, C, H, W, device=device, dtype=dtype)

    # Per-sample Lanczos (operator is the same for all samples)
    for i in range(N):
        y1d = lanczos_fAb_1d(A_mv_1d, xi_flat[i], f=lambda l: l.sqrt(), iters=lanczos_iters)
        out[i] = (sig_s * y1d).view(1, C, H, W)[0]

    return out

# cumulative operator
def M_0_t(ops, t, x):
    """
    Apply the cumulative operator M_{0:t} = M_t M_{t-1} ... M_0 to input x.
    Args:
        ops: List of BilinearOp instances for each time step.
        t: Time step up to which to apply the operators (1-indexed).
        x: Input tensor of shape (N, C, H_in, W_in).
    Returns:
        Tensor after applying the cumulative operator M_{0:t}.
    """
    for i in range(0, t+1):
        x = ops[i].M(x)
    return x

def M_t1_t2(ops, t1, t2, x):
    """
    Apply the cumulative operator M_{t1:t2} = M_t2 M_{t2-1} ... M_t1 to input x.
    Args:
        ops: List of BilinearOp instances for each time step.
        t1: Starting time step (inclusive, 1-indexed).
        t2: Ending time step (inclusive, 1-indexed).
        x: Input tensor of shape (N, C, H_in, W_in).
    Returns:
        Tensor after applying the cumulative operator M_{t1:t2}.
    """
    for i in range(t1, t2+1):
        x = ops[i].M(x)
    return x


class CummulativeOp:
    """
    M: bilinear resize from (H_in, W_in) -> (H_out, W_out)
    MT: adjoint (transpose) operator computed via a single VJP
    """
    def __init__(self, ops, t1, t2):
        self.H_in, self.W_in = ops[t1].H_in, ops[t1].W_in
        self.H_out, self.W_out = ops[t2].size_out
        self.ops = ops
        self.t1 = t1
        self.t2 = t2

    def M(self, x: torch.Tensor) -> torch.Tensor:

        assert x.shape[-2] == self.H_in and x.shape[-1] == self.W_in, \
            f"Input shape H,W ({x.shape[-2]},{x.shape[-1]}) does not match expected ({self.H_in},{self.W_in})"
        # x: (N,C,H_in,W_in)
        # print("M input shape:", x.shape)
        if self.H_in ==self.H_out and self.W_in == self.W_out:
            return x  # Identity
        
        return M_t1_t2(self.ops, self.t1, self.t2, x)

    def MT(self, y: torch.Tensor) -> torch.Tensor:

        assert y.shape[-2] == self.H_out and y.shape[-1] == self.W_out, \
            f"Input shape H,W ({y.shape[-2]},{y.shape[-1]}) does not match expected ({self.H_out},{self.W_out})"
        H_out, W_out = self.H_out, self.W_out
        if self.H_in ==H_out and self.W_in == W_out:
            return y  # Identity
        
        # y: (N,C,H_out,W_out)
        N, C = y.shape[:2]
        with torch.enable_grad():  # <-- ensures autograd is on even if caller used no_grad()
            x = torch.zeros((N, C, self.H_in, self.W_in),
                            device=y.device, dtype=y.dtype, requires_grad=True)
            out = self.M(x)
            # VJP: returns (∂⟨out, y⟩/∂x)
            (g,) = torch.autograd.grad(out, x, grad_outputs=y, retain_graph=False)
        return g