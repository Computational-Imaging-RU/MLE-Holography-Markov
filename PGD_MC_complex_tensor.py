import numpy as np
import torch
import time

def nll_correlation_grad_operator_MC_CGD_numpy_to_gpu_f64(
    x_np, y_np, aperture_np, std_z, num_ite_MC, alpha,
    device="cuda", tol=1e-6, max_iter=1000
):
    """
    Combined single function:
      - takes numpy (CPU) inputs: x (real), y (complex), aperture (real)
      - moves to GPU as float64 / complex128
      - runs full torch-only MC+CGD gradient
      - returns grad as numpy float64 (CPU) for minimal downstream changes

    Inputs:
      x_np:        (H,W) real numpy array (CPU)
      y_np:        (L,H,W) complex numpy array (CPU)
      aperture_np: (H,W) real numpy array (CPU)
      std_z:       float
      num_ite_MC:  int
      alpha:       float

    Returns:
      grad_np:     (H,W) numpy float64 (CPU)
      CG_iter_1:   total CG iterations spent in S^{-1} solves
      CG_iter_2:   total CG iterations spent in M^{-1} solves
    """

    # -------------------------
    # numpy -> torch (GPU)
    # -------------------------
    x_real = torch.from_numpy(np.ascontiguousarray(x_np)).to(device=device, dtype=torch.float64)
    aperture_real = torch.from_numpy(np.ascontiguousarray(aperture_np)).to(device=device, dtype=torch.float64)
    y_mul = torch.from_numpy(np.ascontiguousarray(y_np)).to(device=device, dtype=torch.complex128)

    device = x_real.device
    rdtype = x_real.dtype          # float64
    cdtype = y_mul.dtype           # complex128

    CG_iter_1, CG_iter_2 = 0, 0

    # -------------------------
    # Operators (closures)
    # -------------------------
    def A_operator(h_c):
        # A = F^{-1} M F, with shift conventions matching scipy.fft usage
        D_h = torch.fft.fftshift(torch.fft.fft2(h_c))
        MD_h = D_h * aperture_real.to(dtype=D_h.dtype)   # cast real mask -> complex
        return torch.fft.ifft2(torch.fft.ifftshift(MD_h))

    def B_operator(h_c):
        # B = A X A^H implemented as A( x * A(h) )
        Ah = A_operator(h_c)
        x_c = x_real.to(dtype=Ah.dtype)                  # cast x -> complex
        return A_operator(x_c * Ah)

    def S_operator(h_c):
        # S = B + std_z^2 I
        return B_operator(h_c) + (std_z ** 2) * h_c

    def conjugate_gradient_S_inv(b_c, x0=None):
        # Solve S u = b
        if x0 is None:
            u = torch.zeros_like(b_c, dtype=cdtype, device=device)
        else:
            u = x0.to(device=device, dtype=cdtype).clone()

        r = b_c - S_operator(u)
        p = r.clone()
        rs_old = torch.vdot(r.reshape(-1), r.reshape(-1)).real

        it = 0
        for it in range(max_iter):
            S_start_time = time.time()
            Ap = S_operator(p)
            S_end_time = time.time()
            # print(f"CG iter {it}: S_operator time = {S_end_time - S_start_time:.6f} seconds")
            denom = torch.vdot(p.reshape(-1), Ap.reshape(-1)).real
            if denom.abs() < 1e-30:
                break
            a = rs_old / denom
            u = u + a * p
            r = r - a * Ap
            rs_new = torch.vdot(r.reshape(-1), r.reshape(-1)).real
            if torch.sqrt(rs_new) < tol:
                rs_old = rs_new
                break
            beta = rs_new / rs_old
            p = r + beta * p
            rs_old = rs_new

        return u, it + 1

    def M_operator(h_c):
        # M = S - alpha^2 B S^{-1} B
        nonlocal CG_iter_1  # count inner S^{-1} work from inside M
        Sh = S_operator(h_c)
        Bh = B_operator(h_c)
        Sinv_Bh, it1 = conjugate_gradient_S_inv(Bh, x0=None)
        CG_iter_1 += it1
        BSinv_Bh = B_operator(Sinv_Bh)
        return Sh - (alpha ** 2) * BSinv_Bh

    def conjugate_gradient_M_inv(b_c, x0=None):
        # Solve M u = b
        if x0 is None:
            u = torch.zeros_like(b_c, dtype=cdtype, device=device)
        else:
            u = x0.to(device=device, dtype=cdtype).clone()

        r = b_c - M_operator(u)
        p = r.clone()
        rs_old = torch.vdot(r.reshape(-1), r.reshape(-1)).real

        it = 0
        for it in range(max_iter):
            Ap = M_operator(p)
            denom = torch.vdot(p.reshape(-1), Ap.reshape(-1)).real
            if denom.abs() < 1e-30:
                break
            a = rs_old / denom
            u = u + a * p
            r = r - a * Ap
            rs_new = torch.vdot(r.reshape(-1), r.reshape(-1)).real
            if torch.sqrt(rs_new) < tol:
                rs_old = rs_new
                break
            beta = rs_new / rs_old
            p = r + beta * p
            rs_old = rs_new

        return u, it + 1

    # -------------------------
    # Gradient from l = 1
    # -------------------------
    # diag(A^H S^{-1} A)
    ASinvA_hat_vv_sum = torch.zeros_like(x_real, dtype=cdtype, device=device)
    for _ in range(num_ite_MC):
        v = torch.randn_like(x_real, dtype=rdtype, device=device)   # real probe
        v_c = v.to(dtype=cdtype)

        Av = A_operator(v_c)
        SinvA_hat_v, it1 = conjugate_gradient_S_inv(Av, x0=None)
        CG_iter_1 += it1

        ASinvA_hat_v = A_operator(SinvA_hat_v)
        ASinvA_hat_vv_sum += ASinvA_hat_v * v_c
    ASinvA_diag = (ASinvA_hat_vv_sum.real / float(num_ite_MC)).to(dtype=rdtype)

    # |A^H S^{-1} y_1|^2
    y_1 = y_mul[0]
    Sinv_hat_y, it1 = conjugate_gradient_S_inv(y_1, x0=None)
    CG_iter_1 += it1
    ASinv_hat_y = A_operator(Sinv_hat_y)
    ASinvy_square = (ASinv_hat_y.abs() ** 2).to(dtype=rdtype)

    # -------------------------
    # Gradient from l > 1
    # -------------------------
    L = y_mul.shape[0]

    if L > 1:
        # Compute the consistent part of the gradient that doesn't depend on the specific look.
        gradient_each_look_consistent_sum = torch.zeros_like(x_real, dtype=rdtype, device=device)

        # diag(A M^{-1} A)
        AMinvA_hat_vv_sum = torch.zeros_like(x_real, dtype=cdtype, device=device)
        for _ in range(num_ite_MC):
            v = torch.randn_like(x_real, dtype=rdtype, device=device)
            v_c = v.to(dtype=cdtype)

            Av = A_operator(v_c)
            MinvA_hat_v, it2 = conjugate_gradient_M_inv(Av, x0=None)
            CG_iter_2 += it2

            AMinvA_hat_v = A_operator(MinvA_hat_v)
            AMinvA_hat_vv_sum += AMinvA_hat_v * v_c

        AMinvA_diag = (AMinvA_hat_vv_sum.real / float(num_ite_MC)).to(dtype=rdtype)
        AMinvA_diag_sum = AMinvA_diag * (L - 1)

        # diag(A S^{-1} B M^{-1} A)
        ASinvBMinvA_hat_vv_sum = torch.zeros_like(x_real, dtype=cdtype, device=device)
        for _ in range(num_ite_MC):
            v = torch.randn_like(x_real, dtype=rdtype, device=device)
            v_c = v.to(dtype=cdtype)

            Av = A_operator(v_c)
            MinvA_hat_v, it2 = conjugate_gradient_M_inv(Av, x0=None)
            CG_iter_2 += it2

            BMinvA_hat_v = B_operator(MinvA_hat_v)
            SinvBMinvA_hat_v, it1 = conjugate_gradient_S_inv(BMinvA_hat_v, x0=None)
            CG_iter_1 += it1

            ASinvBMinvA_hat_v = A_operator(SinvBMinvA_hat_v)
            ASinvBMinvA_hat_vv_sum += ASinvBMinvA_hat_v * v_c

        ASinvBMinvA_diag = (ASinvBMinvA_hat_vv_sum.real / float(num_ite_MC)).to(dtype=rdtype)
        ASinvBMinvA_diag_sum = ASinvBMinvA_diag * (L - 1)

        # diag(A S^{-1} B M^{-1} B S^{-1} A)
        ASinvBMinvBSinvA_hat_vv_sum = torch.zeros_like(x_real, dtype=cdtype, device=device)
        for _ in range(num_ite_MC):
            v = torch.randn_like(x_real, dtype=rdtype, device=device)
            v_c = v.to(dtype=cdtype)

            Av = A_operator(v_c)
            SinvA_hat_v, it1 = conjugate_gradient_S_inv(Av, x0=None)
            CG_iter_1 += it1

            BSinvA_hat_v = B_operator(SinvA_hat_v)
            MinvBSinvA_hat_v, it2 = conjugate_gradient_M_inv(BSinvA_hat_v, x0=None)
            CG_iter_2 += it2

            BMinvBSinvA_hat_v = B_operator(MinvBSinvA_hat_v)
            SinvBMinvBSinvA_hat_v, it1 = conjugate_gradient_S_inv(BMinvBSinvA_hat_v, x0=None)
            CG_iter_1 += it1

            ASinvBMinvBSinvA_hat_v = A_operator(SinvBMinvBSinvA_hat_v)
            ASinvBMinvBSinvA_hat_vv_sum += ASinvBMinvBSinvA_hat_v * v_c

        ASinvBMinvBSinvA_diag = (ASinvBMinvBSinvA_hat_vv_sum.real / float(num_ite_MC)).to(dtype=rdtype)
        ASinvBMinvBSinvA_diag_sum = ASinvBMinvBSinvA_diag * (L - 1)

        gradient_each_look_consistent_sum += AMinvA_diag_sum
        gradient_each_look_consistent_sum -= 2.0 * (alpha ** 2) * ASinvBMinvA_diag_sum
        gradient_each_look_consistent_sum += (alpha ** 2) * ASinvBMinvBSinvA_diag_sum


        # Compute the look-specific part of the gradient that depends on the particular look l.
        gradient_each_look_sum = torch.zeros_like(x_real, dtype=rdtype, device=device)

        for look_idx in range(L - 1):
            y = y_mul[look_idx + 1]
            y_prev = y_mul[look_idx]

            # r = y_l - alpha B S^{-1} y_{l-1}
            Sinv_hat_y, it1 = conjugate_gradient_S_inv(y_prev, x0=None)
            CG_iter_1 += it1
            BSinv_hat_y = B_operator(Sinv_hat_y)
            r = y - alpha * BSinv_hat_y

            # A^H M^{-1} r_l
            Minv_hat_r, it2 = conjugate_gradient_M_inv(r, x0=None)
            CG_iter_2 += it2
            AMinv_hat_r = A_operator(Minv_hat_r)

            # A^H S^{-1} B M^{-1} r_l
            BMinv_hat_r = B_operator(Minv_hat_r)
            SinvBMinv_hat_r, it1 = conjugate_gradient_S_inv(BMinv_hat_r, x0=None)
            CG_iter_1 += it1
            ASinvBMinv_hat_r = A_operator(SinvBMinv_hat_r)

            # A^H B S^{-2} y_{l-1}
            Sinv2_hat_y, it1 = conjugate_gradient_S_inv(Sinv_hat_y, x0=None)
            CG_iter_1 += it1
            BSinv2_hat_y = B_operator(Sinv2_hat_y)
            ABSinv2_hat_y = A_operator(BSinv2_hat_y)

            # A^H S^{-1} y_{l-1}
            ASinv_hat_y = A_operator(Sinv_hat_y)

            gradient_each_look_sum -= (AMinv_hat_r.abs() ** 2).to(dtype=rdtype)
            gradient_each_look_sum += (2.0 * (alpha ** 2) * (ASinvBMinv_hat_r * torch.conj(AMinv_hat_r)).real).to(dtype=rdtype)
            gradient_each_look_sum -= ((alpha ** 2) * (ASinvBMinv_hat_r.abs() ** 2)).to(dtype=rdtype)
            gradient_each_look_sum += (2.0 * alpha * (ABSinv2_hat_y * torch.conj(AMinv_hat_r)).real).to(dtype=rdtype)
            gradient_each_look_sum -= (2.0 * alpha * (ASinv_hat_y * torch.conj(AMinv_hat_r)).real).to(dtype=rdtype)

        grad_t = ASinvA_diag - ASinvy_square + gradient_each_look_consistent_sum + gradient_each_look_sum
    else:
        grad_t = ASinvA_diag - ASinvy_square

    grad_t = grad_t / float(L)

    # -------------------------
    # torch -> numpy (CPU)
    # -------------------------
    grad_np = grad_t.detach().cpu().numpy()  # float64 numpy

    return grad_np, CG_iter_1, CG_iter_2