#!/usr/bin/env python3
"""
Verification Script: Ray-level Gradients vs Plain LM in Optical Design
---------------------------------------------------------------------
Demonstrates:
1. Why each parameter affects all rays, but affects each ray differently.
2. Why Plain LM (scalar RMS residual) gives a Rank-1 degenerate Hessian.
3. Why Per-Ray LM (TRA residual) yields a Full-Rank Gauss-Newton Hessian.
4. The exact effect of number of rays (M) on Hessian rank, condition number,
   and optical parameter aliasing.
5. Optimization convergence comparison: GD vs Naive LM vs Per-Ray LM.
"""

import numpy as np

# Set random seed for reproducibility
np.random.seed(42)

def optical_system(params, rho):
    """
    Parametric optical model for transverse ray aberration (spot displacement)
    on the focal plane as a function of pupil coordinate rho in [-1, 1].
    
    Parameters p = [p0, p1, p2, p3, p4]:
      p0: Image shift (tilt / prism)        -> contribution ~ 1.0
      p1: Defocus (mirror curvature error)  -> contribution ~ rho
      p2: Astigmatism / quadratic freeform  -> contribution ~ rho^2
      p3: Coma / cubic freeform term        -> contribution ~ rho^3
      p4: Spherical aberration / 4th order  -> contribution ~ rho^4
    """
    p = np.asarray(params)
    # Landing position error on detector for ray at pupil coordinate rho
    # r_i = p0 + p1*rho + p2*rho^2 + p3*rho^3 + p4*rho^4
    r = p[0] + p[1]*rho + p[2]*(rho**2) + p[3]*(rho**3) + p[4]*(rho**4)
    return r

def compute_jacobians(params, pupil_coords):
    """
    Computes:
    1. Per-ray Jacobian J_ray (M x N): d(r_i) / d(p_j)
    2. Scalar Jacobian J_scalar (1 x N): d(RMS) / d(p_j)
    """
    M = len(pupil_coords)
    N = len(params)
    
    # 1. Per-ray Jacobian J_ray: shape (M, N)
    # J_ray[i, j] = d(r_i) / d(p_j)
    J_ray = np.zeros((M, N))
    for j in range(N):
        J_ray[:, j] = pupil_coords**j
        
    # Ray residuals
    r = optical_system(params, pupil_coords)
    rms = np.sqrt(np.mean(r**2))
    
    # 2. Scalar Jacobian: gradient of aggregate RMS spot size
    # RMS = sqrt( (1/M) sum r_i^2 )
    # d(RMS)/d(p_j) = (1 / (M * RMS)) sum_i r_i * d(r_i)/d(p_j)
    #               = (r^T @ J_ray) / (M * RMS)
    if rms > 1e-12:
        grad_rms = (r @ J_ray) / (M * rms)
    else:
        grad_rms = np.zeros(N)
        
    J_scalar = grad_rms.reshape(1, N) # Shape (1, N)
    
    return r, J_ray, J_scalar, rms

def run_verification():
    print("=" * 75)
    print("OPTICAL REFINEMENT: PER-RAY GRADIENTS VS PLAIN LM ANALYSIS")
    print("=" * 75)
    
    # True target is [0, 0, 0, 0, 0] (perfect sharp focus).
    # Initial aberration state with defocus, coma, spherical:
    initial_params = np.array([0.15, -0.40, 0.25, -0.30, 0.50])
    N = len(initial_params)
    print(f"\nInitial Optical Parameters (N = {N}):")
    print(f"  p0 (Tilt/Shift):    {initial_params[0]:+.3f}")
    print(f"  p1 (Defocus):       {initial_params[1]:+.3f}")
    print(f"  p2 (Astigmatism):   {initial_params[2]:+.3f}")
    print(f"  p3 (Coma):          {initial_params[3]:+.3f}")
    print(f"  p4 (Spherical):     {initial_params[4]:+.3f}")

    # -------------------------------------------------------------
    # 1. Why each parameter affects all rays, but affects each ray differently
    # -------------------------------------------------------------
    print("\n" + "-" * 75)
    print("1. PER-RAY SENSITIVITY: How does 1 parameter affect each ray differently?")
    print("-" * 75)
    sample_rays = np.array([-1.0, -0.5, 0.0, 0.5, 1.0])
    r_sample, J_sample, _, _ = compute_jacobians(initial_params, sample_rays)
    
    print(f"{'Pupil Coordinate rho':<22} | {'d(ray)/dp1 (Defocus)':<20} | {'d(ray)/dp4 (Spherical)':<22}")
    print("-" * 75)
    for i, rho in enumerate(sample_rays):
        ray_desc = "Bottom Marginal" if rho == -1.0 else ("Chief Ray" if rho == 0.0 else ("Top Marginal" if rho == 1.0 else "Zonal Ray"))
        print(f"rho = {rho:+.1f} ({ray_desc:<15}) | d(r)/dp1 = {J_sample[i, 1]:+.4f}       | d(r)/dp4 = {J_sample[i, 4]:+.4f}")
    print("\nInsight: Every parameter alters ALL rays, but because rays strike different")
    print("local mirror coordinates, each ray experiences a completely different derivative!")

    # -------------------------------------------------------------
    # 2. Plain LM vs Per-Ray LM: Rank and Eigenvalues of J^T J
    # -------------------------------------------------------------
    print("\n" + "-" * 75)
    print("2. HESSIAN APPROXIMATION: Plain LM (Rank 1) vs Per-Ray LM (Full Rank)")
    print("-" * 75)
    
    M_dense = 25
    pupil_dense = np.linspace(-1, 1, M_dense)
    r_dense, J_ray_dense, J_scalar_dense, rms_dense = compute_jacobians(initial_params, pupil_dense)
    
    # Gauss-Newton Hessians
    H_ray = (J_ray_dense.T @ J_ray_dense) / M_dense
    H_scalar = J_scalar_dense.T @ J_scalar_dense
    
    rank_ray = np.linalg.matrix_rank(H_ray)
    rank_scalar = np.linalg.matrix_rank(H_scalar)
    
    eig_ray = np.sort(np.linalg.eigvalsh(H_ray))[::-1]
    eig_scalar = np.sort(np.linalg.eigvalsh(H_scalar))[::-1]
    
    print(f"Number of parameters N: {N}")
    print(f"Number of traced rays M: {M_dense}")
    print(f"\n[A] Plain LM (Scalar Residual):")
    print(f"  Jacobian shape:    {J_scalar_dense.shape} (1 row only!)")
    print(f"  Hessian (J^T J):   Rank = {rank_scalar} / {N}")
    print(f"  Eigenvalues:       {[f'{e:.5f}' for e in eig_scalar]}")
    print(f"  ==> {N - rank_scalar} zero eigenvalues! BLIND in {N - rank_scalar} parameter directions!")
    
    print(f"\n[B] Per-Ray LM (TRA Residual):")
    print(f"  Jacobian shape:    {J_ray_dense.shape} ({M_dense} rows)")
    print(f"  Hessian (J^T J):   Rank = {rank_ray} / {N} (FULL RANK)")
    print(f"  Eigenvalues:       {[f'{e:.5f}' for e in eig_ray]}")
    print(f"  Condition Number:  {eig_ray[0] / eig_ray[-1]:.2f}")
    print(f"  ==> Captures complete multi-dimensional optical curvature coupling!")

    # -------------------------------------------------------------
    # 3. Effect of Number of Rays (M)
    # -------------------------------------------------------------
    print("\n" + "-" * 75)
    print("3. EFFECT OF NUMBER OF RAYS (M) ON HESSIAN RANK & STABILITY")
    print("-" * 75)
    print(f"{'Num Rays (M)':<14} | {'Hessian Rank':<14} | {'Min Eigenvalue':<16} | {'Condition Number':<18} | {'Status'}")
    print("-" * 75)
    
    for M in [2, 3, 4, 5, 8, 16, 32, 64]:
        pupil_m = np.linspace(-1, 1, M)
        _, J_m, _, _ = compute_jacobians(initial_params, pupil_m)
        H_m = (J_m.T @ J_m) / M
        rank_m = np.linalg.matrix_rank(H_m)
        eigs_m = np.sort(np.linalg.eigvalsh(H_m))[::-1]
        min_eig = max(eigs_m[-1], 1e-16)
        cond_m = eigs_m[0] / min_eig
        
        if rank_m < N:
            status = f"RANK DEFICIENT! ({N - rank_m} blind null-modes)"
        elif cond_m > 1e5:
            status = "Ill-conditioned (Aliasing risk)"
        else:
            status = "Full-rank & Well-conditioned"
            
        print(f"{M:<14} | {rank_m}/{N:<12} | {min_eig:<16.2e} | {cond_m:<18.2e} | {status}")

    # -------------------------------------------------------------
    # 4. Optimization Race: GD vs Plain LM vs Per-Ray LM
    # -------------------------------------------------------------
    print("\n" + "-" * 75)
    print("4. OPTIMIZATION RACE: Convergence over 15 iterations (M = 30 rays)")
    print("-" * 75)
    
    pupil_race = np.linspace(-1, 1, 30)
    
    def run_opt(method, n_iters=15):
        p = initial_params.copy()
        history = []
        damp = 1e-3
        lr = 0.15
        for it in range(n_iters):
            r, J_ray, J_scalar, rms = compute_jacobians(p, pupil_race)
            history.append(rms)
            
            if method == 'GD':
                # Gradient descent on scalar loss: grad = (r @ J_ray) / len(r)
                grad = (r @ J_ray) / len(r)
                p = p - lr * grad
            elif method == 'Naive-LM':
                # Plain LM using 1D scalar residual: J_scalar (1 x N)
                # update = - (J_s^T J_s + damp*I)^(-1) (J_s^T * residual)
                # where residual = rms
                H = J_scalar.T @ J_scalar
                rhs = J_scalar.T * rms
                step = np.linalg.solve(H + damp * np.eye(N), rhs).flatten()
                p = p - step
            elif method == 'Per-Ray-LM':
                # Per-ray LM using M residuals: J_ray (M x N)
                # update = - (J_ray^T J_ray + damp*I)^(-1) (J_ray^T @ r)
                H = J_ray.T @ J_ray
                rhs = J_ray.T @ r
                step = np.linalg.solve(H + damp * np.eye(N), rhs)
                p = p - step
        return history

    hist_gd = run_opt('GD')
    hist_naive = run_opt('Naive-LM')
    hist_ray = run_opt('Per-Ray-LM')
    
    print(f"{'Iteration':<10} | {'Gradient Descent RMS':<22} | {'Plain LM RMS':<20} | {'Per-Ray LM RMS':<20}")
    print("-" * 75)
    for it in range(len(hist_gd)):
        print(f"{it:<10} | {hist_gd[it]:<22.6f} | {hist_naive[it]:<20.6f} | {hist_ray[it]:<20.6f}")
        
    print("\nConclusion:")
    print(f"  Gradient Descent final RMS: {hist_gd[-1]:.6f} (Slow crawl)")
    print(f"  Plain LM final RMS:         {hist_naive[-1]:.6f} (Stalled by Rank-1 Hessian)")
    print(f"  Per-Ray LM final RMS:       {hist_ray[-1]:.6f} (Instant quadratic convergence)")
    print("=" * 75)

if __name__ == '__main__':
    run_verification()
