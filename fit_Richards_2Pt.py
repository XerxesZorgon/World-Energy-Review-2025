import numpy as np
import warnings

def _S_richards(t, r, t0, nu):
    """
    Richards S-curve function: S(t) ∈ [0,1], strictly increasing for r>0.
    
    This is the normalized Richards function that approaches 0 as t→-∞ and 1 as t→+∞.
    """
    # Protect against numerical overflow in exponential
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        exp_term = np.exp(-r * (t - t0))
        # Clip extreme values to prevent overflow
        exp_term = np.clip(exp_term, 1e-15, 1e15)
        
    # Ensure nu is positive and not too small to avoid division issues
    nu_safe = np.maximum(nu, 1e-12)
    
    return 1.0 / np.power(1.0 + exp_term, 1.0 / nu_safe)

def _solve_AK_from_endpoints(S1, SN, Q1, QN):
    """
    Solve the 2-point boundary value problem for Richards curve parameters A and K.
    
    Given S(t1)=S1, S(tN)=SN and desired Q(t1)=Q1, Q(tN)=QN,
    solve: Q(t) = A + (K-A)*S(t) for A and K such that endpoints are exact.
    
    System: [[1, S1], [1, SN]] * [A, K-A]^T = [Q1, QN]^T
    Which gives: A + (K-A)*S1 = Q1 and A + (K-A)*SN = QN
    """
    den = (SN - S1)
    
    if abs(den) < 1e-12:
        # Degenerate case: S1 ≈ SN (curve is nearly flat)
        # This shouldn't happen with reasonable Richards parameters, but handle gracefully
        print(f"[WARNING] Degenerate S-curve: S1={S1:.6f}, SN={SN:.6f}")
        # Use average value for both A and K
        avg_Q = (Q1 + QN) / 2.0
        return float(avg_Q), float(avg_Q)
    
    # Solve the linear system
    # From Q1 = A + (K-A)*S1 and QN = A + (K-A)*SN
    # We get: (K-A) = (QN - Q1) / (SN - S1)
    # And: A = Q1 - (K-A)*S1 = Q1 - ((QN - Q1) / (SN - S1))*S1
    
    K_minus_A = (QN - Q1) / den
    A = Q1 - K_minus_A * S1
    K = A + K_minus_A
    
    return float(A), float(K)

def _build_series(years, r, t0, nu, Q1, QN, cum_prod):
    """
    Build cumulative discoveries, annual discoveries, and reserves series from Richards parameters.
    
    Args:
        years: Array of years
        r, t0, nu: Richards curve parameters
        Q1, QN: Endpoint values for cumulative discoveries
        cum_prod: Cumulative production array
        
    Returns:
        Tuple of (Q, D, R, A, K) where:
        - Q: cumulative discoveries (endpoints preserved exactly)
        - D: annual discoveries (non-negative)
        - R: reserves (Q - cum_prod)
        - A, K: solved affine parameters
    """
    years = np.asarray(years, dtype=float)
    cum_prod = np.asarray(cum_prod, dtype=float)
    
    # Validate inputs
    if len(years) != len(cum_prod):
        raise ValueError(f"years and cum_prod must have same length: {len(years)} vs {len(cum_prod)}")
    
    # Compute S-curve values
    S = _S_richards(years, r, t0, nu)
    S1, SN = float(S[0]), float(S[-1])
    
    # Solve for A and K to match endpoints exactly
    A, K = _solve_AK_from_endpoints(S1, SN, float(Q1), float(QN))
    
    # Build cumulative discoveries with exact endpoint preservation
    Q = A + (K - A) * S
    
    # Ensure exact endpoints (numerical precision)
    Q[0] = float(Q1)
    Q[-1] = float(QN)
    
    # Calculate annual discoveries as first differences
    D = np.zeros_like(Q)
    D[0] = Q[0]  # First year discovery
    if len(Q) > 1:
        D[1:] = np.diff(Q)  # Subsequent years as differences
    
    # Ensure non-negative annual discoveries (should be guaranteed by monotonic S-curve)
    D = np.maximum(D, 0.0)
    
    # Recompute cumulative to maintain consistency after non-negative constraint
    Q_reconstructed = np.cumsum(D)
    
    # Preserve exact endpoints after reconstruction
    if len(Q_reconstructed) > 1:
        # Scale to preserve end point
        if Q_reconstructed[-1] > 0:
            scale_factor = float(QN) / Q_reconstructed[-1]
            Q_reconstructed *= scale_factor
            D *= scale_factor
        
        # Adjust to preserve start point
        start_adjustment = float(Q1) - Q_reconstructed[0]
        Q_reconstructed += start_adjustment
        D[0] += start_adjustment
    
    # Final cumulative discoveries
    Q = Q_reconstructed
    
    # Calculate reserves
    R = Q - cum_prod
    
    return Q, D, R, A, K

def tune_richards_preserve_endpoints(
    years,
    cum_disc,         # cumulative discoveries at those years
    cum_prod,         # cumulative production at those years
    t0_bounds=(1960.0, 1970.0),
    r_bounds=(1e-3, 5e-1),
    nu_bounds=(1e-1, 10.0),
    smooth_weight=1e-6,
    do_local_refine=True,
    verbose=False,
):
    """
    Find Richards curve parameters (r,t0,nu) that minimize negative reserves 
    while preserving exact endpoint values.
    
    This function uses a 2-point boundary value approach where the Richards curve
    Q(t) = A + (K-A)*S(t) is constrained to pass exactly through the first and
    last points of the cumulative discovery data.
    
    Args:
        years: Array of years (must be monotonically increasing)
        cum_disc: Cumulative discoveries at those years
        cum_prod: Cumulative production at those years  
        t0_bounds: (min, max) bounds for inflection point t0
        r_bounds: (min, max) bounds for growth rate r (must be positive)
        nu_bounds: (min, max) bounds for shape parameter nu (must be positive)
        smooth_weight: Penalty weight for smoothness of annual discoveries (≥0)
        do_local_refine: Whether to use SciPy local optimization after grid search
        verbose: Whether to print progress information
        
    Returns:
        dict with keys:
        - params: {'A': A, 'K': K, 'r': r, 't0': t0, 'nu': nu} - fitted parameters
        - cumulative: np.array - cumulative discoveries (endpoints exact)
        - annual: np.array - annual discoveries (non-negative)  
        - reserves: np.array - reserves (cumulative - production)
        - negatives: {'count': int, 'min_reserve': float} - negative reserve stats
        
    Raises:
        ValueError: If input arrays have different lengths or invalid bounds
        AssertionError: If input validation fails
    """
    # Input validation and conversion
    years = np.asarray(years, dtype=float)
    Qobs = np.asarray(cum_disc, dtype=float)
    Pcum = np.asarray(cum_prod, dtype=float)
    
    # Validate array dimensions and lengths
    if years.ndim != 1 or Qobs.ndim != 1 or Pcum.ndim != 1:
        raise ValueError("All input arrays must be 1-dimensional")
    
    if not (years.size == Qobs.size == Pcum.size):
        raise ValueError(f"All arrays must have same length: years={years.size}, cum_disc={Qobs.size}, cum_prod={Pcum.size}")
    
    if years.size < 3:
        raise ValueError("Need at least 3 data points for Richards curve fitting")
    
    # Validate data quality
    if not np.all(np.isfinite(years)) or not np.all(np.isfinite(Qobs)) or not np.all(np.isfinite(Pcum)):
        raise ValueError("Input data contains NaN or infinite values")
    
    if not np.all(np.diff(years) > 0):
        raise ValueError("Years must be strictly monotonically increasing")
    
    if np.any(Qobs < 0) or np.any(Pcum < 0):
        raise ValueError("Cumulative discoveries and production must be non-negative")
    
    # Validate parameter bounds
    if (t0_bounds[0] >= t0_bounds[1] or r_bounds[0] >= r_bounds[1] or 
        nu_bounds[0] >= nu_bounds[1]):
        raise ValueError("All parameter bounds must have lower < upper")
    
    if r_bounds[0] <= 0 or nu_bounds[0] <= 0:
        raise ValueError("r_bounds and nu_bounds must be positive")
    
    if smooth_weight < 0:
        raise ValueError("smooth_weight must be non-negative")

    Q1, QN = float(Qobs[0]), float(Qobs[-1])
    
    if verbose:
        print(f"[init] Fitting Richards curve to {len(years)} points from {years[0]:.0f} to {years[-1]:.0f}")
        print(f"[init] Discoveries: {Q1:.1f} → {QN:.1f} EJ (endpoints will be preserved exactly)")
        print(f"[init] Production total: {Pcum[-1]:.1f} EJ")
        print(f"[init] Current reserves: {QN - Pcum[-1]:.1f} EJ")

    # ----- coarse grid search (robust) -----
    r_grid = np.geomspace(max(r_bounds[0], 1e-3), r_bounds[1], 12)
    t0_grid = np.linspace(t0_bounds[0], t0_bounds[1], 21)
    nu_grid = np.geomspace(nu_bounds[0], nu_bounds[1], 12)

    best = {"neg": 10**9, "loss": np.inf, "r": None, "t0": None, "nu": None,
            "A": None, "K": None, "Q": None, "D": None, "R": None, "minR": None}

    def score(r, t0, nu):
        """Score function for parameter evaluation."""
        try:
            Q, D, R, A, K = _build_series(years, r, t0, nu, Q1, QN, Pcum)
            
            # Primary objective: minimize negative reserves
            neg_reserves = np.clip(-R, 0.0, None)
            loss = float(neg_reserves @ neg_reserves)
            
            # Secondary objective: smoothness penalty on annual discoveries
            if smooth_weight > 0 and len(D) > 2:
                d2 = np.diff(D, n=2)
                loss += float(smooth_weight * (d2 @ d2))
            
            neg_count = int((R < 0).sum())
            min_reserve = float(R.min()) if len(R) > 0 else 0.0
            
            return Q, D, R, A, K, neg_count, min_reserve, loss
            
        except Exception as e:
            if verbose:
                print(f"[grid] Error at r={r:.4f}, t0={t0:.1f}, nu={nu:.2f}: {e}")
            # Return worst possible score for failed evaluations
            return None, None, None, None, None, 10**9, -10**9, np.inf

    total_combinations = len(r_grid) * len(t0_grid) * len(nu_grid)
    if verbose:
        print(f"[grid] Starting grid search over {total_combinations} parameter combinations")
        print(f"[grid] Parameter ranges: r=[{r_bounds[0]:.4f}, {r_bounds[1]:.4f}], "
              f"t0=[{t0_bounds[0]:.1f}, {t0_bounds[1]:.1f}], nu=[{nu_bounds[0]:.2f}, {nu_bounds[1]:.2f}]")

    combinations_tested = 0
    successful_evaluations = 0
    
    for t0 in t0_grid:
        for r in r_grid:
            for nu in nu_grid:
                combinations_tested += 1
                Q, D, R, A, K, nneg, minR, loss = score(r, t0, nu)
                
                if Q is not None:  # Successful evaluation
                    successful_evaluations += 1
                    better = (nneg < best["neg"]) or (nneg == best["neg"] and loss < best["loss"])
                    if better:
                        best.update(dict(neg=nneg, loss=loss, r=r, t0=t0, nu=nu, A=A, K=K,
                                         Q=Q, D=D, R=R, minR=minR))
                        
                        # Early exit if we find perfect solution
                        if nneg == 0 and loss < 1e-10:
                            if verbose:
                                print(f"[grid] Perfect solution found at r={r:.4f}, t0={t0:.1f}, nu={nu:.2f}")
                            break
            else:
                continue
            break
        else:
            continue
        break
            
        if verbose and combinations_tested % len(r_grid) == 0:
            print(f"[grid] t0={t0:.2f} → best: {best['neg']} negatives, loss={best['loss']:.4g}, min_reserve={best['minR']:.2f}")

    if verbose:
        print(f"[grid] Completed {combinations_tested} combinations ({successful_evaluations} successful)")
        if best["r"] is not None:
            print(f"[grid] Best parameters: r={best['r']:.4f}, t0={best['t0']:.1f}, nu={best['nu']:.2f}")
            print(f"[grid] Best result: {best['neg']} negatives, min_reserve={best['minR']:.2f} EJ")
        else:
            print("[grid] WARNING: No successful parameter combinations found")

    # Check if we found any valid solution
    if best["r"] is None:
        raise RuntimeError("Grid search failed to find any valid parameter combinations. "
                         "Try widening parameter bounds or checking input data quality.")
    
    # Early exit if zero negatives or local refine disabled
    if best["neg"] == 0 or not do_local_refine:
        if verbose:
            print(f"[result] Grid search complete: {best['neg']} negatives, min_reserve={best['minR']:.2f} EJ")
        return {
            "params": {"A": float(best["A"]), "K": float(best["K"]), 
                      "r": float(best["r"]), "t0": float(best["t0"]), "nu": float(best["nu"])},
            "cumulative": best["Q"], "annual": best["D"], "reserves": best["R"],
            "negatives": {"count": int(best["neg"]), "min_reserve": float(best["minR"])}
        }

    # ----- optional local refine (SciPy) -----
    try:
        from scipy.optimize import minimize, Bounds

        def obj(x):
            rr  = float(np.clip(x[0], *r_bounds))
            tt0 = float(np.clip(x[1], *t0_bounds))
            nnu = float(np.clip(x[2], *nu_bounds))
            Q,D,R,A,K, nneg, _, loss = score(rr, tt0, nnu)
            # penalize any negatives heavily to push to zero
            return loss + 1e3 * (nneg>0)

        x0 = np.array([best["r"], best["t0"], best["nu"]], dtype=float)
        bounds = Bounds([r_bounds[0], t0_bounds[0], nu_bounds[0]],
                        [r_bounds[1], t0_bounds[1], nu_bounds[1]])
        res = minimize(obj, x0, method="L-BFGS-B", bounds=bounds, options={"maxiter": 500})
        rr, tt0, nnu = res.x
        Q,D,R,A,K, nneg, minR, _ = score(rr, tt0, nnu)
        if (nneg < best["neg"]) or (nneg == best["neg"] and R.var() < best["R"].var()):
            best.update(dict(r=rr, t0=tt0, nu=nnu, Q=Q, D=D, R=R, A=A, K=K, neg=nneg, minR=minR))
        if verbose:
            print(f"[refine] neg={best['neg']}, minR={best['minR']:.3f}, "
                  f"r={best['r']:.4f}, t0={best['t0']:.2f}, nu={best['nu']:.3f}")
    except Exception as e:
        if verbose:
            print(f"[refine] skipped: {e}")

    return {
        "params": {"A": float(best["A"]), "K": float(best["K"]),
                   "r": float(best["r"]), "t0": float(best["t0"]), "nu": float(best["nu"])},
        "cumulative": best["Q"], "annual": best["D"], "reserves": best["R"],
        "negatives": {"count": int(best["neg"]), "min_reserve": float(best["minR"])},
    }
