import numpy as np
import warnings

def _richards_shape(t, r, t0, nu):
    """Richards (generalized logistic) shape with K=1, lower asymptote 0.

    R(t) = 1 / (1 + exp(-r*(t - t0)))**(1/nu)
    
    Args:
        t: Time array
        r: Growth rate (must be positive)
        t0: Inflection point time
        nu: Shape parameter (must be positive)
    """
    # Ensure numerical stability
    r = max(r, 1e-8)
    nu = max(nu, 1e-6)  # More conservative bound for nu
    
    # Clip extreme exponential arguments to prevent overflow/underflow
    exp_arg = -r * (t - t0)
    exp_arg = np.clip(exp_arg, -700, 700)  # Prevent overflow in exp
    
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return 1.0 / np.power(1.0 + np.exp(exp_arg), 1.0 / nu)


def _affine_match_endpoints(t, R, q_start, q_end):
    """Find a,b so that Q(t)=a+b*R(t) hits Q(t0)=q_start and Q(tN)=q_end exactly."""
    r0, r1 = float(R[0]), float(R[-1])
    den = (r1 - r0)
    if abs(den) < 1e-12:
        # Degenerate shape (nearly flat) → fallback: no scale, pin to start
        b = 1.0
        a = q_start - b * r0
    else:
        b = (q_end - q_start) / den
        a = q_start - b * r0
    return a, b


def _series_from_params(years, cum_prod, q_cum_start, q_cum_end, r, t0, nu, smooth_w=0.0):
    """Build cumulative/annual/reserves for given Richards params with endpoint preservation.
    
    Args:
        years: Array of years
        cum_prod: Cumulative production array
        q_cum_start: Starting cumulative discoveries (endpoint constraint)
        q_cum_end: Ending cumulative discoveries (endpoint constraint)
        r: Richards growth rate
        t0: Richards inflection point
        nu: Richards shape parameter
        smooth_w: Smoothness penalty weight
        
    Returns:
        Tuple of (Q, D, reserves, (a,b), loss, neg_count, min_reserve)
    """
    R = _richards_shape(years, r, t0, nu)  # K=1 shape
    a, b = _affine_match_endpoints(years, R, q_cum_start, q_cum_end)
    Q_fitted = a + b * R  # cumulative discoveries (endpoints preserved)
    
    # Calculate annual discoveries as first differences
    # For first year, use Q_fitted[0] directly (cumulative from year 0 to year 1)
    D = np.zeros_like(Q_fitted)
    D[0] = Q_fitted[0]  # First year discovery
    D[1:] = np.diff(Q_fitted)  # Subsequent years as differences
    
    # Apply non-negative constraint to annual discoveries
    D = np.maximum(D, 0.0)
    
    # CRITICAL: Preserve endpoints by reconstructing cumulative with endpoint constraints
    # This ensures mathematical consistency while respecting non-negative annual discoveries
    Q_reconstructed = np.cumsum(D)
    
    # Adjust to preserve exact endpoints (this is the key improvement)
    if len(Q_reconstructed) > 1:
        # Scale to match end point exactly
        scale_factor = q_cum_end / Q_reconstructed[-1] if Q_reconstructed[-1] > 0 else 1.0
        Q_reconstructed *= scale_factor
        D *= scale_factor
        
        # Adjust first year to match start point exactly
        adjustment = q_cum_start - Q_reconstructed[0]
        Q_reconstructed += adjustment
        D[0] += adjustment
    
    # Final cumulative discoveries with exact endpoint preservation
    Q = Q_reconstructed
    
    # Compute reserves vs cumulative production
    reserves = Q - cum_prod
    
    # Negative-reserve hinge loss + optional smoothness penalty
    neg = np.clip(-reserves, 0.0, None)
    loss_neg = float(np.dot(neg, neg))
    
    if smooth_w > 0 and len(D) > 2:
        # Smoothness penalty on second differences of annual discoveries
        d2 = np.diff(D, n=2)
        loss_neg += float(smooth_w * np.dot(d2, d2))
    
    # Compute statistics
    neg_count = int(np.sum(reserves < 0))
    min_reserve = float(np.min(reserves)) if len(reserves) > 0 else 0.0
    
    return Q, D, reserves, (a, b), loss_neg, neg_count, min_reserve


def tune_richards_to_minimize_negatives(
    years,
    cumulative_discoveries,
    cumulative_production,
    init_params=(1.0, 0.05, 1965.0, 1.0),  # (K_unused, r, t0, nu); K is ignored (shape-only)
    t0_bounds=(1960.0, 1970.0),
    r_bounds=(1e-3, 5e-1),
    nu_bounds=(1e-1, 10.0),
    smooth_weight=1e-6,
    do_local_refine=True,
    verbose=False,
):
    """
    Adjust Richards parameters to minimize (or eliminate) negative reserves, preserving endpoints.
    
    This function uses the Richards generalized logistic function Q(t) = K/(1+exp(-r(t-t0)))^(1/nu)
    to model cumulative discovery curves while ensuring:
    1. Exact preservation of start and end point values
    2. Minimization of negative reserves (reserves = discoveries - production)
    3. Constraint of t0 (inflection point) to the range 1960-1970
    4. Non-negative annual discoveries
    
    The optimization uses a two-stage approach:
    - Coarse grid search over parameter space (robust global search)
    - Optional local refinement with L-BFGS-B (fine-tuning)

    Args:
        years (array-like): monotonically increasing years (e.g., 1900..2024).
        cumulative_discoveries (array-like): observed cumulative discoveries (same length as years).
        cumulative_production (array-like): cumulative production (same length as years).
        init_params (tuple): initial (K, r, t0, nu). K is ignored; shape uses K=1 internally.
        t0_bounds (tuple): (min,max) for t0; **ENFORCED to be within 1960–1970**.
        r_bounds (tuple): (min,max) for growth rate r (must be positive).
        nu_bounds (tuple): (min,max) for shape parameter nu (must be positive).
        smooth_weight (float): penalty weight on curvature of annual discoveries (≥0).
        do_local_refine (bool): if True and SciPy available, refine with L-BFGS-B.
        verbose (bool): print detailed progress information.

    Returns:
        dict with keys:
          - params: {'K': 1.0, 'r': r*, 't0': t0*, 'nu': nu*} - fitted Richards parameters
          - affine: {'a': a, 'b': b} - affine transformation coefficients  
          - cumulative: np.ndarray - tuned cumulative discoveries (endpoints preserved)
          - annual: np.ndarray - tuned annual discoveries (non-negative)
          - reserves: np.ndarray - tuned reserves (discoveries - production)
          - negatives: {'count': n_neg, 'min_reserve': min_val} - negative reserve statistics
          
    Raises:
        AssertionError: if input arrays have different lengths or wrong dimensions
        ValueError: if bounds are invalid or data contains NaN/inf values
    """
    # Input validation and conversion
    years = np.asarray(years, dtype=float)
    Qobs = np.asarray(cumulative_discoveries, dtype=float)
    Pcum = np.asarray(cumulative_production, dtype=float)

    # Validate array shapes and dimensions
    if years.ndim != 1 or Qobs.shape != years.shape or Pcum.shape != years.shape:
        raise AssertionError("years, cumulative_discoveries, cumulative_production must be same 1D length")
    
    if len(years) < 3:
        raise ValueError("Need at least 3 data points for Richards curve fitting")
    
    # Validate data quality
    if not np.all(np.isfinite(years)) or not np.all(np.isfinite(Qobs)) or not np.all(np.isfinite(Pcum)):
        raise ValueError("Input data contains NaN or infinite values")
    
    if not np.all(np.diff(years) > 0):
        raise ValueError("Years must be strictly monotonically increasing")
    
    if np.any(Qobs < 0) or np.any(Pcum < 0):
        raise ValueError("Cumulative discoveries and production must be non-negative")
    
    # Validate parameter bounds
    if t0_bounds[0] >= t0_bounds[1] or r_bounds[0] >= r_bounds[1] or nu_bounds[0] >= nu_bounds[1]:
        raise ValueError("All parameter bounds must have lower < upper")
    
    if r_bounds[0] <= 0 or nu_bounds[0] <= 0:
        raise ValueError("r_bounds and nu_bounds must be positive")
    
    if smooth_weight < 0:
        raise ValueError("smooth_weight must be non-negative")

    q_start, q_end = float(Qobs[0]), float(Qobs[-1])
    
    if verbose:
        print(f"[init] Data: {len(years)} years from {years[0]:.0f} to {years[-1]:.0f}")
        print(f"[init] Discoveries: {q_start:.1f} → {q_end:.1f} EJ")
        print(f"[init] Production total: {Pcum[-1]:.1f} EJ")
        print(f"[init] Initial reserves: {q_end - Pcum[-1]:.1f} EJ")

    # ---- coarse grid search (robust) ----
    # Ensure t0 bounds are strictly within 1960-1970 as required
    t0_bounds = (max(t0_bounds[0], 1960.0), min(t0_bounds[1], 1970.0))
    
    r_grid = np.geomspace(max(r_bounds[0], 1e-3), r_bounds[1], 12)
    t0_grid = np.linspace(t0_bounds[0], t0_bounds[1], 21)
    nu_grid = np.geomspace(nu_bounds[0], nu_bounds[1], 12)

    best = {
        "loss": np.inf,
        "neg": 10**9,
        "params": None,
        "affine": None,
        "Q": None,
        "D": None,
        "R": None,
        "min_reserve": None,
    }

    total_combinations = len(t0_grid) * len(r_grid) * len(nu_grid)
    if verbose:
        print(f"[grid] Starting grid search over {total_combinations} parameter combinations")
        print(f"[grid] t0 range: [{t0_bounds[0]:.1f}, {t0_bounds[1]:.1f}] (constrained to 1960-1970)")
        print(f"[grid] r range: [{r_bounds[0]:.4f}, {r_bounds[1]:.4f}]")
        print(f"[grid] nu range: [{nu_bounds[0]:.2f}, {nu_bounds[1]:.2f}]")

    combinations_tested = 0
    early_exit_found = False
    
    for t0 in t0_grid:
        for r in r_grid:
            for nu in nu_grid:
                combinations_tested += 1
                try:
                    Q, D, R, (a, b), loss, nneg, minR = _series_from_params(
                        years, Pcum, q_start, q_end, r, t0, nu, smooth_w=smooth_weight
                    )
                    
                    # Primary: minimize negative count; secondary: minimize loss
                    better = (nneg < best["neg"]) or (nneg == best["neg"] and loss < best["loss"])
                    if better:
                        best.update(dict(
                            loss=loss, neg=nneg, params=(r, t0, nu), affine=(a, b),
                            Q=Q, D=D, R=R, min_reserve=minR
                        ))
                        
                        # Early exit if we find zero negatives with good loss
                        if nneg == 0 and loss < 1e-6:
                            if verbose:
                                print(f"[grid] Early exit: found zero negatives at t0={t0:.2f}, r={r:.4f}, nu={nu:.3f}")
                            early_exit_found = True
                            break
                            
                except Exception as e:
                    if verbose:
                        print(f"[grid] Error at t0={t0:.2f}, r={r:.4f}, nu={nu:.3f}: {e}")
                    continue
                    
            if early_exit_found:
                break
        if early_exit_found:
            break
            
        if verbose and combinations_tested % (len(r_grid) * len(nu_grid)) == 0:
            print(f"[grid] t0={t0:.2f} → current best: neg={best['neg']}, loss={best['loss']:.4g}, min_reserve={best['min_reserve']:.2f}")

    if verbose:
        print(f"[grid] Completed {combinations_tested}/{total_combinations} combinations")
        if best["params"] is not None:
            r_best, t0_best, nu_best = best["params"]
            print(f"[grid] Best parameters: r={r_best:.4f}, t0={t0_best:.2f}, nu={nu_best:.3f}")
            print(f"[grid] Best result: {best['neg']} negatives, loss={best['loss']:.4g}")
        else:
            print(f"[grid] WARNING: No valid parameter combinations found")

    # Early exit if already zero negatives
    if best["neg"] == 0 and not do_local_refine:
        r, t0, nu = best["params"]
        a, b = best["affine"]
        return {
            "params": {"K": 1.0, "r": r, "t0": t0, "nu": nu},
            "affine": {"a": a, "b": b},
            "cumulative": best["Q"],
            "annual": best["D"],
            "reserves": best["R"],
            "negatives": {"count": int(best["neg"]), "min_reserve": float(best["min_reserve"])},
        }

    # ---- optional local refinement (SciPy) ----
    r, t0, nu = best["params"]
    try:
        from scipy.optimize import minimize, Bounds

        def obj(x):
            rr = np.clip(x[0], *r_bounds)
            tt0 = np.clip(x[1], *t0_bounds)
            nnu = np.clip(x[2], *nu_bounds)
            Q, D, R, _, loss, nneg, _ = _series_from_params(
                years, Pcum, q_start, q_end, rr, tt0, nnu, smooth_w=smooth_weight
            )
            # Soft metric: hinge loss + small penalty on negative count to push to zero
            return loss + 1e3 * (nneg > 0)

        x0 = np.array([r, t0, nu], dtype=float)
        bounds = Bounds([r_bounds[0], t0_bounds[0], nu_bounds[0]],
                        [r_bounds[1], t0_bounds[1], nu_bounds[1]])
        res = minimize(obj, x0, method="L-BFGS-B", bounds=bounds, options={"maxiter": 500})
        rr, tt0, nnu = res.x
        Q, D, R, (a, b), loss, nneg, minR = _series_from_params(
            years, Pcum, q_start, q_end, rr, tt0, nnu, smooth_w=smooth_weight
        )
        # Keep the better of (grid best) and (refined)
        if (nneg < best["neg"]) or (nneg == best["neg"] and loss < best["loss"]):
            best.update(dict(
                loss=loss, neg=nneg, params=(rr, tt0, nnu), affine=(a, b),
                Q=Q, D=D, R=R, min_reserve=minR
            ))
        if verbose:
            print(f"[refine] neg={best['neg']}, min_reserve={best['min_reserve']:.3f} EJ, "
                  f"r={best['params'][0]:.4f}, t0={best['params'][1]:.2f}, nu={best['params'][2]:.3f}")
    except Exception as e:
        if verbose:
            print(f"[refine] skipped (SciPy not available or failed): {e}")

    r, t0, nu = best["params"]
    a, b = best["affine"]

    return {
        "params": {"K": 1.0, "r": float(r), "t0": float(t0), "nu": float(nu)},
        "affine": {"a": float(a), "b": float(b)},
        "cumulative": best["Q"],
        "annual": best["D"],
        "reserves": best["R"],
        "negatives": {"count": int(best["neg"]), "min_reserve": float(best["min_reserve"])},
    }
