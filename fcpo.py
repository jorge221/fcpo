import numpy as np
from numba import njit
from joblib import Parallel, delayed, cpu_count
from scipy.stats import qmc
from scipy.spatial.distance import pdist

"""Feline-inspired constrained population optimizer.

The core algorithm combines PSO-style velocity updates with behavior-state
transitions and an eigenvector-aligned local model extracted from elite points.
"""

# -----------------------------------------------------------------------------
# 1. Initialization Strategy (Latin Hypercube Sampling)
# -----------------------------------------------------------------------------
def _lhs_init(P, S, lower_bounds, upper_bounds, seed=None, tries=1):
    """
    Initializes the population using Latin Hypercube Sampling (LHS) to ensure 
    good coverage of the search space.
    """
    lb = np.asarray(lower_bounds)
    ub = np.asarray(upper_bounds)
    if lb.ndim == 2: lb = lb[0]
    if ub.ndim == 2: ub = ub[0]

    best_U = None
    best_score = -np.inf
    rng_local = np.random.default_rng(seed)

    for _ in range(max(1, int(tries))):
        s = int(rng_local.integers(0, 2**32 - 1))
        sampler = qmc.LatinHypercube(d=S, seed=s)
        U = sampler.random(n=P)

        if tries > 1:
            # Maximin distance criterion
            score = pdist(U, metric='euclidean').min()
            if score > best_score:
                best_score, best_U = score, U
        else:
            best_U = U
            break

    X = lb + best_U * (ub - lb)
    return X

# -----------------------------------------------------------------------------
# 2. Markov Chain State Transitions
# -----------------------------------------------------------------------------
@njit
def _transition_states(states: np.ndarray, Pmat: np.ndarray) -> None:
    """
    Updates the behavioral state of each particle using a Markov Chain.
    States: 
      2: Oscillation (Resting/Settling)
      5: Zoomies (Exploration/Sprinting)
      6: Purr (Exploitation/Vibration)
    """
    for i in range(states.shape[0]):
        r = np.random.rand()
        cum = 0.0
        for j in range(0, 7):
            cum += Pmat[states[i], j]
            if r < cum:
                states[i] = j
                break

# -----------------------------------------------------------------------------
# 3. Eigen-System Update (The "Skateboard" Logic)
# -----------------------------------------------------------------------------
def update_eigen_system(pbest, pbest_loss, S):
    """
    Calculates the Principal Components (Eigenvectors) of the elite particles.
    This creates a coordinate system aligned with the function's ridges/valleys.
    """
    P = pbest.shape[0]
    # Use top 50% elites to ignore outliers
    top_k = max(S + 1, int(P * 0.5)) 
    sort_idx = np.argsort(pbest_loss)[:top_k]
    top_pbest = pbest[sort_idx]
    
    # Calculate Covariance Matrix of the elites
    cov = np.cov(top_pbest, rowvar=False)
    
    # Regularization to prevent singular matrix
    cov += np.eye(S) * 1e-9
    
    # Eigendecomposition
    w, v = np.linalg.eigh(cov)
    
    # Sort by variance (largest first)
    idx = np.argsort(w)[::-1]
    eigen_vecs = v[:, idx]
    eigen_vals = w[idx]
    
    return eigen_vecs, eigen_vals

# -----------------------------------------------------------------------------
# 4. Main Optimization Loop
# -----------------------------------------------------------------------------
def fcpo_optimize(
    loss_fn, dim, lower_bounds, upper_bounds,
    num_particles=None, max_iters=100,
    c1=2.0, c2=2.0,
    transition_period=5,
    early_stop_patience=20, seed=None,
    init_positions=None, workers=1, verbose=False,
    dt_initial=None 
):
    # Dedicated RNG keeps all stochastic behavior reproducible from `seed`.
    rng = np.random.default_rng(seed)
    S = dim
    
    # --- Adaptive Population (LSHADE style) ---
    # Start high for exploration
    if num_particles is None:
        P = 10 * dim 
    else:
        P = int(num_particles)
    
    P_min = 4 # Minimum particles at end of run
    
    lb = np.broadcast_to(lower_bounds, (1, S))
    ub = np.broadcast_to(upper_bounds, (1, S))
    # Keep bounds in shape (1, S) so they can broadcast against particle arrays.

    # --- Initialization ---
    if init_positions is not None:
        x = np.array(init_positions, copy=True)
        P = x.shape[0]
    else:
        x = _lhs_init(P, S, lower_bounds, upper_bounds, seed=seed, tries=5)
    
    # Strict V_max helps stability in early phases
    v_max_abs = 0.2 * (ub - lb)
    v = rng.uniform(-v_max_abs, v_max_abs, (P, S))

    # State Variables (Markov Chain)
    states = rng.integers(0, 7, size=P, dtype=np.int64)
    Pmat   = np.full((7, 7), 1.0/7, dtype=np.float64)

    # Parallel Evaluator
    if workers == -1: workers = cpu_count() - 1
    def evaluate_population(curr_x):
        # Objective evaluations are independent, so they parallelize cleanly.
        return np.array(Parallel(n_jobs=workers)(delayed(loss_fn)(curr_x[i]) for i in range(len(curr_x))))

    # Initial Eval
    pbest = x.copy()
    pbest_loss = evaluate_population(x)
    
    gidx       = int(np.argmin(pbest_loss))
    gbest      = pbest[gidx].copy()
    gbest_loss = pbest_loss[gidx]
    no_imp     = 0
    initial_P  = P
    
    # Eigen-System Cache
    eigen_vecs = np.eye(S)
    eigen_vals = np.ones(S)

    if verbose:
        print(f"Init: P={P}, Loss={gbest_loss:.4e}")

    # =========================================================================
    # MAIN LOOP
    # =========================================================================
    for it in range(1, max_iters+1):
        progress = it / max_iters
        
        # ---------------------------------------------------------------------
        # A. Linear Population Size Reduction (LPSR)
        # ---------------------------------------------------------------------
        # Shrink the swarm over time so compute budget shifts toward refinement.
        plan_P = int(round(((P_min - initial_P) / max_iters) * it + initial_P))
        if P > plan_P:
            # Kill the worst particles
            sort_indices = np.argsort(pbest_loss)
            keep_indices = sort_indices[:plan_P]
            x, v = x[keep_indices], v[keep_indices]
            pbest, pbest_loss = pbest[keep_indices], pbest_loss[keep_indices]
            states = states[keep_indices]
            P = plan_P

        # ---------------------------------------------------------------------
        # B. Standard Physics (Cosine Inertia)
        # ---------------------------------------------------------------------
        # Cosine Decay: Keeps inertia high (~0.9) for exploration, drops to 0.1
        w = 0.4 + 0.5 * np.cos(np.pi * progress) 
        w = max(w, 0.1)

        # Terminal Lockdown (Last 2%): Freezes velocity to prevent oscillation
        curr_v_max = v_max_abs
        if progress > 0.98:
            curr_v_max = v_max_abs * 1e-6
            w = 0.0

        r1 = rng.random((P, S))
        r2 = rng.random((P, S))
        
        # Standard PSO Update
        v = w * v + c1 * r1 * (pbest - x) + c2 * r2 * (gbest - x)
        v = np.clip(v, -curr_v_max, curr_v_max)
        x = x + v
        x = np.clip(x, lb, ub)

        # ---------------------------------------------------------------------
        # C. Cat State Logic (Biasi et al. Physics)
        # ---------------------------------------------------------------------
        
        # Periodically update the "Skateboard" (Eigen System)
        if it % transition_period == 0 and P > S:
             eigen_vecs, eigen_vals = update_eigen_system(pbest, pbest_loss, S)

        # --- STATE 5: "ZOOMIES" (Sprint/Jump) ---
        # Logic: Exploration along the ridge/valley.
        if progress < 0.90: # Disable near end to converge
            mask_zoomies = (states == 5)
            if np.any(mask_zoomies):
                n_z = mask_zoomies.sum()
                if n_z > 1:
                    # 1. Select Elite Pairs
                    top_k = max(2, int(P * 0.4)) 
                    curr_sort = np.argsort(pbest_loss)
                    top_indices = curr_sort[:top_k]
                    
                    choices = rng.choice(top_indices, size=(n_z, 2))
                    idx_1, idx_2 = choices[:, 0], choices[:, 1]
                    
                    # 2. Gradient Guidance (Better - Worse)
                    # This creates a vector pointing DOWN the valley
                    loss_1, loss_2 = pbest_loss[idx_1], pbest_loss[idx_2]
                    swap_mask = loss_1 > loss_2
                    idx_better = np.where(swap_mask, idx_2, idx_1)
                    idx_worse  = np.where(swap_mask, idx_1, idx_2)
                    
                    # 3. Calculate Jump Vector
                    diff_vector = pbest[idx_better] - pbest[idx_worse]
                    F = rng.normal(0.5, 0.3, (n_z, 1)) # Jitter
                    
                    # 4. Apply Jump
                    # Target is the better elite
                    new_pos = pbest[idx_better] + F * diff_vector
                    
                    # [CRITICAL] Update ALL dimensions (cr_mask=True)
                    # This ensures the particle slides ALONG the rotated ridge
                    # without breaking the correlation.
                    x[mask_zoomies] = new_pos
                    v[mask_zoomies] = 0.0 # Stop momentum

        # --- STATE 6: "PURR" (Vibration/Rest) ---
        # Logic: Fine-tuning locally. 
        # [IMPROVEMENT] "Eigen-Purr" -> Vibrates along the ridge shape.
        mask_purr = (states == 6)
        if np.any(mask_purr):
            n_p = mask_purr.sum()
            scale = 0.02 * ((1.0 - progress)**2)
            
            # Generate aligned noise
            noise_aligned = rng.normal(0, 1, (n_p, S))
            
            # Scale dimensions by Eigenvalues (Variances)
            # This makes the noise "Long" along the ridge and "Thin" across it
            scale_factors = np.sqrt(eigen_vals + 1e-10)
            scale_factors = scale_factors / scale_factors.max() 
            noise_aligned *= scale_factors
            
            # Rotate back to world coordinates
            noise_world = noise_aligned @ eigen_vecs.T
            
            # Apply
            x[mask_purr] = pbest[mask_purr] + noise_world * (ub[0]-lb[0]) * scale
            v[mask_purr] *= 0.0

        # --- STATE 2: "OSCILLATION" (Damping) ---
        # Logic: Strong friction to settle down.
        mask_osc = (states == 2)
        if np.any(mask_osc):
             v[mask_osc] *= 0.5 # Strong damping
             x[mask_osc] += 0.5 * (pbest[mask_osc] - x[mask_osc]) # Pull to pbest

        # Clamp Bounds
        x = np.clip(x, lb, ub)

        # ---------------------------------------------------------------------
        # D. Evaluation & Updates
        # ---------------------------------------------------------------------
        # Evaluate all particles at their new positions before updating bests.
        losses = evaluate_population(x)

        improved = losses < pbest_loss
        pbest[improved]      = x[improved]
        pbest_loss[improved] = losses[improved]

        current_best_idx = np.argmin(pbest_loss)
        if pbest_loss[current_best_idx] < gbest_loss:
            gbest = pbest[current_best_idx].copy()
            gbest_loss = pbest_loss[current_best_idx]
            no_imp = 0
        else:
            no_imp += 1

        # ---------------------------------------------------------------------
        # E. Golden State: Multi-Scale Local Search (Fixes F1 Precision)
        # ---------------------------------------------------------------------
        if progress > 0.95:
            # Check 3 scales: coarse, fine, atomic
            domain_size = ub[0,0] - lb[0,0]
            scales = [1e-4, 1e-6, 1e-9] 
            
            candidates = []
            for s in scales:
                # Generate candidates aligned with Eigen-system for precision
                noise = rng.normal(0, 1, (2, S)) * s * domain_size
                # Rotate noise to align with final valley
                noise = noise @ eigen_vecs.T 
                candidates.append(gbest + noise)
            
            candidates = np.vstack(candidates)
            candidates = np.clip(candidates, lb[0], ub[0])
            
            for cand in candidates:
                # Direct calls avoid joblib overhead for this tiny candidate set.
                val = loss_fn(cand)
                if val < gbest_loss:
                    gbest = cand
                    gbest_loss = val

        # ---------------------------------------------------------------------
        # F. State Transitions
        # ---------------------------------------------------------------------
        if it % transition_period == 0:
            alpha = 0.2
            # Reward the state of the best particle
            best_state = states[current_best_idx]
            Pmat[:, best_state] = (1-alpha)*Pmat[:, best_state] + alpha
            
            # Anti-Stagnation: Force Zoomies if stuck
            if no_imp > 10:
                Pmat[:, 5] += 0.4 
            
            Pmat /= Pmat.sum(axis=1, keepdims=True)
            _transition_states(states, Pmat)

        if verbose and it % 10 == 0:
            print(f"Iter {it}/{max_iters} | P={P} | loss={gbest_loss:.4e}")

    return gbest, gbest_loss, 0, pbest
