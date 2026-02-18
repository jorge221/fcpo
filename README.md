# FCPO

FCPO is a population-based, bound-constrained optimizer implemented in Python/Numpy.
It blends Particle Swarm Optimization (PSO) dynamics with:

- Latin Hypercube Sampling (LHS) initialization,
- a Markov-state behavior model,
- covariance/eigenvector-guided movement,
- adaptive population size reduction,
- and a late-stage local search refinement.

The main entrypoint is `fcpo_optimize` in `fcpo.py`.

## Features

- **Space-filling initialization** with LHS.
- **Adaptive exploration/exploitation** through particle states.
- **Eigen-system guidance** from elite particles.
- **Population size reduction** over time (LSHADE-style schedule).
- **Optional parallel objective evaluation** via `joblib`.

## Requirements

Install dependencies:

```bash
pip install -r requirements.txt
```

## Quick Start

```python
import numpy as np
from fcpo import fcpo_optimize

# Sphere benchmark (global minimum at x=0)
def sphere(x):
    return float(np.sum(x**2))

best_x, best_loss, _, pbest = fcpo_optimize(
    loss_fn=sphere,
    dim=10,
    lower_bounds=-5.12,
    upper_bounds=5.12,
    max_iters=200,
    seed=42,
    workers=1,
    verbose=True,
)

print("Best loss:", best_loss)
print("Best x:", best_x)
```

## API

### `fcpo_optimize(...)`

Returns:

- `gbest`: best solution found (`np.ndarray`)
- `gbest_loss`: objective value at `gbest` (`float`)
- `0`: reserved placeholder return value (compatibility)
- `pbest`: final personal-best positions for all particles (`np.ndarray`)

Important parameters:

- `loss_fn`: callable objective, takes a 1D NumPy vector and returns scalar loss.
- `dim`: search-space dimensionality.
- `lower_bounds`, `upper_bounds`: scalar or vector bounds.
- `num_particles`: initial swarm size (default `10 * dim`).
- `max_iters`: number of optimization iterations.
- `workers`: evaluation parallelism (`-1` means `cpu_count()-1`).
- `seed`: random seed for reproducibility.

## Notes

- `loss_fn` should be deterministic and reasonably fast for best performance.
- Keep `workers=1` if your objective already parallelizes internally.
- Bound clipping is enforced at each update step.

## License

See [LICENSE](LICENSE).
