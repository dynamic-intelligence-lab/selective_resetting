# selective_resetting

Reference implementation of the selective-resetting method for parallel prefix scans proposed in "Generalized Orders of Magnitude for Scalable, Parallel, High-Dynamic-Range Computation" (Heinsen and Kozachkov, 2025). Our method enables you conditionally to reset interim states in any linear recurrence _as you compute all states in parallel via a prefix scan_.


## Installing

1. Clone this repository.

2. Install the Python dependencies in `requirements.txt`.

3. There is no third step.


## Sample Use

In principle, you can apply our selective-resetting method to _any_ linear recurrence (diagonal or not, time-variant or not, over $\mathbb{R}$ or another field) computed in parallel via a prefix scan. This repository provides a sample implementation of our method for only one case: non-diagonal linear recurrences over $\mathbb{R}$. This sample implementation, for PyTorch, is in a single file: [sample_implementation.py](sample_implementation.py). We will walk through an example so you can see how to use it.

Execute the following code to compute a non-diagonal linear recurrence without biases, $S_t = A_t S_{t-1}$, with $S_0 = I$:

```python
import torch
import torch_parallel_scan as tps

# Create sequence of transition matrices A:
n, d = (50, 3)
A = torch.randn(n, d, d)

# Compute linear recurrence S in parallel via a prefix scan:
S_without_resets = tps.prefix_scan(A, torch.matmul, dim=-3)
```

Each transition matrix in `A` has L-2 norm greater than 1, so the norm of state vectors in `S_without_resets` will tend to increase with each additional matrix multiplication. If you plot the max vector norm within each state matrix in `S_without_resets`, using the following code,

```python
fig, axis = plt.subplots(layout='constrained', figsize=(5, 3.5))
axis.set(title="Max L-2 Norm of Each State's Trailing Dimension", yscale='log')
axis.bar(range(n), S_without_resets.norm(dim=-1).max(dim=-1).values)
axis.grid(axis='y')
```

you will obtain a plot similar to this one (it won't be the same because the transition matrices in `A` are random):

![states without resetting](states_without_resetting.png)

Let's say you're working on an application which requires that the L-2 norms of state vectors don't spiral out of control. You will want to rescale state vectors whenever their L-2 norms start getting too large -- say, whenever they exceed 5. Our selective-resetting method allows you to do that... _in parallel, as you compute all states via a prefix scan_:

```python
import torch.nn.functional as F
from sample_implementation import UpdateOnRightWithSelectiveResetOnLeft

# Define selective-resetting transformation:
sr_transform = UpdateOnRightWithSelectiveResetOnLeft(
    d=d,
    select_func=lambda mats: (mats.norm(dim=-1) > 5).any(dim=-1)[..., None, None],
    reset_func=lambda mats: F.normalize(mats, dim=-1),
)

# Add a bias initialized with zeros below each transition matrix:
A_atop_B = F.pad(A, (0,0, 0,d), value=0)  # shape is [n, d + d, d]

# Compute a parallel prefix scan that applies the selective-resetting transform:
cumul_A_atop_B = tps.prefix_scan(A_atop_B, sr_transform, dim=-3)  # cumul A's atop B's

S_with_resets = cumul_A_atop_B[..., :d, :] + cumul_A_atop_B[..., d:, :]
```

If you compare the max vector norms using the following code,

```python
fig, axes = plt.subplots(ncols=2, sharey=True, layout='constrained', figsize=(10, 3.5))
fig.suptitle("Max L-2 Norm of Each State's Trailing Dimension")

axis = axes[0]
axis.set(title="Without Selective Resetting", yscale='log', ylim=(1, 10 ** S_without_resets.max().log10().ceil().item()))
axis.bar(range(n), S_without_resets.norm(dim=-1).max(dim=-1).values)
axis.grid(axis='y')

axis = axes[1]
axis.set(title="With Selective Resetting", yscale='log')
axis.bar(range(n), S_with_resets.norm(dim=-1).max(dim=-1).values)
axis.grid(axis='y')
```

you will obtain a plot similar to this one (it won't be the same because the transition matrices in `A` are random):

![comparison](comparison.png)

Appendix C of our paper has an informal explanation of our selective-resetting method with step-by-step examples.


## Other Implementations

Our algorithm for parallel estimation of the spectrum of Lyapunov exponents of a dynamical system applies our selective-resetting method to prevent state colinearity during a parallel prefix scan over generalized orders of magnitude (GOOMs). Our implementation of the parallel algorithm for estimation of Lyapunov exponents is at [https://github.com/glassroom/parallel_lyapunov_exponents](https://github.com/glassroom/parallel_lyapunov_exponents).


## Citing

TODO: Update citation.

```
@misc{heinsenkozachkov2025gooms,
    title={
        Generalized Orders of Magnitude for
        Scalable, Parallel, High-Dynamic-Range Computation},
    author={Franz A. Heinsen, Leo Kozachkov},
    year={2025},
}
```


## Notes

The work here originated with casual conversations over email between us, the authors, in which we wondered if it might be possible to find a succinct expression for computing non-diagonal linear recurrences in parallel, by mapping them to the complex plane. Our casual conversations gradually evolved into the development of generalized orders of magnitude, along with an algorithm for estimating Lyapunov exponents in parallel, and a novel method for selectively resetting interim states in a parallel prefix scan.

We hope others find our work and our code useful.


