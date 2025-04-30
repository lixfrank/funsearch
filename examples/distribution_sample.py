### SYSTEM PROMPT <<<< MODIFY everything contained in the triple quotes, OR SIMPLY DON'T INCLUDE IT TO USE THE DEFAULT SYSTEM PROMPT FROM CONFIG.PY. THE SYSTEM PROMPT ALSO LISTS PACKAGES THAT CAN BE USED >>>>>>
"""You are a state-of-the-art python code completion system that will be used as part of a genetic algorithm to evolve cap sets.
You will be given a list of functions, and you should improve the incomplete last function in the list.
1. Make only small changes but be sure to make some change.
2. Try to keep the code short and any comments concise.
3. Your response should be an implementation of the function function_to_evolve_v# (where # is the current iteration number); do not include any examples or extraneous functions.
4. You may use torch library for tensor algorithm.
5. Think step by step to outline an algorithm, and then implement it.
The code you generate will be appended to the user prompt and run as a python program."""
### END SYSTEM PROMPT
### <<<< user prompt: will be added to the final prompt>>>>>
"""
Find the distribution from points sampled.
The goal is to find a probability density that MINIMIZE the distance between points sampled by this distribution and target distribution. The distance is calucalted by wasserstein-1 distance.
You should only use rational function as probability density evolved, which is enough.
"""


import funsearch
from torch import Tensor


@funsearch.run
def evaluate(num_samples: int = 1000) -> int:
    """
    Evaluates the distance of evolved distribution function and real distribution function (named g_splitting).
    Returns a score based on average of wasserstein distance of results generated from these two distribution.
    """

    import torch

    def g_splitting(z: torch.Tensor):
        """
        Real splitting function
        Args:
            z: a number between [delta, 1-delta], delta=0.05 by default.
        Returns:
            probability density of fraction z
        """
        return 3 * (1 - z * (1 - z))**2 / (z * (1 - z))

    def sample_z_batch(density_func,
                       num_samples,
                       delta=0.05,
                       batch_size=1024,
                       device="cuda"):
        """
        Sample the fraction z for each particles by randomly sampling,based on the splitting function.
        If particle <= 1.0, the fraction z=0
        Args:
        """
        # Calculate maximum value of splitting function
        z_sample = torch.tensor([delta], dtype=torch.float32, device=device)
        max_p = density_func(z_sample).item()

        # sampling for particles need to split
        samples = []
        while len(samples) < num_samples:
            x = torch.rand(batch_size, device=device) * (1 - 2*delta) + delta
            u = torch.rand(batch_size, device=device) * max_p
            fx = density_func(x)
            accepted = u <= fx
            samples.append(x[accepted])

        # select results up to the sampling number
        final_samples = torch.cat(samples)[:num_samples]

        return final_samples

    def flattening(x):
        x = x.flatten()
        x = x[x != 0]
        x, _ = x.sort(descending=True)
        return x

    def compute_distance(x1, x2):
        """
        compute the distance defined by wasserstein-1
        """
        from scipy.stats import wasserstein_distance
        x1 = flattening(x1).cpu().numpy()
        x2 = flattening(x2).cpu().numpy()
        return wasserstein_distance(x1, x2)

    real = sample_z_batch(g_splitting, num_samples)
    pred = sample_z_batch(density_evolve, num_samples)

    reward = -compute_distance(real, pred)
    return reward


@funsearch.evolve  ####<<<< THIS TELLS FUNSEARCH WHAT TO EVOLVE>>>######
def density_evolve(z: Tensor) -> Tensor:
    """
    The 1-d probability density function to be evolved.
    Args:
        z: a number between [delta, 1-delta], delta=0.05 by default.
    Returns:
        probability density of z
    """
    p = z
    return p


##### SYSTEM PROMPT - modify in CONFIG.PY, or at the top of the spec file
##### THE USER PROMPT WILL BE FORMED AS FOLLOWS:
# === PROMPT ===
# """Finds large cap sets (sets of n-dimensional vectors over F_3 that do not contain 3 points on a line)."""
#
# import itertools
# import numpy as np
# import funsearch
#
# @funsearch.run
# def priority_v0(v: tuple[int, ...], n: int) -> float:
#   """Returns the priority, as a floating point number, of the vector `v` of length `n`. The vector 'v' is a tuple of values in {0,1,2}.
#       The cap set will be constructed by adding vectors that do not create a line in order by priority.
#   """
#   return 0
#
# def priority_v1(v: tuple[int, ...], n: int) -> float:
#   """Improved version of `priority_v0`.
#   """
#   return 1
#
# def priority_v2(v: tuple[int, ...], n: int) -> float:
#   """Improved version of `priority_v1`.
#   """
#   return 2
# === END PROMPT ===
