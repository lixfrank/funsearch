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
Find the splitting function for a toy model of particle splitting.
Toy model means it not the model used in practical generators of collider HEP, which includes particles with only one parameter energy.
In splitting process, one particle with energy E will split to two particles with energy zE and (1-z)E, where z is randomly sampled from distribution P(z), called splitting funcion, and particle with energy < 1.0 will stop splitting.
The function simulate_splitting_all is the main function of simulation, which needs arguments "splitting_function" for sampling energy.
The goal is to find a splitting function that MINIMIZE the distance between simulating results by the splitting function and real splitting function. The distance is calucalted by wasserstein-1 distance.
You should only use rational function as splitting function evolved, which is enough.
"""


import funsearch
from torch import Tensor


@funsearch.run
def evaluate(num_input: int = 10, num_sample: int = 1000) -> int:
    """
    Evaluates the distance of evolved splitting function and real splitting function.
    Returns a score based on average of wasserstein distance of results generated from these two splitting function.
    """

    import torch

    def g_splitting(z: torch.Tensor):
        """
        Real splitting function
        Args:
            z: fraction of energy in particle splitting
        Returns:
            probability density of fraction z
        """
        return 3 * (1 - z * (1 - z))**2 / (z * (1 - z))

    def sort_and_remove_zero_cols(tensor: torch.Tensor):
        """
        A tool to sort each row of a torch tensor and remove zero columns after sort
        Args:
            tensor: a 2d torch Tensor
        Returns:
            a 2d torch Tensor
        """
        sorted_tensor, _ = torch.sort(tensor, dim=1, descending=True)
        non_zero_cols_mask = ~torch.all(sorted_tensor == 0, dim=0)
        result = sorted_tensor[:, non_zero_cols_mask]

        return result

    def sample_z_batch(particles: torch.Tensor,
                       splitting_func,
                       max_p,
                       delta=0.05,
                       batch_size=1024):
        """
        Sample the fraction z for each particles by randomly sampling,based on the splitting function.
        If particle <= 1.0, the fraction z=0
        Args:
        """
        device = particles.device
        mask = particles > 1.0  # mask of particles need to split
        num_samples = mask.sum().item()

        # init z tensor as zero
        z_values = torch.zeros_like(particles)

        if num_samples == 0:
            return z_values

        # sampling for particles need to split
        samples = []
        while len(samples) < num_samples:
            x = torch.rand(batch_size, device=device) * (1 - 2*delta) + delta
            u = torch.rand(batch_size, device=device) * max_p
            fx = splitting_func(x)
            accepted = u <= fx
            samples.append(x[accepted])

        # select results up to the sampling number
        final_samples = torch.cat(samples)[:num_samples]

        z_values[mask] = final_samples.to(device)

        return z_values

    def simulate_splitting_all(splitting_function,
                               initial_particles,
                               num_sample,
                               delta=0.05,
                               device="cuda"):
        """
        The main function
        Args:
            splitting_function: a Callable function which is probability density of energy fraction z, will be modified by FunSearch.
            num_input: the number of different initial particles
            num_sample: the number of repeated sample for each initial particle.
            delta: the cut of fraction z, making the sample of z in [delta, 1-delta] to avoid singularities.
            device: the device used for computation
        Returns:
            particles after splitting, a 2d torch tensor
        """
        def simulate_splitting_one(splitting_function, initial_particle, num_sample, max_p):
            initial_particles = initial_particle.repeat(num_sample)

            particles = initial_particles.clone()
            while True:
                # Find particles that need splitting
                mask = particles >= 1.0
                if not mask.any():
                    break

                particles = particles.flatten()
                # Sample z values for all particles to split
                z = sample_z_batch(particles, splitting_function, max_p, delta)

                # Calculate new particle energies
                particles = torch.stack([(1-z)*particles, z*particles], dim=1).flatten()
                particles = particles.view(num_sample, -1)
                particles = sort_and_remove_zero_cols(particles)

            return particles

        # Calculate maximum value of splitting function
        z_sample = torch.tensor([delta], dtype=torch.float32, device=device)
        max_p = splitting_function(z_sample).item()
        
        final_particles = [simulate_splitting_one(splitting_function, p, num_sample, max_p) for p in initial_particles]
        return final_particles

    def flattening(x):
        x = x.flatten()
        x = x[x != 0]
        x, _ = x.sort(descending=True)
        return x

    def compute_distance(x1, x2):
        from scipy.stats import wasserstein_distance
        x1 = flattening(x1).numpy()
        x2 = flattening(x2).numpy()
        return wasserstein_distance(x1, x2)

    initial_particles = 2. + 8. * torch.rand(size=(num_input,))
    real = simulate_splitting_all(g_splitting, initial_particles, num_sample)
    pred = simulate_splitting_all(splitting_function_evolve, initial_particles, num_sample)

    distances = torch.tensor([compute_distance(r, p) for r, p in zip(real, pred)])
    reward = -torch.mean(distances)
    return reward


@funsearch.evolve  ####<<<< THIS TELLS FUNSEARCH WHAT TO EVOLVE>>>######
def splitting_function_evolve(z: Tensor) -> Tensor:
    """
    The splitting function of a toy splitting model,
    that describe probability density of energy fraction in the process of one particle splitting.
    Args:
        z:  energy fraction in particle splitting
    Returns:
        probability density of fraction z
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
