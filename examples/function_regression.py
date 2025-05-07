### SYSTEM PROMPT <<<< MODIFY everything contained in the triple quotes, OR SIMPLY DON'T INCLUDE IT TO USE THE DEFAULT SYSTEM PROMPT FROM CONFIG.PY. THE SYSTEM PROMPT ALSO LISTS PACKAGES THAT CAN BE USED >>>>>>
"""You are a state-of-the-art python code completion system that will be used as part of a genetic algorithm to evolve cap sets.
You will be given a list of functions, and you should improve the incomplete last function in the list.
1. Make only small changes but be sure to make some change.
2. Try to keep the code short and any comments concise.
3. Your response should be an implementation of the function function_to_evolve_v# (where # is the current iteration number); do not include any examples or extraneous functions.
4. Think step by step to outline an algorithm, and then implement it.
The code you generate will be appended to the user prompt and run as a python program."""
### END SYSTEM PROMPT
### <<<< user prompt: will be added to the final prompt>>>>>
"""
Find a certion function from valued points.
The goal is to find a symbolic function f(x) that MINIMIZE the mean square loss between f(x_i) and y_i at a set of valued points (x_i, y_i)
You should consider both accuracy and simplicity of the function, avoiding too complex function and over fitting.
You should only use rational function as probability density evolved, which is enough.
"""


import funsearch
@funsearch.run
def evaluate(num_samples: int = 1000) -> int:
    """
    Evaluates the distance of evolved distribution function and real distribution function (named g_splitting).
    Returns a score based on average of wasserstein distance of results generated from these two distribution.
    """

    import numpy as np

    delta = 1e-2
    z_points = np.random.rand(num_samples) * (1 - 2*delta) + delta
    real = function_real(z_points)
    pred = function_evolve(z_points)

    reward = -np.mean((real - pred)**2)
    return reward


def function_real(z):
    """
    Real function f(z) to be learned
    Args:
        z: a number between [delta, 1-delta], delta=1e-2 by default.
    Returns:
        value of function f(z)
    """
    return 3 * (1 - z * (1 - z))**2 / (z * (1 - z))


@funsearch.evolve  ####<<<< THIS TELLS FUNSEARCH WHAT TO EVOLVE>>>######
def function_evolve(z):
    """
    The 1-d probability density function to be evolved.
    Args:
        z: a number between [delta, 1-delta], delta=0.05 by default.
    Returns:
        probability density of z
    """
    p = z
    return p
