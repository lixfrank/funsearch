### SYSTEM PROMPT <<<< MODIFY everything contained in the triple quotes, OR SIMPLY DON'T INCLUDE IT TO USE THE DEFAULT SYSTEM PROMPT FROM CONFIG.PY. THE SYSTEM PROMPT ALSO LISTS PACKAGES THAT CAN BE USED >>>>>>
"""
1. Make only small changes but be sure to make some change.
2. Try to keep the code short and any comments concise.
3. Your response should be an implementation of the function function_to_evolve_v# (where # is the current iteration number); do not include any examples or extraneous functions.
4. You may use torch library for tensor algorithm.
5. Think step by step to outline an algorithm, and then implement it.
The code you generate will be appended to the user prompt and run as a python program."""
### END SYSTEM PROMPT
### <<<< user prompt: will be added to the final prompt>>>>>
"""
Task: Develop a program to generate a model of particle splitting. Your starting point consists of two gluons, each with an energy of 20 GeV.
Objective: Utilize your knowledge and creativity to write a program that simulates the particle production process. The core function of the simulation is simulate_evo, which you must identify and implement.
Optimization Goal: The aim is to determine a splitting function that minimizes the difference between the simulated splitting results and the actual splitting function. The difference is quantified using the Wasserstein-1 distance.

1. Make only small changes but be sure to make some change.
2. Try to keep the code short and any comments concise.
3. Your response should be an implementation of the function function_to_evolve_v# (where # is the current iteration number); do not include any examples or extraneous functions.
4. You may use torch library for tensor algorithm. (make sure to import them in suitable place)
5. Think step by step to outline an algorithm, and then implement it.
6. All import of package actions should be in the function function_to_evolve_v# 
The code you generate will be appended to the user prompt and run as a python program.
"""


import funsearch
import torch
from torch import Tensor


@funsearch.run
def evaluate(num_input: int = 1000) -> float:
    """
    Evaluates the distance of sample
    Returns a score based on average of wasserstein distance of results generated from these two splitting function.
    """
    num_sample = num_input
    import torch

    from scipy.stats import wasserstein_distance
    import numpy as np

    def flatten_event(event):
        """Flatten a single event into a 1D array."""
        return np.array(event).flatten()

    def pad_event(flat_event, max_length):
        """Pad a flattened event to the specified length."""
        padded_event = np.zeros(max_length)
        padded_event[:len(flat_event)] = flat_event
        return padded_event

    def compute_distance(sample_A, sample_B):
        """Calculate the average Wasserstein distance between two sets of events."""
        n = min(len(sample_A), len(sample_B))
        distances = []

        # Find the maximum flattened length among all events in both samples
        max_length = max(
            max(len(flatten_event(event)) for event in sample_A),
            max(len(flatten_event(event)) for event in sample_B)
        )

        for i in range(n):
            # Flatten and pad each event to the maximum length
            event_A = pad_event(flatten_event(sample_A[i]), max_length)
            event_B = pad_event(flatten_event(sample_B[i]), max_length)

            # Calculate the Wasserstein distance between the padded events
            distance = wasserstein_distance(event_A, event_B)
            distances.append(distance)

        # Return the average distance between samples
        return np.mean(distances)

    def simulate_pythia():
        import pythia8

        # Initialize Pythia
        pythia = pythia8.Pythia()

        import random
        random_seed = random.randint(1, 100000)
        pythia.readString(f'Random:seed = {random_seed}')
        pythia.readString('Random:setSeed = on')
        
        # Switch off process level but enable parton level
        pythia.readString("ProcessLevel:all = off")
        pythia.readString("PartonLevel:all = on")
        pythia.readString("Check:event = on")
        
        pythia.readString("Next:numberShowInfo = 0")
        pythia.readString("Next:numberShowProcess = 0")
        pythia.readString("Next:numberShowEvent = 0")
        
        # Enable parton shower and hadronization
        pythia.readString("PartonLevel:FSR = on")
        pythia.readString("PartonLevel:ISR = on")
        pythia.readString("HadronLevel:all = on")
        
        # Set particle energy and number of events
        energy = 20.0  # GeV
        
        # Initialize Pythia
        pythia.init()

        final_sample = []
        
        for i_event in range(num_sample):
            # Reset event record for new event
            pythia.event.reset()
        
            # Append two gluons with opposite momenta and color connection
            col1, col2 = 101, 102
            pythia.event.append( 21, 23, 101, 102, 0., 0.,  energy, energy)
            pythia.event.append( 21, 23, 102, 101, 0., 0., -energy, energy)
        
            pythia.event[1].scale(energy)
            pythia.event[2].scale(energy)
            pythia.forceTimeShower(1, 2, energy)
        
            # Switch off the hadronization
            # if not pythia.next():
            #    print("Error generating event!")
            #    break
        
            # Print final particle information after hadronization
            for particle in pythia.event:
                final_particles = [] 
                if particle.isFinal():
                    final_particles.append([particle.id(), particle.pT(), particle.eta(), particle.phi()])
            final_sample.append(final_particles)
        return final_sample
        
    real = simulate_pythia()
    pred = [simulate_evo() for _ in range(num_sample)]

    reward = -compute_distance(real, pred)
    reward = reward.item()
    print(f"This reward: {reward}");
    return reward


@funsearch.evolve  ####<<<< THIS TELLS FUNSEARCH WHAT TO EVOLVE>>>######
def simulate_evo():
    """
    The main function
    Args:
    Returns:
        A list of particles, each particle is a list of [pid, pt, eta, phi]
    """
    energy = 20.0  # GeV
    return [[21, 0, 1e100, 0], [21, 0, -1e100, 0]]


    final_particles = [simulate_splitting_one(p, num_sample) for p in initial_particles]
    return final_particles   

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
