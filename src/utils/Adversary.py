import numpy as np
from joblib import Parallel, delayed
from .Markov import Markov
from .Helper import frequency, number_sequence

class Adversary:
    """
    Adversary class for generating attack sequences based on a given v_sequence.

    Args:
        v_sequence (list): The input sequence.
        squadron (int, optional): Number of attack sequences to generate. Defaults to 1.
        num_regions (int, optional): Number of regions. Defaults to 1.
    """

    def __init__(self, v_sequence, squadron=1, num_regions=1):
        self.v_sequence = v_sequence
        self.markov = Markov(v_sequence, num_regions=num_regions)
        self.state_probabilities = frequency(number_sequence(v_sequence), percentage=True)
        self.states = np.array(list(self.state_probabilities.keys()))
        self.probabilities = np.array(list(self.state_probabilities.values()))
        self.distribution = {state: self.markov.markov_table.loc[state, self.states].values.flatten().tolist()
                             for state in self.states}
        self.len_v_sequence = len(self.v_sequence)
        if squadron > 1:
            self.sequences = Parallel(n_jobs=-1)(delayed(self.attack)() for _ in range(squadron))
        else:
            self.sequence = self.attack()

    def attack(self):
        """
        Generate an attack sequence.

        Returns:
            list: The generated attack sequence.
        """
        rg = np.random.default_rng() 
        current_state = rg.choice(self.states, p=self.probabilities)

        sequence = [current_state]
        for _ in range(self.len_v_sequence - 1):
            next_state = rg.choice(self.states, p=self.distribution[current_state])
            sequence.append(next_state)
            current_state = next_state

        return [[sequence[i], self.v_sequence[i][1]] for i in range(self.len_v_sequence)]
