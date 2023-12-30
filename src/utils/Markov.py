from .Helper import binary 
import pandas as pd

class Markov:
    """
    Markov class represents a Markov chain model.

    Attributes:
        sequence (list): The sequence of states.
        num_regions (int): The number of regions.
        markov_probs (dict): The probabilities of transitioning from one state to another.
        markov_table (DataFrame): The Markov table.

    Methods:
        markov_probabilities(states, num_regions): Calculates the probabilities of transitioning from one state to another.
        markov_to_df(standardized=False, region_distribution={}): Converts the Markov probabilities to a DataFrame.
        print_markov_table(): Prints the Markov table.
        print_markov_probs(): Prints the Markov probabilities.
    """

    def __init__(self, sequence, num_regions):
        """
        Initializes a new instance of the Markov class.

        Args:
            sequence (list): The sequence of states.
            num_regions (int): The number of regions.
        """
        self.sequence = sequence
        self.num_regions = num_regions
        self.markov_probs = self.markov_probabilities([s[0] for s in self.sequence])
        self.markov_table = self.markov_to_df()

    def markov_probabilities(self, states, num_regions=1):
        """
        Calculates the probabilities of transitioning from one state to another.

        Args:
            states (list): The sequence of states.
            num_regions (int): The number of regions.

        Returns:
            dict: The probabilities of transitioning from one state to another.
        """
        mapping = binary(self.num_regions)
        markov_table = {}
        cap = len(states)-1 if num_regions==1 else (2**num_regions)-1
        for i in range(cap):
            state_one = states[i] if num_regions==1 else states[mapping[i]]
            state_two = states[i + 1] if num_regions==1 else states[mapping[i + 1]]
            if state_one in markov_table:
                markov_table[state_one].append(state_two)
            else:
                markov_table[state_one] = [state_two]
                
        markov_probs = {s : [] for s in markov_table.keys()}
        for state_one in markov_table.keys():
            unique_states = list(set(markov_table[state_one]))
            length = len(markov_table[state_one])
            for u_state in unique_states:
                occurences = markov_table[state_one].count(u_state)
                if length != 0:
                    markov_probs[state_one].append([u_state , occurences/length])
        return markov_probs

    def markov_to_df(self, standardized=False, region_distribution={}):
        """
        Converts the Markov probabilities to a DataFrame.

        Args:
            standardized (bool): Indicates whether to standardize the probabilities.
            region_distribution (dict): The region distribution.

        Returns:
            DataFrame: The Markov table.
        """
        scene_names = [scene for scene in self.markov_probs]
        df = pd.DataFrame(columns=["Scene"] + scene_names)
        for scene in self.markov_probs:
            index = len(df.index)
            df.loc[index] = [scene] + [0*i for i in range(len(self.markov_probs))]
            for next_scene in self.markov_probs[scene]:
                multiplier = region_distribution[df.loc[index]["Scene"]] if standardized else 1
                # explicit type cast to prevent FutureWarning
                df[next_scene[0]] = df[next_scene[0]].astype(str)
                df.at[index,next_scene[0]] = float(next_scene[1] * multiplier) 
        #df.loc['Column_Total']= df.sum(numeric_only=True, axis=0)
        #df.loc[:,'Row_Total'] = df.sum(numeric_only=True, axis=1)
        return df.set_index("Scene")        

    def print_markov_table(self):
        """
        Prints the Markov table.
        """
        print("Markov Table")
        [print(i, ":", j) for i, j in self.markov_table.items()]

    def print_markov_probs(self):
        """
        Prints the Markov probabilities.
        """
        print("\nMarkov Probabilities")
        for i, j in self.markov_probs.items():
            for k in j:
                print(i, "->", k[0], ":", k[1] * 100, "%")
