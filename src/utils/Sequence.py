from .Encoder import Encoder
from .Helper import string_sequence, print_sequences
from .Adversary import Adversary
from .config import CAP, units_per_second
from .Region import region_mapping, generate_sequences

gap = 1000 // units_per_second

class Sequences:
    """
    Class representing sequences of data.

    Parameters:
    - vdata: List of data for the "v_sequence"
    - cdata: List of data for the "c_sequence"
    - cap: Capacity limit (default: CAP)
    - state: State value (default: None)
    - encode: Whether to encode the sequences (default: False)
    - dynamic: Whether the sequences are dynamic (default: False)
    - window_size: Size of the encoding window (default: None)
    - show_encoded: Whether to show the encoded sequences (default: False)
    - show_sequence: Whether to show the sequences (default: False)
    - self_transitions: Whether to allow self-transitions (default: True)
    - squadron_size: Size of the squadron (default: 1)
    - num_regions: Number of regions (default: 1)
    """

    def __init__(
        self,
        vdata,
        cdata,
        cap=CAP,
        state=None,
        encode=False,
        dynamic=False,
        window_size=None,
        show_encoded=False,
        show_sequence=False,
        self_transitions=True,
        squadron_size=1,
        num_regions=1
    ):
        self.num_regions = num_regions
        self.v_sequence, self.c_sequence = generate_sequences(vdata, cdata, num_regions=num_regions)
    
        self.cap = cap
        self.state = state
        self.dynamic = dynamic
        self.window_size = window_size
        self.self_transitions = self_transitions
        self.squadron_size = squadron_size
        self.filter()
        self.a_sequence = self.adversary_sequence()
        self.consecutivize()
        if encode:
            self.v_encoded = self.encode_sequence(self.v_sequence)
            self.c_encoded = self.encode_sequence(self.c_sequence)
            if squadron_size == 1:
                self.a_encoded = self.encode_sequence(self.a_sequence)
                if show_sequence:
                    print_sequences([self.v_encoded, self.c_encoded, self.a_encoded])
            else:
                self.a_encoded = [self.encode_sequence(sequence) for sequence in self.a_sequence]
                if show_sequence:
                    for i in range(0, len(self.v_encoded), 100):
                        print("V:", string_sequence(self.v_encoded[i:i+99]))
                        print("C:", string_sequence(self.c_encoded[i:i+99]))
                        print("\n")
        elif show_sequence:
            print_sequences([self.v_sequence, self.c_sequence, self.a_sequence])

    def check(self):
        """
        Check the time gaps between consecutive elements in the v_sequence.
        """
        out = []
        for i in range(1, len(self.v_sequence)):
            current_time = int(self.v_sequence[i][1])
            previous_time = int(self.v_sequence[i-1][1])
            delta = current_time - previous_time
            if delta != gap:
                out.append(delta)
        print(out)

    def consecutivize(self):
        """
        Insert missing elements in the sequences to make them consecutive.
        """
        i = 1
        while i < len(self.v_sequence):
            current_time = int(self.v_sequence[i][1])
            previous_time = int(self.v_sequence[i-1][1])
            delta = current_time - previous_time
            if delta != gap:
                insert = [[0, str(previous_time + j * gap)] for j in range(1, delta // gap)]
                self.v_sequence[i:i] = insert
                self.c_sequence[i:i] = insert
                if self.squadron_size > 1:
                    for s in self.a_sequence:
                        s[i:i] = insert
                else:
                    self.a_sequence[i:i] = insert
                i += len(insert)
            else:
                i += 1

    def filter(self):
        """
        Filter the sequences based on the given conditions.
        """
        index_list = []
        if self.cap is not None and self.num_regions == 1:
            self.v_sequence = [v if int(v[0]) <= self.cap else [self.cap, v[1]] for v in self.v_sequence]
            self.c_sequence = [c if int(c[0]) <= self.cap else [self.cap, c[1]] for c in self.c_sequence]
        if not self.dynamic and self.self_transitions and self.state is None:
            index_list = [i for i in range(len(self.v_sequence))]
        else:
            for i in range(len(self.v_sequence)):
                c1 = self.v_sequence[i][0] != 0 and self.dynamic
                c2 = self.v_sequence[i][0] == self.state and self.state is not None
                c3 = self.v_sequence[i][0] != self.v_sequence[i-1][0] and not self.self_transitions
                if c1 or c2 or c3:
                    index_list.append(i)
        self.v_sequence = [self.v_sequence[i] for i in index_list]
        self.c_sequence = [self.c_sequence[i] for i in index_list]

    def adversary_sequence(self):
        """
        Generate the adversary sequence.
        """
        if self.squadron_size > 1:
            adversary = Adversary(self.v_sequence, squadron=self.squadron_size, num_regions=self.num_regions)
            return adversary.sequences
        else:
            adversary = Adversary(self.v_sequence, num_regions=self.num_regions)
            return adversary.sequence

    def encode_sequence(self, sequence):
        """
        Encode the given sequence using the Encoder class.
        """
        x = Encoder(sequence, window_size=self.window_size)
        return x.encoded_sequence