import numpy as np
from .Sequence import Sequences
from .Helper import number_sequence, print_divider, string_sequence
from .Region import calculate_gamma
from .config import units_per_second, CAP


class Protocol:
    def __init__(
            self,
            dataset,
            cap=CAP,
            state=None,
            encode=False,
            dynamic=False,
            show_sequence=False,
            self_transitions=True,
            encode_window=10,
            sample_rate=30,
            kernel_size=1, 
            alpha=0.50,
            beta=0,
            K=1,
            squadron_size=10,
            num_regions=1,
            tau=1
        ):
        """
        Initializes the Protocol object.

        Args:
            dataset: The dataset object.
            cap: The capacity.
            state: The state.
            encode: A flag indicating whether to encode the sequences.
            dynamic: A flag indicating whether to use dynamic sequences.
            show_sequence: A flag indicating whether to show the sequence.
            self_transitions: A flag indicating whether to allow self transitions.
            encode_window: The encoding window size.
            sample_rate: The sample rate.
            kernel_size: The kernel size.
            alpha: The alpha value.
            beta: The beta value.
            K: The K value.
            squadron_size: The squadron size.
            num_regions: The number of regions.
            tau: The tau value.
        """
        self.tau = tau
        self.N = K
        self.beta = beta
        self.alpha = alpha
        self.cap = cap
        self.encode = encode
        self.squadron_size = squadron_size
        self.sample_rate = sample_rate

        # Initialize Sequences object
        sequences = Sequences(
            dataset.verifier_data, 
            dataset.candidate_data, 
            cap=cap,
            state=state,
            dynamic=dynamic, 
            self_transitions=self_transitions,
            encode=encode,
            window_size=encode_window,
            show_sequence=show_sequence,
            show_encoded=encode,
            squadron_size=squadron_size,
            num_regions=num_regions
        )
        
        self.num_regions = num_regions
        self.s = self.sample_rate  # sample rate (maximum 30)
        self.k = kernel_size * self.s  # kernel_size * units_per_second if encode else kernel_size
        self.s = units_per_second // self.s
        
        self.sv = sequences.v_encoded if encode else sequences.v_sequence
        self.sc = sequences.c_encoded if encode else sequences.c_sequence
        self.sa = sequences.a_encoded if encode else sequences.a_sequence 

        # Print sequence lengths
        print(len(self.sv), len(self.sc), len(self.sa))

        # Downsample sequences
        self.sv = number_sequence(self.sv)[::self.s]
        self.sc = number_sequence(self.sc)[::self.s] 
        
        if self.squadron_size > 1:
            self.sa = [number_sequence(sa)[::self.s] for sa in self.sa]         
        else:
            self.sa = number_sequence(self.sa)[::self.s]
        
        # Print downsampled sequence lengths
        print(len(self.sv), len(self.sc), len(self.sa))

        # Calculate Ri and R
        self.Ri()
        self.R()

    def calculate_test_results(self, test_slices):
        """
        Calculates the test results.

        Args:
            test_slices: The test slices.

        Returns:
            The valid results, total test count, valid test count, average pass rates, and standard deviation of pass rates.
        """
        results = []
        gamma_string = ""
        error_counter = 0
        same = 0
        total = 0

        for v_slice, c_or_a_slice in test_slices:
            gamma = Protocol.test(v_slice, c_or_a_slice, alpha=self.alpha, encode=self.encode, num_regions=self.num_regions, tau=self.tau) 

            if gamma == -1:
                error_counter += 1 
                continue
            else:
                if type(gamma) == list:
                    for g in gamma:
                        total += 1
                        if g == 1:
                            same += 1
                        
                    gamma_string.join([str(g) for g in gamma])
                    gamma = np.mean(gamma)
                    results.append(1 if gamma > self.alpha else 0)
                    
                else:
                    results.append(gamma)
            
        valid_results = results
        total_test_count = len(results)
        valid_test_count = len(valid_results)
        tests_per_protocol = self.N
        pass_rates = self._calculate_pass_rates(valid_results, tests_per_protocol)
        avg_pass_rates = np.mean(pass_rates)
        sd_pass_rates = np.std(pass_rates)

        return valid_results, total_test_count, valid_test_count, avg_pass_rates, sd_pass_rates

    def _calculate_pass_rates(self, valid_results, tests_per_protocol):
        """
        Calculates the pass rates.

        Args:
            valid_results: The valid results.
            tests_per_protocol: The number of tests per protocol.

        Returns:
            The pass rates.
        """
        return [
            np.sum(valid_results[i:i + tests_per_protocol]) / tests_per_protocol 
            for i in range(0, len(valid_results) - tests_per_protocol + 1, tests_per_protocol)
        ]

    def Ri(self):
        """
        Calculates Ri.
        """
        length = min(len(self.sv), len(self.sc))
        test_slices_candidate = self._generate_test_slices(self.sv, self.sc, length, self.k)
        self.RiC, self.c_total_test_count, self.c_valid_test_count, self.avg_cpr_ri, self.sd_cpr_ri = self.calculate_test_results(test_slices_candidate)

        if self.squadron_size == 1:
            test_slices_adversary = self._generate_test_slices(self.sv, self.sa, length, self.k)
            self.RiA, self.a_total_test_count, self.a_valid_test_count, self.avg_apr_ri, self.sd_apr_ri = self.calculate_test_results(test_slices_adversary)
        else:
            self._process_squadron_tests(length)

    def _generate_test_slices(self, sv, sc, length, k):
        """
        Generates test slices.

        Args:
            sv: The verifier sequence.
            sc: The candidate sequence.
            length: The length of the sequences.
            k: The kernel size.

        Returns:
            The test slices.
        """
        return [(sv[i:i + k], sc[i:i + k]) for i in range(0, length - k + 1, k)]

    def _process_squadron_tests(self, length):
        """
        Processes squadron tests.
        """
        self.squadron_RiA = []
        avg = []
        sd = []

        for sa in self.sa:
            test_slices_squadron = self._generate_test_slices(self.sv, sa, length, self.k)
            RiA, _, _, avg_apr, sd_apr = self.calculate_test_results(test_slices_squadron)

            avg.append(avg_apr)
            sd.append(sd_apr)
            self.squadron_RiA.append(RiA)

        self.avg_apr_ri = np.mean(avg)
        self.sd_apr_ri = np.mean(sd)

    def R(self):
        """
        Calculates R.
        """
        self.cscores, self.RC = self.score(self.RiC)
        self.cpr_r = sum(self.RC)/len(self.RC)
        
        if self.squadron_size > 1:
            self.ascores = []
            self.apr_r = []
            for RiA in self.squadron_RiA:
                self.ascores, self.RA = self.score(RiA)
                self.apr_r.append(sum(self.RA)/len(self.RA))
            self.apr_r = np.mean(self.apr_r)
        else:
            self.ascores, self.RA = self.score(self.RiA)
            self.apr_r = sum(self.RA)/len(self.RA)
    
    def score(self, Ri):
        """
        Calculates the scores.

        Args:
            Ri: The Ri values.

        Returns:
            The expected score and R values.
        """
        N = self.N  # Number of tests per protocol
        scores = [sum(Ri[r*N:r*N+N]) for r in range((len(Ri)//N)-1)]
        exp = sum(scores)/len(scores) if scores else 0
        R = [1 if score/N > self.beta else 0 for score in scores]
        return exp, R

    @staticmethod
    def test(s1, s2, alpha=0, encode=False, num_regions=1, tau=1):
        """
        Performs the test.

        Args:
            s1: The first sequence.
            s2: The second sequence.
            alpha: The alpha value.
            encode: A flag indicating whether the sequences are encoded.
            num_regions: The number of regions.
            tau: The tau value.

        Returns:
            The gamma values.
        """
        filtered_pairs = [(s1_elem, s2_elem) for s1_elem, s2_elem in zip(s1, s2) if not (s1_elem == s2_elem and (s1_elem == "0" * num_regions or s1_elem == 0))]
        
        if not filtered_pairs: 
            return -1

        s1, s2 = zip(*filtered_pairs) if filtered_pairs else ([], [])
        gamma_values = [calculate_gamma(s1_elem, s2_elem, tau) for s1_elem, s2_elem in zip(s1, s2) if calculate_gamma(s1_elem, s2_elem, tau) != -1]
        
        if not gamma_values:  # Avoid division by zero
            return -1

        return gamma_values
