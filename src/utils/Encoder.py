from .Helper import string_sequence
from .config import units_per_second

class Encoder:
    """
    Encoder class for encoding sequences.

    Args:
        sequence (list): The input sequence to be encoded.
        window_size (int, optional): The size of the window for window averaging. Defaults to 1.
        method (str, optional): The encoding method to be used. Defaults to "activity".

    Attributes:
        encoded_sequence (list): The encoded sequence.
        sequence (list): The input sequence.
        states (list): The states of the input sequence.
        times (list): The times of the input sequence.

    """

    def __init__(self, sequence, window_size=1, method="activity"):
        self.encoded_sequence = []
        self.sequence = sequence

        if window_size > 1:
            self.window_average(window_size)

        if method == "activity":
            self.split_sequence()
            self.activity_encoding()
            self.upsample()

    def split_sequence(self):
        """
        Splits the input sequence into states and times.
        """
        utc_start_time = int(self.sequence[0][1])
        self.states = [s[0] for s in self.sequence]
        self.times = [int(s[1]) - utc_start_time for s in self.sequence] 
        self.times = [t for t in self.times] # scaling to seconds 

    def activity_encoding(self):
        """
        Encodes the sequence based on activity.
        """
        initial_time = self.times[0]
        start_time = initial_time # start of first activity 
        for i in range(1, len(self.states)):
            if self.states[i] != self.states[i-1]:
                end_time = self.times[i] # end of activity 
                self.encoded_sequence.append([self.states[i-1], (start_time-initial_time, end_time-initial_time)])
                start_time = self.times[i] # start of next activity 
        self.encoded_sequence.append([self.states[-1], (start_time-initial_time, self.times[-1])])

    def window_average(self, window_size):
        """
        Applies window averaging to the sequence.

        Args:
            window_size (int): The size of the window for averaging.
        """
        new_sequence = []
        number_sequence = [int(x[0]) for x in self.sequence]
        for i in range(window_size, len(self.sequence), window_size):
            averaged_state = round(sum(number_sequence[i-window_size:i])/window_size)
            starting_time = self.sequence[i-window_size][1]
            new_sequence.append([averaged_state, starting_time])
        self.sequence = new_sequence

    def upsample(self):
        """
        Upsamples the encoded sequence.
        """
        new_sequence = []
        ms_per_unit = 1000 // units_per_second
        for s in self.encoded_sequence:
            repetitions = (int(s[1][1])-int(s[1][0])) // ms_per_unit
            upsampled_bit = [[s[0], "1ms"]]*repetitions
            new_sequence.extend(upsampled_bit)   
        self.encoded_sequence = new_sequence

    def get_encoded_sequence(self):
        """
        Returns the encoded sequence.

        Returns:
            list: The encoded sequence.
        """
        return self.encoded_sequence

    def get_sequence_string(self):
        """
        Returns the string representation of the encoded sequence.

        Returns:
            str: The string representation of the encoded sequence.
        """
        return string_sequence(self.encoded_sequence)
