from .Label import Label
from pathlib import Path
from .config import pickle, time, verifier_sensor_path, candidate_sensor_path, SIM
from .Helper import sensor_update_matrix

def save(obj, filename):
    """
    Save an object to a file using pickle serialization.

    Args:
        obj: The object to be saved.
        filename: The name of the file to save the object to.
    """
    with open(filename, 'wb') as outp:  # Overwrites any existing file.
        pickle.dump(obj, outp, pickle.HIGHEST_PROTOCOL)

def load(filename):
    """
    Load an object from a file using pickle deserialization.

    Args:
        filename: The name of the file to load the object from.

    Returns:
        The loaded object.
    """
    start = time.time()
    with open(filename, 'rb') as f:
        dataset = pickle.load(f)
    end = time.time()
    #print("Pickle Initialization:", end - start, "seconds")
    return dataset

def get_calibration_dictionary(vehicle):
    """
    Get the calibration dictionary for a specific vehicle.

    Args:
        vehicle: The name of the vehicle.

    Returns:
        The calibration dictionary.
    """
    import pickle
    with open("data/" + vehicle + '_calibration_dictionary.pkl', 'rb') as f:
        return pickle.load(f) 

def numerical_sort(file):
    """
    Extract the numerical part of the filename for sorting.

    Args:
        file: The file to extract the numerical part from.

    Returns:
        The numerical part of the filename.
    """
    return int(file.stem)

class Dataset:
    """
    A class representing a dataset.

    Attributes:
        verifier_data: The verifier data.
        candidate_data: The candidate data.
        times: A list of times.
    """

    def __init__(self, verifier_label_directory, candidate_label_directory, sync=False, sensor=False, pkl=""):
        """
        Initialize a Dataset object.

        Args:
            verifier_label_directory: The directory containing the verifier labels.
            candidate_label_directory: The directory containing the candidate labels.
            sync: A boolean indicating whether to synchronize the verifier and candidate data.
            sensor: A boolean indicating whether to update the data with sensor information.
            pkl: The name of the pickle file to load the dataset from.
        """
        self.times = []

        if pkl != "":
            start = time.time()
            with open(pkl + '.pkl', 'rb') as f:
                self = pickle.load(f)
            end = time.time()
            print("Pickle Initialization:", end - start, "seconds")

        else:
            start = time.time()
            self.verifier_data = self.init_labels(Path(verifier_label_directory), vehicle="verifier")
            end = time.time()
            print("Verifier Data Initialization:", end - start, "seconds")

            start = time.time()
            self.candidate_data = self.init_labels(Path(candidate_label_directory), vehicle="candidate")
            # for label in self.candidate_data: label.utc = str(int(label.utc)-264)
            end = time.time()
            print("Candidate Data Initialization:", end - start, "seconds")
            
            if sync: 
                start = time.time()
                self.sync() 
                end = time.time()
                print("Synchronization:", end - start, "seconds" )
            if sensor:
                self.verifier_data =  sensor_update_matrix(self.verifier_data, verifier_sensor_path)
                self.candidate_data = sensor_update_matrix(self.candidate_data, candidate_sensor_path)

    def sync(self):
        """
        Synchronize the verifier and candidate data based on their timestamps.
        """
        v_times = [label.utc for label in self.verifier_data]
        c_times = [label.utc for label in self.candidate_data]
        times = [t for t in v_times if t in c_times]
        self.verifier_data = [l for l in self.verifier_data if l.utc in times]
        self.candidate_data = [l for l in self.candidate_data if l.utc in times]
        print(len(v_times)-len(self.verifier_data), "frames dropped")

    def init_labels(self, label_directory, vehicle):
        """
        Initialize the labels for a specific vehicle.

        Args:
            label_directory: The directory containing the labels.
            vehicle: The name of the vehicle.

        Returns:
            The initialized labels.
        """
        path = Path(label_directory)
        sorted_label_paths = sorted(path.glob('*.txt'), key=numerical_sort)
        if self.times != []:
            sorted_label_paths = [x for x in sorted_label_paths if any(y in str(x) for y in self.times)]
        labels = [Label(path, vehicle=vehicle) for path in sorted_label_paths]
        labels = [l for l in labels if l.follower is not None]
        return labels
