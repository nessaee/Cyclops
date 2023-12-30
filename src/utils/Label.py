from .Target import Target
from .Matrix import Matrix
from .Horizon import Horizon
from .Helper import iou, to_xyxy, normalize_utc
from .config import IMG_WIDTH, MIN_DISTANCE, MAX_DISTANCE, CONFIDENCE_THRESHOLD, SIM


class Label:
    """
    Class representing a label.

    Attributes:
    - boundaries: List of boundaries
    - vehicle: Vehicle information
    - utc: UTC time
    - targets: List of targets
    - follower: Follower target
    - matrix: Matrix object
    - horizon: Horizon object
    """

    def __init__(self, path, normalize_times=True, vehicle=""):
        """
        Initializes a Label object.

        Parameters:
        - path: Path to the label file
        - normalize_times: Flag indicating whether to normalize times
        - vehicle: Vehicle information
        """
        self.boundaries = []
        self.vehicle = vehicle
        with open(path) as f:
            self.utc = path.name[:-4]
            self.targets: list[Target] = []
            for line in f.readlines():
                t = Target(line=line.split(), time=normalize_utc(self.utc) if normalize_times else self.utc)
                t.y_center_bottom()
                if t.valid and t.passes_conditions():
                    self.targets.append(t)
            self.follower = self.find_follower(option="center" if SIM else "lane")
            if self.targets != [] and self.follower_passes_conditions():
                self.targets.remove(self.follower)
                drop_list = []
                for target1 in self.targets:
                    for target2 in self.targets:
                        target_iou = iou(to_xyxy(target1.bbox), to_xyxy(target2.bbox))
                        if target1 != target2 and target_iou > 0.95:
                            if not (self.targets.index(target1) or self.targets.index(target2) in drop_list):
                                print("DOUBLED TARGET:", self.utc, target_iou)
                                drop_list.append(self.targets.index(target1))
                self.targets = [self.targets[i] for i in range(len(self.targets)) if i not in drop_list]
                self.matrix = Matrix(vehicle=self.vehicle, distance=self.follower.distance)
                self.matrix.tune(self.follower)
            else:
                self.follower = None
                print(self.utc, "NO FOLLOWER FOUND")

    def follower_passes_conditions(self):
        """
        Checks if the follower target passes the conditions.

        Returns:
        - True if the follower target passes the conditions, False otherwise
        """
        follower: Target = self.follower
        return False if follower is None or follower.distance < MIN_DISTANCE or follower.distance > MAX_DISTANCE else True

    def apply_horizon(self, theta=0, delta=0):
        """
        Applies the horizon to the label.

        Parameters:
        - theta: Theta value
        - delta: Delta value
        """
        self.horizon = Horizon(self.follower, self.matrix, theta=theta, delta=delta)
        self.targets = [t for t in self.targets if self.horizon.isInside(t.bottom_inner_corner()) and t.y > self.follower.y]

    def find_follower(self, option="lane"):
        """
        Finds the follower target.

        Parameters:
        - option: Option for finding the follower target

        Returns:
        - The follower target
        """
        if option == "center":
            distance_list = [t.x_distance_to_center() for t in self.targets]
            if len(distance_list) == 0:
                return None
            index = distance_list.index(min(distance_list, key=abs))
            return self.targets[index]
        elif option == "lane":
            filtered_list = [t for t in self.targets if t.is_in_lane()]
            if len(filtered_list) == 0:
                return None
            distance_list = [t.distance for t in filtered_list]
            index = distance_list.index(min(distance_list))
            return filtered_list[index]

    def __str__(self):
        """
        Returns a string representation of the label.

        Returns:
        - String representation of the label
        """
        return self.utc + " label"
