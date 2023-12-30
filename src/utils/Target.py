from .config import IMG_HEIGHT, IMG_WIDTH, CONFIDENCE_THRESHOLD, SIM

class Target:
    """
    Represents a target object detected in an image.

    Attributes:
        object_class (int): The class label of the target.
        x (float): The x-coordinate of the target's bounding box.
        y (float): The y-coordinate of the target's bounding box.
        w (float): The width of the target's bounding box.
        h (float): The height of the target's bounding box.
        bbox (list): The bounding box coordinates [x, y, w, h].
        confidence (float): The confidence score of the target detection.
        distance (float): The distance of the target from the camera.
        is_candidate (bool): Indicates if the target is a candidate for further processing.
        valid (bool): Indicates if the target is valid.
        time (int): The timestamp of the target detection.

    Methods:
        __init__(line=None, time=0): Initializes a Target object.
        passes_conditions(): Checks if the target passes the confidence and class conditions.
        candidate(): Marks the target as a candidate.
        y_center_bottom(): Moves the y-coordinate of the target to the center bottom.
        x_distance_to_center(): Calculates the distance of the target's x-coordinate from the center.
        bottom_inner_corner(): Calculates the coordinates of the bottom inner corner of the target.
        is_on_side(side): Checks if the target is on the specified side of the image.
        is_in_lane(): Checks if the target is within the lane interval.
        to_list(): Converts the target attributes to a list.
    """

    def __init__(self, line=None, time=0):
        """
        Initializes a Target object.

        Args:
            line (list): A list containing the target information.
            time (int): The timestamp of the target detection.
        """
        if line is not None:
            self.object_class = int(line[0])
            self.x = float(line[1]) * IMG_WIDTH
            self.y = float(line[2]) * IMG_HEIGHT
            self.w = float(line[3]) * IMG_WIDTH
            self.h = float(line[4]) * IMG_HEIGHT
            self.bbox = [self.x, self.y, self.w, self.h]
            self.confidence = float(line[5])
            self.distance = float(line[6]) # * 3.28084
            self.is_candidate = False
            self.valid = True
            self.time = time
        else:
            self.distance = 10000
            self.valid = False

    def passes_conditions(self):
        """
        Checks if the target passes the confidence and class conditions.

        Returns:
            bool: True if the target passes the conditions, False otherwise.
        """
        c1 = self.confidence > CONFIDENCE_THRESHOLD / 100
        c2 = self.object_class == 0 if not SIM else True
        return True if c1 and c2 else False

    def candidate(self):
        """
        Marks the target as a candidate.
        """
        self.is_candidate = True

    def y_center_bottom(self):
        """
        Moves the y-coordinate of the target to the center bottom.
        """
        self.y = self.y + self.h/2

    def x_distance_to_center(self):
        """
        Calculates the distance of the target's x-coordinate from the center.

        Returns:
            float: The distance of the target's x-coordinate from the center.
        """
        return (self.x/IMG_WIDTH - 0.5)

    def bottom_inner_corner(self):
        """
        Calculates the coordinates of the bottom inner corner of the target.

        Returns:
            tuple: The coordinates of the bottom inner corner (x, y).
        """
        x = (self.x + self.w/2) if self.x < IMG_WIDTH/2 else (self.x - self.w/2)
        return (int(x), int(self.y))

    def is_on_side(self, side):
        """
        Checks if the target is on the specified side of the image.

        Args:
            side (str): The side of the image to check ("left" or "right").

        Returns:
            bool: True if the target is on the specified side, False otherwise.
        """
        return (self.x < IMG_WIDTH/2) if side == "left" else (self.x > IMG_WIDTH/2)

    def is_in_lane(self):
        """
        Checks if the target is within the lane interval.

        Returns:
            bool: True if the target is within the lane interval, False otherwise.
        """
        interval = [0.45*IMG_WIDTH, 0.55*IMG_WIDTH]
        return True if interval[0] < self.x < interval[1] else False

    def to_list(self):
        """
        Converts the target attributes to a list.

        Returns:
            list: The target attributes as a list.
        """
        return [self.object_class, self.x, self.y, self.w, self.h, self.confidence, self.distance]
