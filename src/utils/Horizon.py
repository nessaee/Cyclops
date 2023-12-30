import math
from .config import IMG_HEIGHT, IMG_WIDTH, FOV, SIM
from .Matrix import Matrix

class Horizon:
    """
    Horizon class represents the horizon line in a 2D space.
    It calculates the coordinates of the left and right endpoints of the horizon line
    based on the follower's position and the matrix transformation.
    It also provides methods to check if a point lies inside the triangle formed by the horizon line.
    """

    def __init__(self, follower, matrix: Matrix, theta=0, delta=5, adaptive=True):
        """
        Initializes a Horizon object.

        Args:
            follower: The follower object.
            matrix: The matrix object.
            theta: The angle of rotation.
            delta: The delta value.
            adaptive: Whether to use adaptive mode or not.
        """
        global FOV
        self.delta = delta
        self.theta = theta

        self.x1 = round(follower.x)
        self.y1 = round(follower.y)

        theta_center = FOV
        matrix.get_horizon(follower.distance, theta=self.theta, delta=self.delta)

        self.x2_left = int(matrix.left[0])
        self.y2_left = int(matrix.left[1])

        self.x2_right = int(matrix.right[0])
        self.y2_right = int(matrix.right[1])

    def get_end_point(self, left=True):
        """
        Calculates the coordinates of the left or right endpoint of the horizon line.

        Args:
            left: Whether to calculate the left endpoint or the right endpoint.

        Returns:
            The x and y coordinates of the endpoint.
        """
        x2 = 0 if left else IMG_WIDTH
        theta = self.theta_left if left else self.theta_right
        y2 = round(abs(x2 - self.x1) / math.tan(math.radians(theta)) + self.y1)

        return x2, y2

    def area(self, x1, y1, x2, y2, x3, y3):
        """
        Calculates the area of a triangle formed by three points.

        Args:
            x1, y1: The coordinates of the first point.
            x2, y2: The coordinates of the second point.
            x3, y3: The coordinates of the third point.

        Returns:
            The area of the triangle.
        """
        l1 = math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)
        l2 = math.sqrt((x2 - x3) ** 2 + (y2 - y3) ** 2)
        l3 = math.sqrt((x3 - x1) ** 2 + (y3 - y1) ** 2)
        p = (l1 + l2 + l3) / 2
        try:
            area = math.sqrt(p * (p - l1) * (p - l2) * (p - l3))
        except:
            print("Failed area calculation")
            area = 0
        return area

    # A function to check whether point P(x, y)
    # lies inside the triangle formed by
    # A(x1, y1), B(x2, y2) and C(x3, y3)
    def isInside(self, point):
        """
        Checks if a point lies inside the triangle formed by the horizon line.

        Args:
            point: The coordinates of the point.

        Returns:
            True if the point lies inside the triangle, False otherwise.
        """
        x = point[0]
        y = point[1]

        x1 = self.x1
        x2 = self.x2_left
        x3 = self.x2_right

        y1 = self.y1
        y2 = self.y2_left
        y3 = self.y2_right
        # Calculate area of triangle ABC
        A = self.area(x1, y1, x2, y2, x3, y3)
        # Calculate area of triangle PBC
        A1 = self.area(x, y, x2, y2, x3, y3)
        # Calculate area of triangle PAC
        A2 = self.area(x1, y1, x, y, x3, y3)
        # Calculate area of triangle PAB
        A3 = self.area(x1, y1, x2, y2, x, y)

        # Check if sum of A1, A2 and A3 is same as A
        y_min = min(y2, y3)
        left = x > x2 and y > y_min
        right = x < x3 and y > y_min
        condition = (abs(A - (A1 + A2 + A3)) < 10) or (left and right)
        #print("LEFT:", (x2,y2), "RIGHT:", (x3,y3), "TOP:", (x1,y1))
        return condition

    def __str__(self):
        """
        Returns a string representation of the Horizon object.

        Returns:
            A string representation of the Horizon object.
        """
        points = (self.x1, self.y1, self.x2_right, self.y2_right, self.x2_left, self.y2_left)
        return "".join([str(p) + " " for p in points])