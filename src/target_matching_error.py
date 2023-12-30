from utils.Dataset import Dataset, save, load
from utils.config import verifier_label_directory, candidate_label_directory, IOU_THRESHOLD, VISIBLE, X_PX_INTERVAL
from utils.Filter import Filter
import numpy as np
import matplotlib.pyplot as plt
import math
from scipy.optimize import minimize

IMAGE_HEIGHT = 1080


def rotation_matrix_pitch(pitch):
    """Create a rotation matrix for a given pitch angle (in radians)."""
    R = np.array([
        [np.cos(pitch), -np.sin(pitch), 0],
        [np.sin(pitch), np.cos(pitch), 0],
        [0, 0, 1]
    ])
    return R


def project_point(point_3d, intrinsic_matrix, extrinsic_matrix):
    """
    Project a 3D point to 2D using the intrinsic matrix, rotation matrix, and camera position.
    """
    # Convert the 3D point to homogeneous coordinates
    point_3d_homogeneous = np.append(point_3d, 1)

    # Project the point
    point_camera = np.dot(extrinsic_matrix, point_3d_homogeneous)
    point_camera = [point_camera[1], -point_camera[2], point_camera[0]]
    point_image_homogeneous = np.dot(intrinsic_matrix, point_camera[:3])

    # Convert back to non-homogeneous coordinates
    point_image = point_image_homogeneous[:2] / point_image_homogeneous[2]

    return point_image


def find_pitch(ground_point, image_point, camera_position, intrinsic_matrix):
    """
    Find the pitch angle that aligns a given 3D ground point with its corresponding 2D image point,
    focusing only on minimizing the difference in the y-axis.
    """

    def error_function(pitch):
        rotation_matrix = rotation_matrix_pitch(pitch)
        projected_point = project_point(ground_point, intrinsic_matrix, rotation_matrix)
        # Calculate error only based on the y-coordinate
        return (projected_point[1] - image_point[1]) ** 2

    result = minimize(error_function, x0=0, bounds=[(-np.pi, np.pi)])
    return result.x[0]


def divide_regions(distance, region_size, num_regions):
    """
    Divide a distance into regions based on the desired region size and number of regions.

    :param distance: Total distance between two vehicles.
    :param region_size: Minimum size of each region.
    :param num_regions: Desired number of regions.
    :return: A tuple containing the actual number of regions and the size of each region.
    """
    # Calculate the maximum number of regions that can fit with the minimum region size
    max_regions = distance // region_size

    # Adjust the number of regions if necessary
    actual_regions = min(max_regions, num_regions)

    # If no regions can fit, return zero
    if actual_regions == 0:
        return 0, 0

    # Calculate the new region size to evenly distribute the distance
    new_region_size = distance / actual_regions

    return actual_regions, new_region_size


def find_bottom_center_of_bbox(bbox):
    """
    Find the bottom center point of a bounding box.

    :param bbox: A tuple or list containing the bounding box parameters (x, y, width, height).
    :return: A tuple containing the (x, y) coordinates of the bottom center point.
    """
    x, y, w, h = bbox
    bottom_center_x = x + w / 2
    bottom_center_y = y + h
    return (bottom_center_x, bottom_center_y)


def find_overlap_region(bbox_center, regions):
    """Find which region the bottom 10% of bbox overlaps with, given bbox center coordinates."""
    x, y, w, h = bbox_center
    bottom_10_start = y + h / 2 - 0.1 * h
    bottom_10_end = y + h / 2

    max_overlap = 0
    overlapping_region = None
    for region in regions:
        y_low, y_high = region
        overlap = min(bottom_10_end, y_low) - max(bottom_10_start, y_high)
        if overlap > max_overlap and overlap > 0:  # Ensure there's an actual overlap
            max_overlap = overlap
            overlapping_region = region

    return overlapping_region


def encode_bbox_regions(bboxes, regions):
    """Encode the regions overlapped by each bounding box into one-hot bit sequences."""
    encoded_sequences = []
    for bbox in bboxes:
        overlapping_region = find_overlap_region(bbox.bbox, regions)
        if overlapping_region:
            # Create a one-hot encoded bit sequence for the overlapping region
            bit_sequence = [0] * len(regions)
            region_index = regions.index(overlapping_region)
            bit_sequence[region_index] = 1
            encoded_sequences.append(bit_sequence)
        else:
            # If no overlap, encode as all zeros
            encoded_sequences.append([0] * len(regions))
    if not bboxes:
        encoded_sequences.append([0] * len(regions))
    return encoded_sequences


def compare_sequences(seq1, seq2):
    matches = sum(1 for bit1, bit2 in zip(seq1, seq2) if bit1 == bit2)
    total_bits = len(seq1)  # Assuming both sequences have the same length
    similarity = matches / total_bits
    return similarity


def do_comp(scene_ver, scene_can, region_size, num_regions):
    scene_ver.matrix.tune(scene_ver.follower)
    scene_can.matrix.tune(scene_can.follower)

    actual_regions_ver, new_region_size_ver = divide_regions(scene_ver.follower.distance, region_size,
                                                             num_regions)

    actual_regions_can, new_region_size_can = divide_regions(scene_can.follower.distance, region_size,
                                                             num_regions)
    if actual_regions_ver != actual_regions_can or actual_regions_ver != num_regions:
        return None

    actual_regions_can = int(actual_regions_can)
    actual_regions_ver = int(actual_regions_ver)
    ver_regions = []
    for region in range(1, actual_regions_ver + 1):
        ground_point = np.array([region * new_region_size_ver, 0, 0])
        image_point_for_ground_point = project_point(ground_point, scene_ver.matrix.int,
                                                     scene_ver.matrix.ext)
        if region == 1:
            ver_regions.append((IMAGE_HEIGHT, image_point_for_ground_point[1]))
        else:
            ver_regions.append((ver_regions[-1][1], image_point_for_ground_point[1]))

    can_regions = []
    for region in range(1, actual_regions_can + 1):
        ground_point = np.array([region * new_region_size_can, 0, 0])
        image_point_for_ground_point = project_point(ground_point, scene_can.matrix.int,
                                                     scene_can.matrix.ext)
        if region == 1:
            can_regions.append((IMAGE_HEIGHT, image_point_for_ground_point[1]))
        else:
            can_regions.append((can_regions[-1][1], image_point_for_ground_point[1]))

    veri_encoded = encode_bbox_regions(scene_ver.targets, ver_regions)
    cand_encoded = encode_bbox_regions(scene_can.targets, can_regions)

    return compare_sequences(veri_encoded, cand_encoded[::-1])


def plot_data(data):
    """Plot a list of numbers with x-axis ranging from 1 to len(data) - 1."""
    # Generate x values
    x_values = list(range(1, len(data) + 1))

    # Create the plot
    plt.figure(figsize=(10, 6))
    plt.scatter(x_values, data, marker='o')  # Plot the data
    plt.title("Scene Similarity vs Regions")
    plt.xlabel("Number of Regions")
    plt.ylabel("Scene Similarity")
    plt.xticks(range(int(min(x_values)), int(max(x_values)) + 1))
    plt.ylim(0, 1)
    plt.grid(True)
    plt.savefig("/Users/Lewis/Documents/GitHub/UA-CAT-PoF-2022/Local-Data/ssvsregions.png")


def run_test(region_size=6, num_regions=3):
    # verifier_label_directory = "/Users/Lewis/Documents/GitHub/UA-CAT-PoF-2022/Local-Data/Data-Collection-5-Videos/ver_out_labels"
    # candidate_label_directory = "/Users/Lewis/Documents/GitHub/UA-CAT-PoF-2022/Local-Data/Data-Collection-5-Videos/can_out_labels_offset"

    verifier_label_directory = "/Users/Lewis/Documents/GitHub/UA-CAT-PoF-2022/Local-Data/Simulated-Data/verifier"
    candidate_label_directory = "/Users/Lewis/Documents/GitHub/UA-CAT-PoF-2022/Local-Data/Simulated-Data/candidate"
    dataset = Dataset(verifier_label_directory, candidate_label_directory, sync=True, sensor=False)
    filter = Filter(
        dataset,
        split="left",
        iou=True,
        horizon=True,
        visible_horizon=VISIBLE,
        iou_threshold=IOU_THRESHOLD,
        x_px_interval=X_PX_INTERVAL,
    )
    dataset = filter.dataset
    # refine data:
    cand = []
    ver = []
    for i in range(0, len(dataset.verifier_data)):
        if len(dataset.verifier_data[i].targets) != 0:
            ver.append(dataset.verifier_data[i])
            cand.append(dataset.candidate_data[i])

    dataset.verifier_data = ver
    dataset.candidate_data = cand

    results = []
    for i in range(0, len(dataset.verifier_data)):
        results.append(do_comp(dataset.verifier_data[i], dataset.candidate_data[i], region_size, num_regions))

    cleaned_list = [x for x in results if x is not None]

    passing_rate = sum(cleaned_list) / len(cleaned_list)
    return passing_rate


def main():
    results = []
    for i in range(1, 4):
        results.append(run_test(region_size=6, num_regions=i))
    results.append(0)
    results.append(0)
    plot_data(results)


main()
