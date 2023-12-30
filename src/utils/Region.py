import numpy as np
IMAGE_WIDTH = 1920
IMAGE_HEIGHT = 1080
IMG_WIDTH = IMAGE_WIDTH
import matplotlib.pyplot as plt
import matplotlib.patches as patches

def plot_bbox(ax, bbox, name="BBox", color="red"):
    center_x, center_y, w, h = bbox
    x = center_x - w / 2
    y = center_y - h / 2
    rect = patches.Rectangle((x, y), w, h, linewidth=0.5, edgecolor=color, facecolor='none', rotation_point="center")
    ax.add_patch(rect)
    # Label for the bounding box
    plt.text(x + w / 2, y + h / 2, name, horizontalalignment='center', verticalalignment='center', color=color)
def plot_regions(scene, height_intervals):

    # Example bounding box (x, y, width, height)
    bounding_boxes = [target.bbox for target in scene.targets]  # Add your bounding boxes here
    # Create a matplotlib figure and axis
    fig, ax = plt.subplots()

    # Colors for different regions
    region_colors = ['blue', 'green']
    half_width = scene.follower.bbox[0]
    print(half_width)
    region_counter = 1
    # Plot regions on the left side
    for i, (upper, lower) in enumerate(height_intervals):
        ax.fill_betweenx([lower, upper], 0, half_width, color=region_colors[i % len(region_colors)], alpha=0.5)
        plt.text(half_width / 2, (upper + lower) / 2, f'Region {region_counter}', horizontalalignment='center', verticalalignment='center')
        region_counter += 1

    # Plot regions on the right side
    for i, (upper, lower) in enumerate(height_intervals):
        ax.fill_betweenx([lower, upper], half_width, IMG_WIDTH, color=region_colors[i % len(region_colors)], alpha=0.5)
        plt.text(half_width + half_width / 2, (upper + lower) / 2, f'Region {region_counter}', horizontalalignment='center', verticalalignment='center')
        region_counter += 1
    # Plot bounding boxes
    counter = 1
    for bbox in bounding_boxes:
        plot_bbox(ax, bbox, name=str(counter))
        counter += 1
    plot_bbox(ax, scene.follower.bbox, name="0", color="black")
    print(scene.follower.bbox)
    # Set plot limits
    ax.set_xlim(0, IMG_WIDTH)
    # ax.set_ylim(0, max(upper for upper, _ in height_intervals) + 100)
    ax.set_ylim(1080, 0)
    # Set labels and title
    ax.set_xlabel('Width')
    ax.set_ylabel('Height')
    ax.set_title('Horizontal Bands and Bounding Boxes')

    # Show the plot
    plt.show()



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


def project_regions(matrix, region_size, regions):
    regions = int(regions)
    regions_in_image = []
    for region in range(1, regions + 1):
        ground_point = np.array([region * region_size, 0, 0])
        image_point_for_ground_point = project_point(ground_point, matrix.int,
                                                     matrix.ext)
        if region == 1:
            regions_in_image.append((IMAGE_HEIGHT, image_point_for_ground_point[1]))
        else:
            regions_in_image.append((regions_in_image[-1][1], image_point_for_ground_point[1]))
    return regions_in_image


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


def calculate_gamma(B_V_str, B_C_str, tau):
    # Convert string representations of bit sequences into lists of integers
    B_V = [int(bit) for bit in B_V_str]
    B_C = [int(bit) for bit in B_C_str]

    if len(B_V) != len(B_C):
        raise ValueError("The lengths of B_V and B_C must be the same.")

    # Initialize counters for numerator and denominator
    numerator = 0
    denominator = 0

    # Iterate over each bit and perform the comparison
    for i in range(len(B_V)):
        if B_V[i] != 0 or B_C[i] != 0:  # Counting all positions where either bit is not 0
            denominator += 1
            if B_V[i] == B_C[i] and B_V[i] != 0:  # Counting matching positions where bits are not 0
                numerator += 1

    # Calculating gamma, avoiding division by zero
    gamma = 0
    if denominator != 0:
        gamma = int((numerator / denominator) >= tau)

    return gamma
def region_mapping(scene, number_of_regions, region_size=6):

    # get regions in real space
    if number_of_regions == 1:
        activity = len(scene.targets)
        if activity >= 1:
            activity = 1
        return np.array([activity])
    elif number_of_regions >= 2:
        # check and see if we can create regions of at least 6 meters
        regions, new_region_size = divide_regions(scene.follower.distance, region_size, number_of_regions // 2)
        if regions != number_of_regions // 2:
            return -1
        # get regions in image space
        regions_in_image = project_regions(scene.matrix, new_region_size, regions)
        # print(regions_in_image)
        # if len(scene.targets) > 1:
        #     print(len(scene.targets))
        #     plot_regions(scene, regions_in_image)
        # perform 'scene encoding'
        if number_of_regions >= 2:
            targets_left = []
            targets_right = []
            for target in scene.targets:
                if target.x < IMAGE_WIDTH / 2:
                    targets_left.append(target)
                if target.x > IMAGE_WIDTH / 2:
                    targets_right.append(target)
            encoded_sequences_left = encode_bbox_regions(targets_left, regions_in_image)[0]
            encoded_sequences_right = encode_bbox_regions(targets_right, regions_in_image)[0]
            return np.array([encoded_sequences_left, encoded_sequences_right])
    return -1

def generate_sequences(vdata, cdata, num_regions):
    v_sequence = []
    c_sequence = []

    for v_element, c_element in zip(vdata, cdata):
        c_region_map = region_mapping(c_element, num_regions)
        v_region_map = region_mapping(v_element, num_regions)
        # Skip elements if region_mapping returns -1 for cdata element
        if isinstance(c_region_map, int) and c_region_map == -1:
            print("Dropping due to c")
            continue
        # Skip elements if region_mapping returns -1 for vdata element
        elif isinstance(v_region_map, int) and v_region_map == -1:
            print("Dropping due to v")
            continue

        # Process vdata element
        v_region_map = v_region_map.flatten()
        v_sequence_str = ''.join(map(str, v_region_map))
        v_sequence.append([v_sequence_str, v_element.utc])

        # Process cdata element
        c_region_map_flattened = c_region_map.flatten()
        c_sequence_str = ''.join(map(str, c_region_map_flattened))[::-1]
        c_sequence.append([c_sequence_str, c_element.utc])

    return v_sequence, c_sequence