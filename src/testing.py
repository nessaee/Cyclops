from utils.Region import region_mapping
from utils import Dataset
from utils import Filter
from utils.config import *

# This data is the 40 meter data set
can_out_images = "/Users/user/Documents/MATLAB/LFMSignalRadar/UA-Mono-PoF/Local-Data/Images/candidate"
ver_out_images = "/Users/user/Documents/MATLAB/LFMSignalRadar/UA-Mono-PoF/Local-Data/Images/verifier"

can_out_labels = "/Users/user/Documents/MATLAB/LFMSignalRadar/UA-Mono-PoF/Local-Data/test_data/candidate"
ver_out_labels = "/Users/user/Documents/MATLAB/LFMSignalRadar/UA-Mono-PoF/Local-Data/test_data/verifier"


def main():
    test_dataset = Dataset.Dataset(ver_out_labels, can_out_labels, sync=True)
    print(len(test_dataset.verifier_data))
    filter_ = Filter.Filter(test_dataset,
                            iou=True, horizon=True,
                            visible_horizon=VISIBLE,
                            iou_threshold=IOU_THRESHOLD,
                            claimed_distance_delta=CLAIMED_DISTANCE,
                            x_px_interval=X_PX_INTERVAL)

    num_regions = 4
    # TODO:
    #       Plot scenes to see why things are bing funky
    v_sequences = []
    for scene_v in test_dataset.verifier_data:
        if scene_v.utc == '12':
            here = 1
        encoded_verifier_scene = region_mapping(scene_v, num_regions, region_size=6)
        encoded_verifier_scene_string = ''.join(map(str, encoded_verifier_scene.flatten()))
        v_sequences.append([encoded_verifier_scene_string, scene_v.utc, len(scene_v.targets)])
    c_sequences = []
    for scene_c in test_dataset.candidate_data:
        if scene_c.utc == '12':
            here = 1
        encoded_candidate_scene = region_mapping(scene_c, num_regions, region_size=6)
        if num_regions >= 2:
            encoded_candidate_scene = np.array([encoded_candidate_scene[1, ::-1], encoded_candidate_scene[0, ::-1]])
        encoded_candidate_scene_string = ''.join(map(str, encoded_candidate_scene.flatten()))
        c_sequences.append([encoded_candidate_scene_string, scene_c.utc, len(scene_c.targets)])

    # print("Verifier: ", v_sequences)
    # print("Candidate: ", c_sequences)

    same = 0
    refined_v = []
    refined_c = []
    for i in range(len(v_sequences)):
        if v_sequences[i][0] == '0'*num_regions and c_sequences[i][0] == '0'*num_regions:
            continue
        else:
            refined_v.append(v_sequences[i])
            refined_c.append(c_sequences[i])
    for i in range(len(refined_v)):
        if refined_c[i][0] == refined_v[i][0]:
            same += 1

    similarity = same/len(refined_v)
    print(similarity)


    # print("Verifier: ", refined_v)
    # print("Candidate: ", refined_c)


    # num_regions = 1
    #
    # ''.join(map(str, Region.region_mapping(x, num_regions).flatten()))


main()
