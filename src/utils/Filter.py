from .Dataset import Dataset
from .Matrix import Matrix
from .Horizon import Horizon
from .Helper import iou, to_xyxy, find_nearest
from .config import np, pickle, time, cv2, Path, shutil, os, pd
from .config import IMG_HEIGHT, IMG_WIDTH, IOU_THRESHOLD, show_output
from .config import verifier_image_directory, candidate_image_directory, verifier_sensor_path, candidate_sensor_path

def round_to_multiple(number, multiple):
    return multiple * round(number / multiple)        
                
def scale_image(img, percent):
    width = int(img.shape[1] * percent / 100)
    height = int(img.shape[0] * percent / 100)
    return cv2.resize(img, (width, height), interpolation = cv2.INTER_AREA)

def draw_lines(in1, label1, in2, label2, save_dir=None, save_title=""):
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    
    # verifier image
    h1 = label1.horizon
    f1 = label1.follower
    image1 = cv2.imread(in1)
    image1 = cv2.line(image1, (h1.x1, h1.y1), (h1.x2_left, h1.y2_left), color=(0, 255, 0), thickness=2)
    image1 = cv2.line(image1, (h1.x1, h1.y1), (h1.x2_right, h1.y2_right), color=(0, 255, 0), thickness=2)
    image1 = cv2.line(image1, (0, int(f1.y)), (IMG_WIDTH,  int(f1.y)), color=(255, 0, 0), thickness=2)

    #image1 = image1[0:IMG_HEIGHT, 0:int(IMG_WIDTH/2)]
    

    # candidate image
    h2 = label2.horizon
    f2 = label2.follower
    image2 = cv2.imread(in2)
    image2 = cv2.line(image2, (h2.x1, h2.y1), (h2.x2_left, h2.y2_left), color=(0, 255, 0), thickness=2)
    image2 = cv2.line(image2, (h2.x1, h2.y1), (h2.x2_right, h2.y2_right), color=(0, 255, 0), thickness=2)
    image2 = cv2.line(image2, (0, int(f2.y)), (IMG_WIDTH,  int(f2.y)), color=(255, 0, 0), thickness=2)

    for target in label1.targets:
        image1 = cv2.circle(image1, target.bottom_inner_corner(), radius=10 ,color=(255, 0, 0), thickness=4)
    label2_target_coords = []
    for target in label2.targets:
        image2 = cv2.circle(image2, target.bottom_inner_corner(), radius=10 ,color=(255, 0, 0), thickness=4)
        label2_target_coords +=  [target.bottom_inner_corner()]
    
    if SPLIT == "left":
        # image1 = image1[0:IMG_HEIGHT, 0:int(IMG_WIDTH/2)]
        # image2 = image2[0:IMG_HEIGHT, int(IMG_WIDTH/2):IMG_WIDTH]
        # axis = 1  
        axis = 0 
    elif SPLIT == "right":
        image1 = image1[0:IMG_HEIGHT, int(IMG_WIDTH/2):IMG_WIDTH]
        image2 = image2[0:IMG_HEIGHT, 0:int(IMG_WIDTH/2)]
        axis = 1   
    else:
        axis = 0

    image1 = scale_image(image1, 50)
    image2 = scale_image(image2, 50)

    cat = np.concatenate((image1, image2), axis=axis)

    if save_dir is not None:

        # print(save_title + ":\n")
        # print((h2.x1, h2.y1),(h2.x2_right, h2.y2_right),(h2.x2_left, h2.y2_left))
        
        # print(label2_target_coords)
        cv2.imwrite(save_dir + "/" + save_title, cat)
    else:
        cv2.imshow('Image', cat)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    #cv2.imshow("Window", changed_image)
    


def save_times(times,file_name):
    with open('data/' + file_name + '.pkl', 'wb') as f:
        pickle.dump(times, f)  

class Filter:
    def __init__(self, dataset : Dataset, claimed_distance_delta=None, split=None, iou=False, iou_threshold=None, horizon=False, theta=0, visible_horizon=5, frames=[], dist_interval=[], x_px_interval=[]):
        
        global IOU_THRESHOLD,SPLIT
        SPLIT = split
        self.dataset = dataset
        self.visible_delta = visible_horizon
        self.theta = theta

        print(f"RAW CANDIDATE DATA: {len(dataset.candidate_data)}\nRAW VERIFIER DATA: {len(dataset.verifier_data)}" )
        if iou_threshold is not None:
            global IOU_THRESHOLD 
            IOU_THRESHOLD = iou_threshold 
        if frames != []: self.time_filter(frames[0], frames[1])
        if split is not None: self.split_lr(split)
        if iou: self.iou_filter()
        if claimed_distance_delta is not None: self.claimed_distance(claimed_distance_delta)
        if dist_interval != []: self.distance_interval(dist_interval)
        if x_px_interval != []: self.x_pixel_interval(x_px_interval)
        if horizon: self.horizon_filter()
        print(f"FILTERED CANDIDATE DATA: {len(dataset.candidate_data)}\nFILTERED VERIFIER DATA: {len(dataset.verifier_data)}" )
    def distance_interval(self, dist_interval):
        drop_list = []
        times = [x.utc for x in self.dataset.verifier_data]
        distances = [x.follower.distance for x in self.dataset.verifier_data]
        counter = 0
        for i in range(len(times)):
            if distances[i] > dist_interval[1] or distances[i] < dist_interval[0]:
                drop_list.append(i)
            else:
                counter+=1
        print(drop_list)
        times = [times[i] for i in range(len(times)) if i not in drop_list]
        self.dataset.verifier_data = [x for x in self.dataset.verifier_data if x.utc in times]
        self.dataset.candidate_data = [x for x in self.dataset.candidate_data if x.utc in times]

    def x_pixel_interval(self, x_px_interval):
        drop_list = []
        for i in range(len(self.dataset.verifier_data)):
            label = self.dataset.verifier_data[i]
            if label.follower.x > x_px_interval[1] or label.follower.x < x_px_interval[0]:
                drop_list.append(i)
        self.dataset.verifier_data = [self.dataset.verifier_data[i] for i in range(len(self.dataset.verifier_data)) if i not in drop_list]
        self.dataset.candidate_data = [self.dataset.candidate_data[i] for i in range(len(self.dataset.candidate_data)) if i not in drop_list]
    
    def claimed_distance(self, delta_threshold):
        v_follower_distances = [x.follower.distance for x in self.dataset.verifier_data]
        c_follower_distances = [x.follower.distance for x in self.dataset.candidate_data]
        delta = [v - c for v, c in zip(v_follower_distances, c_follower_distances)]
        times = [x.utc for x, d in zip(self.dataset.verifier_data, delta) if abs(d) < delta_threshold]
        self.dataset.verifier_data = [x for x in self.dataset.verifier_data if x.utc in times]
        self.dataset.candidate_data = [x for x in self.dataset.candidate_data if x.utc in times]

    def split_lr(self, side = "left"):

        for label in self.dataset.verifier_data:  
            label.targets = [t for t in label.targets if t.is_on_side(side)]
        for label in self.dataset.candidate_data: 
            label.targets = [t for t in label.targets if not t.is_on_side(side)]

    def horizon_filter(self):
 
        try:
            shutil.rmtree("./data/mismatch/")
            shutil.rmtree("./data/match/")
        except FileNotFoundError:
            print("Cant delete!")
            pass
        for label in self.dataset.candidate_data:
           label.apply_horizon(theta=self.theta, delta=self.visible_delta)
        for label in self.dataset.verifier_data:
            label.apply_horizon(theta=self.theta, delta=self.visible_delta)  
        if show_output:  
            from .Helper import create_directory
            create_directory("./data/mismatch/")
            create_directory("./data/match/")
            for i in range(len(self.dataset.verifier_data)):
                vlabel = self.dataset.verifier_data[i]
                clabel = self.dataset.candidate_data[i]
                vdir = verifier_image_directory + "/" + str(vlabel.utc) + ".jpg"
                cdir = candidate_image_directory + "/" + str(clabel.utc) + ".jpg"    
                title = str(vlabel.utc)+str((len(vlabel.targets),len(clabel.targets))) + str((vlabel.follower.distance, clabel.follower.distance)) + ".jpg"
                number = 1

                if len(clabel.targets)!=len(vlabel.targets) and (len(vlabel.targets) == 0 or len(clabel.targets) == 0) :
                    draw_lines(vdir, vlabel, cdir, clabel, save_dir="./data/mismatch/" + str(number), save_title=title)
                elif (len(vlabel.targets) != 0 and len(clabel.targets) != 0):
                    draw_lines(vdir, vlabel, cdir, clabel, save_dir="./data/match/"+ str(number), save_title=title)

    def iou_filter(self):
        iou_list = []
        delta_dist = []
        v_followers = [label.follower for label in self.dataset.verifier_data]
        bboxes = [follower.bbox for follower in v_followers]

        iou_list = [iou(to_xyxy(bboxes[i]), to_xyxy(bboxes[i-1])) for i in range(1, len(v_followers))]
        iou_list = [val if 0 < val <= 1 else 0 for val in iou_list]

        times = [x.utc for x, iou_val in zip(self.dataset.verifier_data, iou_list) if iou_val > IOU_THRESHOLD]

        #save_times(times, "filtered-times")
        self.dataset.verifier_data = [x for x in self.dataset.verifier_data if x.utc in times]
        self.dataset.candidate_data = [x for x in self.dataset.candidate_data if x.utc in times]

    def frame_filter(self, start, end):
        self.dataset.verifier_data =  self.dataset.verifier_data[start:end]
        self.dataset.candidate_data =  self.dataset.candidate_data[start:end]

    