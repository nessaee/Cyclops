import math, time
import pickle
import cv2
import numpy as np
import pandas as pd
from tabulate import tabulate
import matplotlib.pyplot as plt
from pathlib import Path
import shutil
import os


''' DATASET PARAMETERS '''
SIM = False
IMG_LABEL = False
       
''' CONSTANT PARAMETERS '''
UTC_START_TIME = 0 if SIM else 1680463634030
IMG_HEIGHT = 1080
IMG_WIDTH = 1920
FOV = 105 if SIM else 63
units_per_second = 10 if SIM else 30 # ups for encoded sequence 

# Outputs pass and fail cases in data/match and data/mismatch
# NOTE: Make sure you have corresponding image directory in paths below setup properly!
show_output = False


''' FILTER PARAMETERS '''
CAP = 1
MIN_DISTANCE = 0 if SIM else 5
MAX_DISTANCE = 1000 if SIM else 100
IOU_THRESHOLD = 0.92
X_PX_INTERVAL = [920,1000] # Candidate x pixel restriction
CONFIDENCE_THRESHOLD = 25 if SIM else 60
CLAIMED_DISTANCE = 2 # Claimed distance error restriction
VISIBLE = 3 if SIM else 5 # Best known single lane visible distance ahead of follower
THETA = 0 # Best known angle adjustment to FOV

''' TEST PARAMETERS '''
KERNEL_SIZE = 1
WINDOW_SIZE = 1
SQUADRON = 1

''' PLOT PARAMETERS '''
ROWS = 1
COLS = 1


''' PATH PARAMETERS '''
DATASET_PKL = "data/synchronized_sim_dataset_tuned.pkl" if SIM else "data/real_synchronized_dataset.pkl"
FILTERED_DATASET_PKL = "data/filtered_synchronized_sim_dataset.pkl" if SIM else "data/real_filtered_synchronized_dataset.pkl"

sim_verifier = '../Local-Data/uniform/verifier' if IMG_LABEL else None
sim_candidate = '../Local-Data/uniform/candidate' if IMG_LABEL else None
real_verifier = '../Local-Data/REAL/verifier'
real_candidate = '../Local-Data/REAL/can_out_labels'

candidate_label_directory = sim_candidate if SIM else real_candidate
verifier_label_directory = sim_verifier if SIM else real_verifier
verifier_sensor_path = "../Local-Data/Environment-Sensing-Data-Collection-5/Verifier-Orientation.csv"
candidate_sensor_path = "../Local-Data/Environment-Sensing-Data-Collection-5/Candidate-Orientation.csv"
verifier_image_directory = '../Local-Data/CARLA-40/verifier' if IMG_LABEL else '../Local-Data/Images/verifier'
candidate_image_directory = '../Local-Data/CARLA-40/candidate' if IMG_LABEL else '../Local-Data/Images/candidate'

iter_info = {
        "theta": list(range(1,11)),
       "kernel": list(range(2,10)),
         "dist": list(range(20,50,5)),
         "x_px": list(range(3)),
      "claimed": list(range(5)),
      "visible": list(range(2,8,1)),
          "iou": list(range(1,9,2)),
       "window": list(range(5,35,5)),
    "threshold": [x/1000 for x in range(10,1001,10)],
         "rate": [3,5,6,10,15,30],
      "scoring": [x/100 for x in range(10,100,10)]
}
label_info = {
    "theta":"FOV Angle Increment (degrees)",
   "kernel":"Kernel Size",
     "dist":"Follower Distance from Verifier",
     "x_px":"Candidate x pixel requirement",
  "claimed":"Claimed distance error threshold (m)",
  "visible":"Visible distance minimum (m)",
      "iou":"IOU increment (%)",
   "window":"Encoding Window Size",
"threshold":"Alpha",
     "rate":"Sample Rate (fps)",
  "scoring":"Beta",
       "ab":"Beta"
}