import pickle

import cv2
import time
import torch
import numpy as np

from misc.visualization import draw_points, draw_skeleton, draw_points_and_skeleton, joints_dict, check_video_rotation


frame = cv2.imread("outputs/plank_2.jpg")
with open("outputs/wrong_plank_2.pickle", 'rb') as f:
    wrong_dict = pickle.load(f)

with open("outputs/crt_plank_2.pickle", 'rb') as f:
    crt_dict = pickle.load(f)


for i, (pt, pid) in enumerate(zip(wrong_dict['pts'], wrong_dict['person_ids'])):
    frame = draw_points_and_skeleton(frame, pt, wrong_dict['skeleton'], person_index=pid,
                                        points_color_palette='gist_rainbow', skeleton_color_palette='jet',
                                        points_palette_samples=10, posture='wrong')
    
for i, (pt, pid) in enumerate(zip(crt_dict['pts'], crt_dict['person_ids'])):
    frame = draw_points_and_skeleton(frame, pt, crt_dict['skeleton'], person_index=pid,
                                        points_color_palette='gist_rainbow', skeleton_color_palette='jet',
                                        points_palette_samples=10, posture='crt')
    
cv2.namedWindow("frame", cv2.WINDOW_FULLSCREEN) 
cv2.imshow('frame', frame)
k = cv2.waitKey()
if k == ord('s'):
    cv2.imwrite(f"outputs/plank_postures_visualized.jpg", frame)
