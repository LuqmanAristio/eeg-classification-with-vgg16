#%%
import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
# import pickle
#from lesson_functions import *
import time
from vehicle_detector import find_cars
from vehicle_detector import generate_grid
from draw_utils import draw_boxes
from vehicle_detector import generate_heatmap, apply_threshold
from scipy.ndimage import label
from draw_utils import draw_labeled_bboxes
from moviepy.editor import VideoFileClip
from IPython.display import HTML
from collections import deque
from vehicle_classifier import load_trained_model

classifier = load_trained_model('SVM_HLS_1.p')

svc = classifier['classifier']
scaler = classifier['scaler']

config = {}
config['color_space']     = classifier['color_space']
config['spatial_size']    = classifier['spatial_size']
config['hist_bins']       = classifier['hist_bins']
config['orient']          = classifier['orient']
config['pix_per_cell']    = classifier['pix_per_cell']
config['cell_per_block']  = classifier['cell_per_block']
config['hog_channel']     = classifier['hog_channel']
config['spatial_feat']    = classifier['spatial_feat']
config['hist_feat']       = classifier['hist_feat']
config['hog_feat']        = classifier['hog_feat']

def search_all_scales(image):

    ystart_1 = 400-2*8
    ystop_1 = 400+2*64+2*8
    scale_1 = 1.3
    grid_1 = generate_grid(image, ystart_1, ystop_1, scale_1)

    ystart_2 = 400-2*8
    ystop_2 = 656
    scale_2 = 1.8
    grid_2 = generate_grid(image, ystart_2, ystop_2, scale_2)

    ystart_3 = 400-32
    ystop_3 = 656
    scale_3 = 2.3
    grid_3 = generate_grid(image, ystart_3, ystop_3, scale_3)
    
    detected_boxes = find_cars(image, ystart_1, ystop_1, scale_1, svc, scaler, config)
    detected_boxes += find_cars(image, ystart_2, ystop_2, scale_2, svc, scaler, config)
    detected_boxes += find_cars(image, ystart_3, ystop_3, scale_3, svc, scaler, config)
    
    return detected_boxes

class BoxesManager:
    
    def __init__(self, n=10):
        self.n = n 
        self.recent_boxes = deque([], maxlen=n)
        self.current_boxes = None
        self.allboxes = []
        
    def add_boxes(self):
        self.recent_boxes.appendleft(self.current_boxes)
    
    def pop_data(self):
        if self.n_buffered > 0:
            self.recent_boxes.pop()
            
    def set_current_boxes(self, boxes):
        self.current_boxes = boxes
    
    def get_all_boxes(self):
        allboxes = []
        for boxes in self.recent_boxes:
            allboxes += boxes
            
        if len(allboxes) == 0:
            self.allboxes = None
        else:
            self.allboxes = allboxes
    
    def update(self, boxes):
        self.set_current_boxes(boxes)
        self.add_boxes()
        self.get_all_boxes()

def draw_debug_board(img, bboxes, hot_windows, heatmap, labels):
    
    # prepare RGB heatmap image from float32 heatmap channel
    img_heatmap = (np.copy(heatmap) / np.max(heatmap) * 255.).astype(np.uint8);
    img_heatmap = cv2.applyColorMap(img_heatmap, colormap=cv2.COLORMAP_HOT)
    img_heatmap = cv2.cvtColor(img_heatmap, cv2.COLOR_BGR2RGB)

    # prepare RGB labels image from float32 labels channel
    img_labels = (np.copy(labels) / np.max(labels) * 255.).astype(np.uint8);
    img_labels = cv2.applyColorMap(img_labels, colormap=cv2.COLORMAP_HOT)
    img_labels = cv2.cvtColor(img_labels, cv2.COLOR_BGR2RGB)
    
    # draw hot_windows in the frame
    img_hot_windows = np.copy(img)
    img_hot_windows = draw_boxes(img_hot_windows, hot_windows, thick=2)
    
    ymax = 0
    
    board_x = 5
    board_y = 5
    board_ratio = (img.shape[0] - 3*board_x)//3 / img.shape[0] #0.25
    board_h = int(img.shape[0] * board_ratio)
    board_w = int(img.shape[1] * board_ratio)
        
    ymin = board_y
    ymax = board_h + board_y
    xmin = board_x
    xmax = board_x + board_w

    offset_x = board_x + board_w

    # draw hot_windows in the frame
    img_hot_windows = cv2.resize(img_hot_windows, dsize=(board_w, board_h), interpolation=cv2.INTER_LINEAR)
    img[ymin:ymax, xmin:xmax, :] = img_hot_windows
    
    # draw heatmap in the frame
    xmin += offset_x
    xmax += offset_x
    img_heatmap = cv2.resize(img_heatmap, dsize=(board_w, board_h), interpolation=cv2.INTER_LINEAR)
    img[ymin:ymax, xmin:xmax, :] = img_heatmap
    
    # draw heatmap in the frame
    xmin += offset_x
    xmax += offset_x
    img_labels = cv2.resize(img_labels, dsize=(board_w, board_h), interpolation=cv2.INTER_LINEAR)
    img[ymin:ymax, xmin:xmax, :] = img_labels
    
    return img;

boxesManager = BoxesManager(n=30)

def process_image(image, boxes_manager):
    draw_image = np.copy(image)
    image = image.astype(np.float32) / 255
    detected_windows = search_all_scales(image)
    boxes_manager.update(detected_windows)
    heatmap = np.zeros_like(image[:,:,0]).astype(np.float32)
    heatmap = generate_heatmap(heatmap, boxes_manager.allboxes)
    heatmap = apply_threshold(heatmap, 15)
    labels = label(heatmap)
    window_image = draw_labeled_bboxes(draw_image, labels)
    window_image = draw_debug_board(window_image, boxes_manager.allboxes, detected_windows, heatmap, labels[0])
    cv2.putText(window_image, 'Heatmap', (550, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2, cv2.LINE_AA)
    cv2.putText(window_image, 'Label', (1000, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2, cv2.LINE_AA)
    cv2.putText(window_image, 'Raw Detection', (150, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2, cv2.LINE_AA)
    cv2.putText(window_image, 'Merge Detection', (550, 300), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2, cv2.LINE_AA)
    return window_image


def process_video(inpfile, outfile):
    clip = VideoFileClip(inpfile)
    boxes_manager = BoxesManager(n=30)
    out_clip = clip.fl_image(lambda x: process_image(x, boxes_manager))
    out_clip.write_videofile(outfile, audio=False)
# %%
