import sys
sys.path.append('maskrcnn/')
from maskrcnn_benchmark.config import cfg
from maskrcnn.demo.predictor import COCODemo

import numpy as np
import cv2 as cv

config_file = "maskrcnn/configs/caffe2/e2e_mask_rcnn_R_50_FPN_1x_caffe2.yaml"
cfg.merge_from_file(config_file)
cfg.merge_from_list(["MODEL.MASK_ON", False])
visual_model = COCODemo(cfg, confidence_threshold=0.2)

img = cv.imread("test_img.jpg")

vis, bbox, features = visual_model.run_on_opencv_image(img)

from pdb import set_trace as bp
bp()
