import os

import cv2
import numpy as np
from skimage.color import rgb2gray

from src.data_parsing.parse_data import avi_to_frames
import src.cell_detection.detect_blob as cell_detection


def MSER(frame, save_path, delta=5, min_area=10, max_area=10000):

    img = cv2.imread(frame, 0)

    # 	int 	delta = 5,
    #   int 	min_area = 60,
    #   int 	max_area = 14400,
    #   double 	max_variation = 0.25,
    #   double 	min_diversity = .2,
    #   int 	max_evolution = 200,
    #   double 	area_threshold = 1.01,
    #   double 	min_margin = 0.003,
    #   int 	edge_blur_size = 5
    mser = cv2.MSER_create(delta=delta, min_area=min_area, max_area=max_area)

    img_gray = rgb2gray(img)

    # regions = list with pixel positions
    regions, _ = mser.detectRegions(img_gray)

    # generate an empty mask for the regions
    mask = np.zeros((img.shape[0], img.shape[1], 1), dtype=np.uint8)
    mask = cv2.dilate(mask, np.ones((150, 150), np.uint8))

    # mark the counters
    contours = [cv2.convexHull(p.reshape(-1, 1, 2)) for p in regions]

    # add the countours to the img
    output = None
    for contour in contours:
        cv2.drawContours(mask, [contour], -1, (255, 255, 255), -1)

        output = cv2.bitwise_and(img, img, mask=mask)

    if output is None:
        raise AssertionError('No cells found')

    # output path
    cv2.imwrite(save_path, output)


if __name__ == '__main__':
    while "src" in os.getcwd():
        os.chdir("..")

    file_name = 'A2_mo4_1x_5.avi'
    # file_name = 'C2_uninjected_1x_3.avi'
    file_name = 'D1_5fps_1.5 x magnification_2004.avi'
    frame_dir = avi_to_frames(file_name, remove_scale=True)

    if not os.path.exists(os.path.join('code', 'MSER')):
        os.mkdir(os.path.join('code', 'MSER'))

    s_path = os.path.join('code', 'MSER', file_name)
    if not os.path.exists(s_path):
        os.mkdir(s_path)

    tmp_path = 'tmp.jpg'

    for filename in os.listdir(frame_dir):
        MSER(os.path.join(frame_dir, filename), tmp_path)
        cell_detection.difference_of_gaussian(tmp_path, save_path=os.path.join(s_path, filename), min_sigma=0.4, max_sigma=5, threshold=0.03)

    os.remove(tmp_path)
