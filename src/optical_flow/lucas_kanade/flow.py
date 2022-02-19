import numpy as np
import cv2
from scipy.ndimage import gaussian_filter

from src.cell_detection.detect_blob import opt_blobs
from src.optical_flow.horn_schunck.hornschunck import preprocess


def lk_optical_flow(old_frame_path, new_frame_path, win_size, max_level, criteria, sigma=0):
    # lucas kanade parameters
    lk_params = dict(winSize=win_size,
                     maxLevel=max_level,
                     criteria=criteria)
    # read images
    im1 = cv2.imread(str(old_frame_path))
    im2 = cv2.imread(str(new_frame_path))
    # get location of cells
    blobdata = np.array(opt_blobs(str(old_frame_path)))
    blobs = blobdata[:, :2]
    sizes = blobdata[:, 2]
    blobs[:, [0, 1]] = blobs[:, [1, 0]]
    blobs = blobs.reshape((-1, 1, 2)).astype('float32')
    # convert images to grayscale
    im1_gray = cv2.cvtColor(im1, cv2.COLOR_BGR2GRAY)
    im2_gray = cv2.cvtColor(im2, cv2.COLOR_BGR2GRAY)
    # blur
    im1_blur = gaussian_filter(im1_gray, sigma=sigma)
    im2_blur = gaussian_filter(im2_gray, sigma=sigma)
    # calculate optical flow
    flow, st, err = cv2.calcOpticalFlowPyrLK(im1_blur, im2_blur, blobs, None, **lk_params)
    # get good points
    goodblobs = blobs[st == 1]
    goodflow = flow[st == 1]
    goodsizes = sizes[st.flatten() == 1].reshape(-1, 1)
    return np.concatenate((goodblobs, goodflow, goodsizes), axis=1)
