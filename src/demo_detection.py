import os
import cv2
from multiprocessing import Pool
import logging
from tqdm import tqdm

import src.data_parsing.parse_data as parser

import src.cell_detection.detect_blob as cell_detection
import src.cell_detection.MSER as mser_filter


def parse_file(f_name):
    return parser.avi_to_frames(f_name, remove_scale=True)


if __name__ == '__main__':
    while "src" in os.getcwd():
        os.chdir("..")
    logging.basicConfig(level=logging.INFO)

    logging.info(f"Parsing Files...")
    # parse all the datafiles into frames
    data_path = os.path.join('data')
    with Pool(processes=os.cpu_count()-2) as p:
        data_files = [f for f in os.listdir(data_path) if f.endswith('.avi')]
        frame_paths = list(tqdm(p.imap(parse_file, data_files), leave=None, total=len(data_files)))

    # create a temporary directory to store the frames for the animation
    TMP = 'tmp'
    os.makedirs(TMP, exist_ok=True)

    MODE_LOG = "LoG"
    MODE_MSER_DOG = "MSER_Dog"
    # modes
    modes = [MODE_LOG, MODE_MSER_DOG]

    COLOR = 'magenta'

    # Create animations
    for file_name, frame_dir in tqdm(zip(data_files, frame_paths), total=len(data_files)):
        # make directories to save frames
        base_path = os.path.join(TMP, file_name)
        for mode in modes:
            os.makedirs(os.path.join(base_path, mode), exist_ok=True)

        logging.info(f"LoG for {file_name}")
        # make frames for: LoG
        for frame in os.listdir(frame_dir):
            # do LoG on original image
            cell_detection.laplacian_of_gaussian(os.path.join(frame_dir, frame),
                                                 save_path=os.path.join(base_path, MODE_LOG, frame), color=COLOR,
                                                 min_sigma=3, max_sigma=6, threshold=0.07)

        logging.info(f"MSER + DoG for {file_name}")
        # make frames for: MSER + DoG
        for frame in os.listdir(frame_dir):
            # save path for MSER image
            tmp_MSER = f'tmp{frame}'

            # original image
            orig_image = cv2.imread(os.path.join(frame_dir, frame))

            # save MSER filtered image
            mser_filter.MSER(os.path.join(frame_dir, frame), save_path=tmp_MSER, delta=5, min_area=14, max_area=5000)

            # detect blobs on MSER image
            blobs = cell_detection.blob_difference_of_gaussian_calc(tmp_MSER,
                                                                    min_sigma=1, max_sigma=6, threshold=0.05)

            # draw blobs on original image
            cell_detection.save_plot(orig_image, blobs, save_path=os.path.join(base_path, MODE_MSER_DOG, frame),
                                     color=COLOR, title='MSER + Difference Of Gaussian')

            # remove MSER image
            os.remove(tmp_MSER)

        # make directory for the animations
        animation = os.path.join('animation', file_name)
        os.makedirs(animation, exist_ok=True)

        logging.info(f"Animations for {file_name}")
        # make animations for each mode
        for mode in modes:
            parser.frames_to_avi(os.path.join(base_path, mode), save_path=os.path.join(animation, f'{mode}.avi'))
