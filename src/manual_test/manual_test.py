import os
import src.cell_detection.detect_blob as cell_detection
from src.cell_detection.MSER import MSER
import numpy as np
from tqdm import tqdm
from src.data_parsing.parse_data import avi_to_frames
import cv2

# TODO BLUR
def generate_test_images(file_paths, file_names, image_name):

    base_path = os.path.join('code', 'manual_test')

    for f_name in file_names:
        if not os.path.exists(os.path.join(base_path, f_name)):
            os.mkdir(os.path.join(base_path, f_name))
        if not os.path.exists(os.path.join(base_path, f_name, image_name)):
            os.mkdir(os.path.join(base_path, f_name, image_name))

    for index, f_path in enumerate(file_paths):
        print(f'Generating images for {f_path}')
        base_path = os.path.join('code', 'manual_test', file_names[index], image_name)

        # without MSER
        for min_sigma in tqdm(np.arange(0.1, 2, 0.2)):
            for max_sigma in tqdm(np.arange(1, 10, 1)):
                for threshhold in np.arange(0.01, 0.3, 0.02):
                    if min_sigma < max_sigma:
                        image_name = f'min{min_sigma}_max{max_sigma}_thr{threshhold}.jpg'

                        save_path = os.path.join(base_path, 'dog')
                        if not os.path.exists(save_path):
                            os.mkdir(save_path)
                        cell_detection.difference_of_gaussian(f_path, save_path=os.path.join(save_path, image_name), min_sigma=min_sigma, max_sigma=max_sigma, threshold=threshhold)

                        save_path = os.path.join(base_path, 'log')
                        if not os.path.exists(save_path):
                            os.mkdir(save_path)
                        cell_detection.laplacian_of_gaussian(f_path, save_path=os.path.join(save_path, image_name), min_sigma=min_sigma, max_sigma=max_sigma, threshold=threshhold)

                        image_name = f'min{min_sigma}_max{max_sigma}_thr{threshhold / 10}.jpg'
                        save_path = os.path.join(base_path, 'doh')
                        if not os.path.exists(save_path):
                            os.mkdir(save_path)
                        cell_detection.determinant_of_hessian(f_path, save_path=os.path.join(save_path, image_name), min_sigma=min_sigma, max_sigma=max_sigma, threshold=threshhold / 10)

        tmp_path = 'tmp.jpg'
        # with MSER
        base_path = os.path.join('code', 'manual_test', file_names[index], image_name, 'MSER')
        if not os.path.exists(base_path):
            os.mkdir(base_path)
        print('MSER\n')
        for delta in tqdm(np.arange(3, 9, 1)):
            for min_area in tqdm(np.arange(2, 21, 4)):
                for max_area in tqdm([100, 1000, 5000, 10000, 20000]):
                    # calc mser image
                    MSER(f_path, save_path=tmp_path, delta=delta, min_area=min_area, max_area=max_area)
                    for min_sigma in np.arange(0.1, 2, 0.2):
                        for max_sigma in np.arange(1, 10, 1):
                            for threshhold in np.arange(0.01, 0.3, 0.02):
                                if min_sigma < max_sigma:
                                    image_name = f'delta{delta}_minarea{min_area}_maxarea{max_area}_min{min_sigma}_max{max_sigma}_thr{threshhold}.jpg'

                                    save_path = os.path.join(base_path, 'log')
                                    if not os.path.exists(save_path):
                                        os.mkdir(save_path)
                                    cell_detection.laplacian_of_gaussian(tmp_path, save_path=os.path.join(save_path, image_name), min_sigma=min_sigma, max_sigma=max_sigma, threshold=threshhold)

                                    save_path = os.path.join(base_path, 'dog')
                                    if not os.path.exists(save_path):
                                        os.mkdir(save_path)
                                    cell_detection.difference_of_gaussian(tmp_path, save_path=os.path.join(save_path, image_name), min_sigma=min_sigma, max_sigma=max_sigma, threshold=threshhold)

                                    image_name = f'delta{delta}_minarea{min_area}_maxarea{max_area}_min{min_sigma}_max{max_sigma}_thr{threshhold / 10}.jpg'
                                    save_path = os.path.join(base_path, 'doh')
                                    if not os.path.exists(save_path):
                                        os.mkdir(save_path)
                                    cell_detection.determinant_of_hessian(tmp_path, save_path=os.path.join(save_path, image_name),
                                                           threshold=threshhold / 10, min_sigma=min_sigma, max_sigma=max_sigma)

        os.remove(tmp_path)


def generate_test_images2(file_paths, file_names, im_name):

    base_path = os.path.join('code', 'manual_test')

    for f_name in file_names:
        if not os.path.exists(os.path.join(base_path, f_name)):
            os.mkdir(os.path.join(base_path, f_name))
        if not os.path.exists(os.path.join(base_path, f_name, im_name)):
            os.mkdir(os.path.join(base_path, f_name, im_name))

    for index, f_path in enumerate(file_paths):
        print(f'Generating images for {f_path}')
        base_path = os.path.join('code', 'manual_test', file_names[index], im_name)

        # without MSER
        for min_sigma in tqdm(np.arange(0.1, 2, 0.4)):
            for max_sigma in tqdm(np.arange(1, 10, 2)):
                for threshhold in np.arange(0.01, 0.3, 0.04):
                    if min_sigma < max_sigma:
                        image_name = f'min{min_sigma}_max{max_sigma}_thr{threshhold}.jpg'

                        save_path = os.path.join(base_path, 'dog')
                        if not os.path.exists(save_path):
                            os.mkdir(save_path)
                        cell_detection.difference_of_gaussian(f_path, save_path=os.path.join(save_path, image_name), min_sigma=min_sigma, max_sigma=max_sigma, threshold=threshhold)

                        save_path = os.path.join(base_path, 'log')
                        if not os.path.exists(save_path):
                            os.mkdir(save_path)
                        cell_detection.laplacian_of_gaussian(f_path, save_path=os.path.join(save_path, image_name), min_sigma=min_sigma, max_sigma=max_sigma, threshold=threshhold)

                        image_name = f'min{min_sigma}_max{max_sigma}_thr{threshhold / 10}.jpg'
                        save_path = os.path.join(base_path, 'doh')
                        if not os.path.exists(save_path):
                            os.mkdir(save_path)
                        cell_detection.determinant_of_hessian(f_path, save_path=os.path.join(save_path, image_name), min_sigma=min_sigma, max_sigma=max_sigma, threshold=threshhold / 10)

# TODO BLUR
def generate_test_images3(file_paths, file_names, im_name):
    tmp_path = 'tmp.jpg'

    for index, f_path in enumerate(file_paths):
        print(f'Generating images for {f_path}')
        # with MSER
        base_path = os.path.join('code', 'manual_test', file_names[index], im_name, 'MSER')
        if not os.path.exists(base_path):
            os.makedirs(base_path, exist_ok=True)

        for delta in tqdm(np.arange(6, 8, 1)):
            for min_area in tqdm(np.arange(2, 21, 4)):
                for max_area in tqdm([100, 1000, 5000, 10000, 20000]):
                    # calc mser image
                    try:
                        MSER(f_path, save_path=tmp_path, delta=delta, min_area=min_area, max_area=max_area)

                        for min_sigma in [0.5, 1]:
                            for max_sigma in [5, 6, 7]:
                                for threshhold in np.arange(0.01, 0.15, 0.01):

                                    image_name = f'delta{delta}_minarea{min_area}_maxarea{max_area}_min{min_sigma}_max{max_sigma}_thr{threshhold}.jpg'
                                    # TODO BLUR
                                    save_path = os.path.join(base_path, 'dog')
                                    if not os.path.exists(save_path):
                                        os.mkdir(save_path)
                                    cell_detection.difference_of_gaussian(tmp_path, save_path=os.path.join(save_path, image_name),
                                                           min_sigma=min_sigma, max_sigma=max_sigma, threshold=threshhold)

                                    image_name = f'delta{delta}_minarea{min_area}_maxarea{max_area}_min{min_sigma}_max{max_sigma}_thr{threshhold / 10}.jpg'
                                    save_path = os.path.join(base_path, 'doh')
                                    if not os.path.exists(save_path):
                                        os.mkdir(save_path)
                                    cell_detection.determinant_of_hessian(tmp_path, save_path=os.path.join(save_path, image_name),
                                                           threshold=threshhold / 10, min_sigma=min_sigma,
                                                           max_sigma=max_sigma)
                        for min_sigma in [1, 1.5]:
                            for max_sigma in [5, 6, 7]:
                                for threshhold in np.arange(0.01, 0.15, 0.01):

                                    image_name = f'delta{delta}_minarea{min_area}_maxarea{max_area}_min{min_sigma}_max{max_sigma}_thr{threshhold}.jpg'

                                    save_path = os.path.join(base_path, 'log')
                                    if not os.path.exists(save_path):
                                        os.mkdir(save_path)
                                    cell_detection.laplacian_of_gaussian(tmp_path, save_path=os.path.join(save_path, image_name),
                                                          min_sigma=min_sigma, max_sigma=max_sigma, threshold=threshhold)
                    except AssertionError:
                        pass

    os.remove(tmp_path)


def test_MSER(frames):
    MSER_path = 'tmp.jpg'
    for frame_path in frames:
        im = cv2.imread(frame_path)
        MSER(frame_path, save_path=MSER_path, delta=5, min_area=14, max_area=5000)
        blobs_dog_mser = cell_detection.blob_difference_of_gaussian_calc(MSER_path, min_sigma=1, max_sigma=6, threshold=0.05)
        cell_detection.save_plot(im, blobs_dog_mser, 'magenta', 'Difference Of Gaussian with MSER', show=True)

        blobs_dog = cell_detection.blob_difference_of_gaussian_calc(frame_path, min_sigma=1, max_sigma=6, threshold=0.09)
        cell_detection.save_plot(im, blobs_dog, 'magenta', 'Difference Of Gaussian', show=True)

        blobs_log_mser = cell_detection.blob_laplacian_of_gaussian_calc(MSER_path, min_sigma=3, max_sigma=6, threshold=0.05)
        cell_detection.save_plot(im, blobs_log_mser, 'magenta', 'Laplacian Of Gaussians with MSER', show=True)

        blobs_log = cell_detection.blob_laplacian_of_gaussian_calc(frame_path, min_sigma=3, max_sigma=6, threshold=0.07)
        cell_detection.save_plot(im, blobs_log, 'magenta', 'Laplacian Of Gaussians', show=True)


# TODO BLUR
if __name__ == '__main__':
    while "src" in os.getcwd():
        os.chdir("..")

    if not os.path.exists(os.path.join('code')):
        os.mkdir(os.path.join('code'))

    if not os.path.exists(os.path.join('code', 'manual_test')):
        os.mkdir(os.path.join('code', 'manual_test'))
    # TODO BLUR
    filenames = ['A2_mo4_1x_5.avi', 'D1_5fps_1.5 x magnification_2004.avi', 'B2_std mo_1x_4.avi', 'C2_uninjected_1x_3.avi']
    filenames_mag = ['A6_5fps_0.75 x magnification_1001.avi', 'A11_mo4_2x magnification.avi', 'C11_uninjected_2x magnification.avi', 'D5_5fps_0.75 x magnification_2005.avi']
    for file in filenames_mag:
        pass
        # avi_to_frames(file, remove_scale=True)
    f_paths = [os.path.join('data', 'frames', f_name, '000000.jpg') for f_name in filenames_mag]
    # test_MSER(f_paths)
    # TODO BLUR
    # this will take a very long time
    # generate_test_images3(f_paths, filenames, '000000.jpg')
    MSER_path = 'tmp.jpg'
    f_path = os.path.join('data', 'frames', 'A11_mo4_2x magnification.avi', '000000.jpg')
    MSER(f_path, save_path=MSER_path, delta=5, min_area=14, max_area=5000)
    blobs = cell_detection.blob_difference_of_gaussian_calc(MSER_path, min_sigma=1, max_sigma=6, threshold=0.05)
    blobs2 = cell_detection.blob_laplacian_of_gaussian_calc(f_path, min_sigma=3, max_sigma=6, threshold=0.07)
    blobs3 = cell_detection.blob_determinant_of_hessian_calc(MSER_path, min_sigma=1, max_sigma=6, threshold=0.011)
    # save blobs on orig image
    image = cv2.imread(f_path)
    cell_detection.save_plot(image, blobs2, 'magenta', 'Difference Of Gaussian', 'testje.jpg', show=True)
    # TODO MSER MET LAGERE THRESHOLDS VOORAL VERSCHIL IN D 0.03? 0.04?
    # KIJK OOK IN A OF HIER NIET SLECHTER WORDT threshold
