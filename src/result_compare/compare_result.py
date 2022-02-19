from src.data_parsing.parse_data import avi_to_frames
import os
from src.cell_detection.animation import MODE_DIFFERENCE_OF_GAUSSIAN, MODE_LAPLACIAN_OF_GAUSSIAN
from src.cell_detection.MSER import MSER
from src.cell_detection.detect_blob import blob_difference_of_gaussian_calc, blob_laplacian_of_gaussian_calc
from src.cell_detection.concurrentie import competitor
from tqdm import tqdm


def avi_to_count_detection(avi_file_name, mser=False, method=MODE_DIFFERENCE_OF_GAUSSIAN):
    frame_dir = avi_to_frames(avi_file_name, remove_scale=True)
    out = []
    tmp_path = 'tmp.jpg'
    for frame in tqdm(os.listdir(frame_dir)):
        file_path = os.path.join(frame_dir, frame)
        use_path = file_path
        if mser:
            MSER(file_path, save_path=tmp_path, delta=5, min_area=7, max_area=5000)
            use_path = tmp_path

        if method == MODE_DIFFERENCE_OF_GAUSSIAN:
            out.append(len(blob_difference_of_gaussian_calc(use_path, min_sigma=1, max_sigma=6, threshold=0.07)))

        if method == MODE_LAPLACIAN_OF_GAUSSIAN:
            out.append(len(blob_laplacian_of_gaussian_calc(use_path, min_sigma=3, max_sigma=6, threshold=0.07)))

    if mser:
        os.remove(tmp_path)
    return out


if __name__ == '__main__':
    while "src" in os.getcwd():
        os.chdir("..")
    filenames = ['A2_mo4_1x_5.avi', 'D1_5fps_1.5 x magnification_2004.avi', 'B2_std mo_1x_4.avi',
                 'C2_uninjected_1x_3.avi']
    file_name = filenames[1]

    counts_own = avi_to_count_detection(file_name, mser=False, method=MODE_DIFFERENCE_OF_GAUSSIAN)
    counts_competitor = competitor.get_counts(file_name)
    assert(len(counts_own) == len(counts_competitor))
    results = [counts_own[i] - counts_competitor[i] for i in range(len(counts_own))]
    # counts = avi_to_count_detection(file_name, mser=True, method=MODE_DIFFERENCE_OF_GAUSSIAN)
    # counts = avi_to_count_detection(file_name, mser=False, method=MODE_LAPLACIAN_OF_GAUSSIAN)
    # counts = avi_to_count_detection(file_name, mser=True, method=MODE_LAPLACIAN_OF_GAUSSIAN)

    print(results)
