import numpy as np
import cv2
import os
from pathlib import Path
import shutil
import sys
from multiprocessing import Pool
from tqdm import tqdm
from src.optical_flow.lucas_kanade.utils import getimgfiles
from src.optical_flow.lucas_kanade.flow import lk_optical_flow
from src.cell_detection.detect_blob import opt_blobs
from path import construct_paths
from math import sqrt
from matplotlib import pyplot as plt
from scipy.stats.stats import pearsonr


def lk_tracking_multiprocess(params):
    old_frame_path, new_frame_path, win_size, max_level, criteria = params
    blobdata = np.array(opt_blobs(str(new_frame_path)))
    return lk_optical_flow(old_frame_path, new_frame_path, win_size, max_level, criteria), blobdata


def lk_tracking(stem: Path, pat: str, save_location: Path, win_size=(15, 15), max_level=2,
                criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)):
    flist = getimgfiles(stem, pat)
    with Pool(processes=os.cpu_count() - 2) as p:
        iterable = [(flist[j], flist[j+1], win_size, max_level, criteria) for j in range(len(flist) - 1)]
        results = list(tqdm(p.imap(lk_tracking_multiprocess, iterable), total=len(iterable), desc=f"Processing frames of {stem}"))
    cellpaths = construct_paths(results)
    calc_statistics(cellpaths, save_location)


def calc_statistics(paths, save_location):
    print("Calculating statistics")
    meandistances = []
    meansizes = []
    for path, _, _ in paths:
        distances = []
        sizes = []
        for i in range(len(path)-1):
            distances.append(sqrt((path[i+1][0]-path[i][0])**2+(path[i+1][1]-path[i][1])**2))
            sizes.append(path[i+1][2])
        meandistances.append(sum(distances)/len(distances))
        meansizes.append(sum(sizes)/len(sizes))
    plt.scatter(meansizes, meandistances)
    plt.xlabel('Average cell radius (pixels)')
    plt.ylabel('Average speed (pixels per frame)')
    plt.title("Cell radius vs speed")
    plt.savefig(str(save_location))
    plt.close()
    print(pearsonr(meansizes, meandistances))


if __name__ == "__main__":

    win_size = (15, 15)
    max_level = 1
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)
    file_name = "A2_mo4_1x_5.avi"
    # Execution from project root
    if "lucas_kanade" in os.getcwd():
        while "src" in os.getcwd():
            os.chdir("..")

    source = Path("data", "frames", file_name)
    destination = Path("resultaten", file_name, "cell_tracking", "lucas_kanade", "win_size=" + str(win_size) +
                       "_max_level=" + str(max_level) + "_criteria=" + str(criteria))

    dir_names = ["plots"]
    paths = []
    for dir_name in dir_names:
        path = Path(destination, dir_name)
        paths.append(path)
        if os.path.exists(path) and os.path.isdir(path):
            try:
                shutil.rmtree(path)
            except PermissionError:
                print("Cant remove {}.\nfiles still opened in other programs!".format(destination), file=sys.stderr)
                exit(42069)
        path.mkdir(parents=True)

    lk_tracking(source, "*.jpg", save_location=Path(paths[0], "speed.jpg"), win_size=win_size, max_level=max_level, criteria=criteria)
