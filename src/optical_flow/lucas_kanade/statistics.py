import numpy as np
import cv2
import os
from pathlib import Path
import shutil
import sys
from multiprocessing import Pool
from tqdm import tqdm
from flow import lk_optical_flow
from utils import getimgfiles
from matplotlib import pyplot as plt


def lk_speed_vs_size_multiprocess(params):
    old_frame_path, new_frame_path, win_size, max_level, criteria = params
    # calculate optical flow
    flow = lk_optical_flow(old_frame_path, new_frame_path, win_size, max_level, criteria)
    # calculate speeds
    sizes = flow[:, 4]
    goodblobs = flow[:, :2]
    goodflow = flow[:, 2:4]
    return np.stack((sizes, np.sqrt(np.sum(np.square(np.subtract(goodflow, goodblobs)), axis=1))), axis=1)


def lk_speed_vs_size(stem: Path, pat: str, save_location: Path, win_size=(15, 15), max_level=2,
                     criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)):
    flist = getimgfiles(stem, pat)
    with Pool(processes=os.cpu_count() - 2) as p:
        iterable = [(flist[j], flist[j+1], win_size, max_level, criteria) for j in range(len(flist) - 1)]
        results = np.concatenate(list(tqdm(p.imap(lk_speed_vs_size_multiprocess, iterable), total=len(iterable), desc=f"Processing frames of {stem}")))
    plt.scatter(results[:, 0], results[:, 1])
    plt.xlabel('Cell radius (pixels)')
    plt.ylabel('Speed (pixels per frame)')
    plt.title("Cell radius vs speed")
    plt.savefig(str(save_location))
    plt.close()



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
    destination = Path("resultaten", file_name, "optical_flow", "lucas_kanade", "win_size=" + str(win_size) +
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
                print("Cant remove {}.\nfiles still opened in other programs!".format(path), file=sys.stderr)
                exit(42069)
        path.mkdir(parents=True)

    lk_speed_vs_size(source, "*.jpg", save_location=Path(paths[0], "speed.jpg"), win_size=win_size, max_level=max_level, criteria=criteria)

