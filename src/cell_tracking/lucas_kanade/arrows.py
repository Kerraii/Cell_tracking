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
import imageio
from matplotlib import pyplot as plt
from src.data_parsing.parse_data import frames_to_avi


def lk_tracking_multiprocess(params):
    old_frame_path, new_frame_path, win_size, max_level, criteria = params
    blobdata = np.array(opt_blobs(str(new_frame_path)))
    return lk_optical_flow(old_frame_path, new_frame_path, win_size, max_level, criteria), blobdata


def lk_tracking(stem: Path, pat: str, chosen_path: int, save_location: Path, win_size=(15, 15), max_level=2,
                criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)):
    flist = getimgfiles(stem, pat)
    with Pool(processes=os.cpu_count() - 2) as p:
        iterable = [(flist[j], flist[j+1], win_size, max_level, criteria) for j in range(len(flist) - 1)]
        results = list(tqdm(p.imap(lk_tracking_multiprocess, iterable), total=len(iterable), desc=f"Processing frames of {stem}"))
    cellpaths = construct_paths(results)
    draw_path(cellpaths[chosen_path], flist, save_location, win_size, max_level, criteria)


def draw_path(pathdata, images, save_location, win_size, max_level, criteria):
    print("drawing arrows")
    imagepath = Path(save_location, "image")
    arrowspath = Path(save_location, "arrows")
    animationspath = Path(save_location, "animations")
    path, first, last = pathdata
    arrows = []
    for i in range(first, last+1):
        im1 = imageio.imread(images[i], as_gray=True)
        im2 = imageio.imread(images[i+1], as_gray=True)
        graph_title = "{} -> {}".format(images[i].name, images[i+1].name)
        arrow = [path[i-first][0], path[i-first][1], path[i+1-first][0]-path[i-first][0], path[i+1-first][1]-path[i-first][1]]
        arrows.append(arrow)
        plot([arrow], im1, im2, title=graph_title, save_location=arrowspath, fn=images[i+1].name)
    im1 = imageio.imread(images[first], as_gray=True)
    im2 = imageio.imread(images[first+1], as_gray=True)
    plot(arrows, im1, im2, title="Constructed path", save_location=imagepath, fn="path.jpg")
    frames_to_avi(str(arrowspath), str(Path(animationspath, "arrows_win_size=" + str(win_size) +
                                          "_max_level=" + str(max_level) + "_criteria=" + str(criteria)+".avi")))



def plot(arrows, frame_prev, frame_next, title, save_location, fn):
    ax = plt.figure(dpi=300).gca()
    ax.imshow(frame_prev, cmap="Purples", origin="lower", alpha=0.7)
    ax.imshow(frame_next, cmap="Greens", origin="lower", alpha=0.3)
    ax.xaxis.tick_top()
    ax.set_ylim([0, 480])
    ax.set_xlim([0, 640])
    ax.invert_yaxis()
    for i, (x, y, dx, dy) in enumerate(arrows):
        color = "green"
        ax.arrow(
            x,
            y,
            dx,
            dy,
            color=color,
            head_width=3,
            head_length=2
        )
    ax.set_title(title)

    plt.savefig(Path(save_location, fn))
    plt.close()



if __name__ == "__main__":

    win_size = (15, 15)
    max_level = 1
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)
    file_name = "A2_mo4_1x_5.avi"
    chosen_path = 0
    # Execution from project root
    if "lucas_kanade" in os.getcwd():
        while "src" in os.getcwd():
            os.chdir("..")

    source = Path("data", "frames", file_name)
    destination = Path("resultaten", file_name, "cell_tracking", "lucas_kanade", "win_size=" + str(win_size) +
                       "_max_level=" + str(max_level) + "_criteria=" + str(criteria))

    dir_names = ["arrows", "animations", "image"]
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

    lk_tracking(source, "*.jpg", chosen_path, save_location=destination, win_size=win_size, max_level=max_level, criteria=criteria)
