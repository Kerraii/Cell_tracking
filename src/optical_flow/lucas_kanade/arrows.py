import cv2
import os
from pathlib import Path
import shutil
import sys
from multiprocessing import Pool
from tqdm import tqdm

from src.cell_detection.detect_blob import opt_blobs
from src.cell_tracking.linker import link
from src.data_parsing.parse_data import frames_to_avi
from flow import lk_optical_flow
from src.optical_flow.horn_schunck.hornschunck import preprocess
from src.optical_flow.horn_schunck.main import clean_results_path
from src.optical_flow.horn_schunck.plots import plotGuessVsActual
from src.optical_flow.utils import calculate_total_squared_error
from utils import getimgfiles


def lk_arrows_multiprocess(params):
    old_frame_path, new_frame_path, save_location, win_size, max_level, criteria = params
    # calculate optical flow
    flow = lk_optical_flow(old_frame_path, new_frame_path, win_size, max_level, criteria)
    # read image
    im1 = cv2.imread(str(old_frame_path))
    # draw arrows
    for i, (fromx, fromy, tox, toy, _) in enumerate(flow):
        im1 = cv2.arrowedLine(im1, (int(fromx), int(fromy)), (int(tox), int(toy)), (0, 0, 255), 3)
    save_path = Path(save_location, "arrows", old_frame_path.name)
    cv2.imwrite(str(save_path), im1)


def lk_arrows_validate_multiprocess(params):
    old_frame_path, new_frame_path, save_location, win_size, max_level, criteria, sigma = params
    im1, _, im2, _ = preprocess(old_frame_path, new_frame_path, 0)
    arrows = lk_optical_flow(old_frame_path, new_frame_path, win_size, max_level, criteria, sigma)
    not_found = len(opt_blobs(str(old_frame_path))) - len(arrows)
    formatted_arrows = []
    for fromx, fromy, tox, toy, r in arrows:
        formatted_arrows.append((fromx, fromy, tox - fromx, toy - fromy))
    blobs_next = opt_blobs(str(new_frame_path))
    linked = link(blobs_next, formatted_arrows)
    graph_title = "{} -> {}".format(old_frame_path.name, new_frame_path.name)
    plotGuessVsActual(formatted_arrows, linked, im1, im2, graph_title, save_location, new_frame_path.name)
    return calculate_total_squared_error(formatted_arrows, linked, not_found)


def lk_arrows(stem: Path, pat: str, save_location: Path, win_size=(15, 15), max_level=2,
                 criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)):
    flist = getimgfiles(stem, pat)
    with Pool(processes=os.cpu_count() - 2) as p:
        iterable = [(flist[j], flist[j+1], save_location, win_size, max_level, criteria) for j in range(len(flist) - 1)]
        list(tqdm(p.imap(lk_arrows_multiprocess, iterable), total=len(iterable), desc=f"Processing frames of {stem}"))
    # save last frame
    im = cv2.imread(str(flist[-1]))
    save_path = Path(save_location, "arrows", Path(flist[-1]).name)
    cv2.imwrite(str(save_path), im)


def lk_arrows_validate(stem: Path, pat: str, save_location: Path, win_size=(15, 15), max_level=2,
                 criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03), sigma=0):
    flist = getimgfiles(stem, pat)
    with Pool(processes=os.cpu_count() - 2) as p:
        iterable = [(flist[j], flist[j+1], save_location, win_size, max_level, criteria, sigma) for j in range(len(flist) - 1)]
        errors = list(tqdm(p.imap_unordered(lk_arrows_validate_multiprocess, iterable), total=len(iterable), desc=f"Processing frames of {stem}", leave=None))
        if any(map(lambda x: x == 0, errors)):
            print("{} has 0 arrows!".format(save_location))
        mse = sum(errors) / len(iterable)
        return mse


def run():
    win_size = (15, 15)
    max_level = 2
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)
    file_name = "A2_mo4_1x_5.avi"
    # Execution from project root
    if "lucas_kanade" in os.getcwd():
        while "src" in os.getcwd():
            os.chdir("..")

    source = Path("data", "frames", file_name)
    destination = Path("resultaten", file_name, "optical_flow", "lucas_kanade", "win_size=" + str(win_size) +
                       "_max_level=" + str(max_level) + "_criteria=" + str(criteria))
    dir_names = ["arrows", "animations"]
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

    lk_arrows(source, "*.jpg", save_location=destination, win_size=win_size, max_level=max_level, criteria=criteria)

    frames_to_avi(str(paths[0]), str(Path(paths[1], dir_names[0] + "_win_size=" + str(win_size) +
                                          "_max_level=" + str(max_level) + "_criteria=" + str(criteria)+".avi")))


def validate():
    file_name = "A2_mo4_1x_5.avi"
    # Execution from project root
    if "lucas_kanade" in os.getcwd():
        while "src" in os.getcwd():
            os.chdir("..")
    results_path = Path("resultaten", file_name, "optical_flow", "lucas_kanade", "validate", "results.csv")
    clean_results_path(results_path, is_lucas=True)
    # for win_size in tqdm([(15, 15), (25, 25)], leave=None, desc="Window Size"):
    #     for max_level in tqdm([0, 1], leave=None, desc="Max Level"):
    #         for sigma in tqdm([0, 0.1, 0.5, 1, 5, 10], leave=None, desc="Sigma"):
    win_size = (15, 15)  # TODO verslag: prob niet ideaal voor andere groottes van cellen, indien ingezoomd ofzo
    max_level = 2
    sigma = 0
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)

    source = Path("data", "frames", file_name)
    destination = Path("resultaten", file_name, "optical_flow", "lucas_kanade", "validate", "win_size=" + str(win_size)
                       + "_max_level=" + str(max_level) + "_criteria=" + str(criteria) + "_sigma=" + str(sigma))
    dir_names = ["arrows", "animations"]
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
    mse = lk_arrows_validate(source, "*.jpg", save_location=destination, win_size=win_size, max_level=max_level, criteria=criteria, sigma=sigma)
    frames_to_avi(str(paths[0]), str(Path(paths[-1], dir_names[0] + "_validate.avi")))
    with open(results_path, "a") as results_file:
        results_file.write("{},{},{},{}\n".format(win_size, max_level, sigma, mse))


if __name__ == "__main__":
    VALIDATE = True
    if VALIDATE:
        validate()
    else:
        run()
    exit(0)
