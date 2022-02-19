import shutil
import sys

from pathlib import Path
import os

from tqdm import tqdm
from multiprocessing import Pool

from src.optical_flow.horn_schunck.hornschunck import HornSchunck, getimgfiles, preprocess
from src.optical_flow.horn_schunck.plots import compareGraphs, plotGuessVsActual
import src.cell_detection.detect_blob as cell_detection
from src.cell_tracking.linker import link
from src.data_parsing.parse_data import frames_to_avi
from src.optical_flow.utils import get_arrows, calculate_total_squared_error


def clean_results_path(results_path, is_lucas=False):
    res_dir = Path(results_path.parent)
    results_path = str(results_path.absolute())
    if os.path.exists(results_path) and os.path.isfile(results_path):
        os.remove(results_path)
    res_dir_str = str(res_dir.absolute())
    if not (os.path.exists(res_dir_str) and os.path.isdir(res_dir_str)):
        Path.mkdir(res_dir, parents=True)
    with open(results_path, "a+") as results_file:
        if not is_lucas:
            results_file.write("n,alpha,sigma,mse\n")
        else:
            results_file.write("window,max_level,sigma,mse\n")


def clean_destination(destination_dir):
    if os.path.exists(destination_dir) and os.path.isdir(destination_dir):
        try:
            shutil.rmtree(destination_dir)
        except PermissionError:
            print("Cant remove {}.\nfiles still opened in other programs!".format(destination_dir),
                  file=sys.stderr)
            raise PermissionError


def horn_schunck_validate_multiprocess(params):
    old_frame_path, new_frame_path, Niter, alpha, sigma, verbose, save_location, full_stats, threshold = params
    im1, im1_blur, im2, im2_blur = preprocess(old_frame_path, new_frame_path, sigma)
    U, V = HornSchunck(im1_blur, im2_blur, alpha=alpha, Niter=Niter, verbose=verbose, save_location=save_location,
                       fn=new_frame_path.name)
    blobs_prev = cell_detection.opt_blobs(str(old_frame_path))
    arrows = get_arrows(blobs_prev, U, V)
    blobs_next = cell_detection.opt_blobs(str(new_frame_path))
    linked = link(blobs_next, arrows, threshold)
    # Plot differences
    graph_title = "{} -> {}".format(old_frame_path.name, new_frame_path.name)
    plotGuessVsActual(arrows, linked, im1, im2, title=graph_title, save_location=save_location,
                      fn=new_frame_path.name)
    total_squared_error = calculate_total_squared_error(arrows, linked)
    return total_squared_error


def horn_schunk_calculate_multiprocess(params):
    old_frame_path, new_frame_path, Niter, alpha, sigma, verbose, save_location, full_stats = params
    im1, im1_blur, im2, im2_blur = preprocess(old_frame_path, new_frame_path, sigma)

    U, V = HornSchunck(im1_blur, im2_blur, alpha=alpha, Niter=Niter, verbose=verbose, save_location=save_location,
                       fn=new_frame_path.name)
    blobs = cell_detection.opt_blobs(str(old_frame_path))
    graph_title = "{} -> {}".format(old_frame_path.name, new_frame_path.name)
    compareGraphs(U, V, im1, blobs, title=graph_title, fn=new_frame_path.name, save_location=save_location,
                  full_stats=full_stats)


def horn_schunck_validate(stem: Path, pat: str, alpha: float, sigma: float, Niter: int, verbose: bool,
                          save_location: Path, threshold):
    flist = getimgfiles(stem, pat)
    iterable = [(flist[j], flist[j + 1], Niter, alpha, sigma, verbose, save_location, False, threshold) for j in
                range(len(flist) - 1)]
    with Pool(processes=os.cpu_count() - 2) as p:
        mse = list(tqdm(p.imap_unordered(horn_schunck_validate_multiprocess, iterable), leave=None,
                        total=len(iterable), desc="Validating frames of {}".format(stem)))
        if any(map(lambda x: x == 0, mse)):
            print("{} has 0 arrows!".format(save_location))
        mse = sum(mse) / len(iterable)
        return mse


def horn_schunck_calculate(stem: Path, pat: str, alpha: float, sigma: float, Niter: int, verbose: bool,
                           save_location: Path, full_stats: bool):
    flist = getimgfiles(stem, pat)
    iterable = [(flist[j], flist[j + 1], Niter, alpha, sigma, verbose, save_location, full_stats) for j in
                range(len(flist) - 1)]
    with Pool(processes=os.cpu_count() - 2) as p:
        # Write frames to disk
        list(tqdm(p.imap_unordered(horn_schunk_calculate_multiprocess, iterable), leave=None, total=len(iterable),
                  desc="Calculating frames of {}".format(stem)))


def main():
    # Select execution mode
    VALIDATE = True
    CALCULATE = False
    assert VALIDATE != CALCULATE, "Choose exactly 1 execution mode!"
    # Select input file
    file_name = "A2_mo4_1x_5.avi"
    source = Path("data", "frames", file_name)
    # Execute from project root
    if "horn_schunck" in os.getcwd():
        while "src" in os.getcwd():
            os.chdir("..")
    # Create results directory
    paths = []
    dir_names = []
    results_path = Path("resultaten", file_name, "optical_flow", "horn_schunck", "validate", "results.csv")
    clean_results_path(results_path)

    N = 75
    alpha = 5
    sigma = 5
    if VALIDATE:
        threshold = 7
        destination = Path("resultaten", file_name, "optical_flow", "horn_schunck", "validate",
                           "alpha=" + str(alpha) + "_N=" + str(N) + "_sigma=" + str(sigma) + "_threshold=" +
                           str(threshold))
        clean_destination(destination)
        paths = []
        dir_names = ["arrows", "animations"]
        for dir_name in dir_names:
            path = Path(destination, dir_name)
            paths.append(path)
            path.mkdir(parents=True)

        mse = horn_schunck_validate(source, "*.jpg", alpha=alpha, sigma=sigma, Niter=N, verbose=False,
                                    save_location=destination, threshold=threshold)
        with open(results_path, "a") as results_file:
            results_file.write("{},{},{},{}\n".format(N, alpha, sigma, mse))
    if CALCULATE:
        destination = Path("resultaten", file_name, "optical_flow", "horn_schunck",
                           "calculate", "alpha=" + str(alpha) + "_N=" + str(N) + "_sigma=" + str(sigma))
        clean_destination(destination)
        paths = []
        dir_names = ["arrows", "animations"]
        for dir_name in dir_names:
            path = Path(destination, dir_name)
            paths.append(path)
            path.mkdir(parents=True)
        horn_schunck_calculate(source, "*.jpg", alpha=alpha, sigma=sigma, Niter=N, verbose=False,
                               save_location=destination, full_stats=False)

    for i in range(len(paths) - 1):
        frames_to_avi(str(paths[i]), str(Path(paths[-1], dir_names[i] + "alpha=" + str(alpha) + "_N=" + str(
            N) + "_sigma=" + str(sigma) + ".avi")))


if __name__ == "__main__":
    main()
