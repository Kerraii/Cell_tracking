# Van https://github.com/scivision/pyoptflow/
from __future__ import annotations

from pathlib import Path

import imageio
import numpy as np
from scipy.ndimage import gaussian_filter
from scipy.signal import convolve2d

HSKERN = np.array(
    [[1 / 12, 1 / 6, 1 / 12], [1 / 6, 0, 1 / 6], [1 / 12, 1 / 6, 1 / 12]], float
)

kernelX = np.array([[-1, 1], [-1, 1]]) * 0.25  # kernel for computing d/dx

kernelY = np.array([[-1, -1], [1, 1]]) * 0.25  # kernel for computing d/dy

kernelT = np.ones((2, 2)) * 0.25


def getimgfiles(stem: Path, pat: str) -> list:
    stem = Path(stem).expanduser()
    flist = sorted([f for f in stem.glob(pat) if f.is_file()])
    if not flist:
        raise FileNotFoundError(f"no files found under {stem} using {pat}")
    return flist


def preprocess(old_frame_path, new_frame_path, sigma):
    # TODO: gaussian filter parameter testen, welke sigma het beste is
    im1 = imageio.imread(old_frame_path, as_gray=True)
    im1_blur = gaussian_filter(im1, sigma=sigma)

    im2 = imageio.imread(new_frame_path, as_gray=True)
    im2_blur = gaussian_filter(im2, sigma=sigma)
    return im1, im1_blur, im2, im2_blur


def HornSchunck(
        im1: np.ndarray,
        im2: np.ndarray,
        *,
        alpha: float = 0.001,
        Niter: int = 8,
        verbose: bool = False,
        save_location: Path = None,
        fn: Path = None
) -> tuple[np.ndarray, np.ndarray]:
    """
    Parameters
    ----------
    im1: numpy.ndarray
        image at t=0
    im2: numpy.ndarray
        image at t=1
    alpha: float
        regularization constant
    Niter: int
        number of iteration
    verbose: bool
        Print dingen
    save_location:
        save hier de plots
    fn:
        naam van frame
    """
    im1 = im1.astype(np.float32)
    im2 = im2.astype(np.float32)

    # set up initial velocities
    uInitial = np.zeros([im1.shape[0], im1.shape[1]], dtype=np.float32)
    vInitial = np.zeros([im1.shape[0], im1.shape[1]], dtype=np.float32)

    # Set initial value for the flow vectors
    U = uInitial
    V = vInitial

    # Estimate derivatives
    [fx, fy, ft] = computeDerivatives(im1, im2)

    # Iteration to reduce error
    for _ in range(Niter):
        # %% Compute local averages of the flow vectors
        uAvg = convolve2d(U, HSKERN, "same")
        vAvg = convolve2d(V, HSKERN, "same")
        # %% common part of update step
        der = (fx * uAvg + fy * vAvg + ft) / (alpha ** 2 + fx ** 2 + fy ** 2)
        # %% iterative step
        U = uAvg - fx * der
        V = vAvg - fy * der

    return U, V


def computeDerivatives(
        im1: np.ndarray, im2: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    fx = convolve2d(im1, kernelX, "same") + convolve2d(im2, kernelX, "same")
    fy = convolve2d(im1, kernelY, "same") + convolve2d(im2, kernelY, "same")

    # ft = im2 - im1
    ft = convolve2d(im1, kernelT, "same") + convolve2d(im2, -kernelT, "same")

    return fx, fy, ft
