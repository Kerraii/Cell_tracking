from skimage.feature import blob_dog, blob_log, blob_doh
from skimage.color import rgb2gray
import cv2
import matplotlib.pyplot as plt
from math import sqrt
import src.cell_detection.MSER as mser
import os
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)


def compare_algo(img_path_l, blobs_list, titles, colors=None, save_path=None, show=False):
    if not isinstance(img_path_l, list):
        img_path_l = [img_path_l]*len(blobs_list)

    images = [cv2.imread(img_path) for img_path in img_path_l]

    if colors is None:
        colors = ['purple', 'red', 'pink']

    fig, axes = plt.subplots(1, len(blobs_list), figsize=(9, 9//len(blobs_list)), sharex=True, sharey=True)
    ax = axes.ravel()
    for idx, (blobs, color, title) in enumerate(zip(blobs_list, colors, titles)):
        ax[idx].set_title(title)
        ax[idx].imshow(images[idx])
        for blob in blobs:
            y, x, r = blob
            c = plt.Circle((x, y), r, color=color, linewidth=0.25, fill=False)
            ax[idx].add_patch(c)
        ax[idx].set_axis_off()
    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path)
    if show:
        plt.show()
    plt.close()


def laplacian_of_gaussian(img_path, min_sigma=1, max_sigma=5, threshold=.1, color='pink', save_path=None, show=False, blur=False, blur_strength=9, blob_mode=False):
    image = cv2.imread(img_path)
    blobs = blob_laplacian_of_gaussian_calc(img_path, min_sigma=min_sigma, max_sigma=max_sigma, threshold=threshold, blur=blur, blur_strength=blur_strength)
    if blob_mode:
        save_blobs(image=image, blobs=blobs, color=color, title='Blobs Laplacian of Gaussian', save_path=save_path, show=show, fill=True)
    else:
        save_plot(image=image, blobs=blobs, color=color, title='Laplacian of Gaussian', save_path=save_path, show=show)


def difference_of_gaussian(img_path, min_sigma=1, max_sigma=5, threshold=.1, color='purple', save_path=None, show=False, blur=False, blur_strength=9, blob_mode=False):
    image = cv2.imread(img_path)
    blobs = blob_difference_of_gaussian_calc(img_path, min_sigma=min_sigma, max_sigma=max_sigma, threshold=threshold, blur=blur, blur_strength=blur_strength)
    if blob_mode:
        save_blobs(image=image, blobs=blobs, color=color, title='Blobs Difference of Gaussian', save_path=save_path, show=show, fill=True)
    else:
        save_plot(image=image, blobs=blobs, color=color, title='Difference of Gaussian', save_path=save_path, show=show)


def determinant_of_hessian(img_path, min_sigma=0.5, max_sigma=5, threshold=.005, color='red', save_path=None, show=False, blur=False, blur_strength=9, blob_mode=False):
    image = cv2.imread(img_path)
    blobs = blob_determinant_of_hessian_calc(img_path, min_sigma=min_sigma, max_sigma=max_sigma, threshold=threshold, blur=blur, blur_strength=blur_strength)
    if blob_mode:
        save_blobs(image=image, blobs=blobs, color=color, title='Blobs Determinant of Hessian', save_path=save_path, show=show, fill=True)
    else:
        save_plot(image=image, blobs=blobs, color=color, title='Determinant of Hessian', save_path=save_path, show=show)


# Laplacian of Gaussian
# https://scikit-image.org/docs/dev/api/skimage.feature.html#skimage.feature.blob_log
def blob_laplacian_of_gaussian_calc(img_path, min_sigma=1, max_sigma=5, threshold=.1, blur=False, blur_strength=9):
    image = cv2.imread(img_path)
    if blur:
        image = cv2.GaussianBlur(image, (blur_strength, blur_strength), cv2.BORDER_DEFAULT)
    img_gray = rgb2gray(image)
    blobs_log = blob_log(img_gray, min_sigma=min_sigma, max_sigma=max_sigma, threshold=threshold)
    blobs_log[:, 2] = blobs_log[:, 2] * sqrt(2)
    return blobs_log


# Difference of Gaussian
# https://scikit-image.org/docs/dev/api/skimage.feature.html#skimage.feature.blob_dog
def blob_difference_of_gaussian_calc(img_path, min_sigma=1, max_sigma=5, threshold=.1, blur=False, blur_strength=9):
    image = cv2.imread(str(img_path))
    if blur:
        image = cv2.GaussianBlur(image, (blur_strength, blur_strength), cv2.BORDER_DEFAULT)
    img_gray = rgb2gray(image)
    blobs_dog = blob_dog(img_gray, min_sigma=min_sigma, max_sigma=max_sigma, threshold=threshold)
    blobs_dog[:, 2] = blobs_dog[:, 2] * sqrt(2)
    return blobs_dog


# Determinant of Hessian
# https://scikit-image.org/docs/dev/api/skimage.feature.html#skimage.feature.blob_doh
def blob_determinant_of_hessian_calc(img_path, min_sigma=0.5, max_sigma=5, threshold=.005, blur=False, blur_strength=9):
    image = cv2.imread(img_path)
    if blur:
        image = cv2.GaussianBlur(image, (blur_strength, blur_strength), cv2.BORDER_DEFAULT)
    img_gray = rgb2gray(image)
    return blob_doh(img_gray, min_sigma=min_sigma, max_sigma=max_sigma, threshold=threshold)


def save_plot(image, blobs, color, title='NO TITLE', save_path=None, show=False, fill=False, height=None, width=None):
    fig, ax = plt.subplots(figsize=(9, 9))
    ax.xaxis.tick_top()
    ax.set_title(title)
    if height and width:
        ax.set_ylim([0, height])
        ax.set_xlim([0, width])
        ax.invert_yaxis()
    if image is not None:
        plt.imshow(image)
    for blob in blobs:
        y, x, r = blob
        c = plt.Circle((x, y), r, color=color, linewidth=1, fill=fill)
        ax.add_patch(c)
    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path)
    if show:
        plt.show()
    plt.close()


def save_blobs(image, blobs, color, title='NO TITLE', save_path=None, show=False, fill=False):
    height, width, _ = image.shape
    save_plot(image=None, blobs=blobs, color=color, title=title, save_path=save_path, show=show, fill=fill, height=height, width=width)


# LoG zonder MSER levert beste resultaten op
def opt_blobs(frame_path):
    return blob_laplacian_of_gaussian_calc(frame_path, min_sigma=3, max_sigma=6, threshold=0.07)


# DoG met MSER levert ook goede resulaten op
def opt_blobs2(frame_path):
    tmp = 'tmp{}.jpg'.format(frame_path.split(os.sep)[-1])
    mser.MSER(frame_path, save_path=tmp, delta=5, min_area=14, max_area=5000)
    blobs = blob_difference_of_gaussian_calc(tmp, min_sigma=1, max_sigma=6, threshold=0.05)
    os.remove(tmp)
    return blobs


if __name__ == '__main__':
    import os
    from src.data_parsing.parse_data import avi_to_frames

    while "src" in os.getcwd():
        os.chdir("..")

    file = 'D1_5fps_1.5 x magnification_2004.avi'
    # file = 'A2_mo4_1x_5.avi'
    frame_dir = avi_to_frames(file)
    frame = os.path.join(frame_dir, '000000.jpg')
    print(opt_blobs(frame))
    # difference_of_gaussian(frame, show=True, color='red', threshold=0.03)
