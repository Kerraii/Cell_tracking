import cv2
import os
from pathlib import Path
from tqdm import tqdm


def avi_to_frames(avi, overwrite=True, frame_dir_name='frames', normalize=False, remove_scale=True):
    while "src" in os.getcwd():
        os.chdir("..")
    base_path = os.path.join('data')
    vidcap = cv2.VideoCapture(os.path.join(base_path, avi))
    frame_dir_name = os.path.join(base_path, frame_dir_name)
    if not os.path.exists(base_path):
        os.mkdir(base_path)
    if not os.path.exists(frame_dir_name):
        os.mkdir(frame_dir_name)
    if not os.path.exists(os.path.join(frame_dir_name, avi)):
        os.mkdir(os.path.join(frame_dir_name, avi))

    # EMPTY or overwriting images and dir is not empty
    if not os.listdir(os.path.join(frame_dir_name, avi)) or (overwrite and os.listdir(os.path.join(frame_dir_name, avi))):
        for file in os.listdir(os.path.join(frame_dir_name, avi)):
            os.remove(os.path.join(frame_dir_name, avi, file))
        count = 0
        success, image = vidcap.read()
        while success:
            if normalize:
                # het saven van de image doet verkeerde dingen als hij normalized is
                image = cv2.normalize(image, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
                cv2.imshow('s', image)
                cv2.waitKey(0)

            # todo verslag remove scale gele pixels naar een random waarde (donker groen die al voorkwam in de afbeelding A2_mo4_1x_5.avi)
            if remove_scale:
                for index1, image_ in enumerate(image[400:]):
                    for index2, im in enumerate(image_[500:]):
                        if im[1] > 80 and im[2] > 80:
                            image[400+index1][500+index2] = [6, 23, 0]

            cv2.imwrite(os.path.join(frame_dir_name, avi, f"{count:06}.jpg"), image)  # save frame as JPEG file
            success, image = vidcap.read()
            count += 1
    return os.path.join(frame_dir_name, avi)


def frames_to_avi(frame_dir, save_path, overwrite=True):
    if os.path.exists(save_path) and not overwrite:
        return

    img_array = []
    for filename in sorted([f for f in os.listdir(frame_dir)]):
        img = cv2.imread(os.path.join(frame_dir, filename))
        height, width, layers = img.shape
        size = (width, height)
        img_array.append(img)

    out = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'DIVX'), 6, size)

    for i in range(len(img_array)):
        out.write(img_array[i])
    out.release()


def main():
    # Execution from project root
    while "src" in os.getcwd():
        os.chdir("..")

    pathlist = list(Path("data").rglob('*.avi'))
    for path in tqdm(pathlist, desc="Processing files", total=len(pathlist), leave=True):
        # because path is object not string
        if not os.path.isdir(path):
            basename = os.path.basename(path)
            avi_to_frames(basename, remove_scale=True)


# Zal alle .avi-bestanden uit de data-directory omzetten in frames
if __name__ == '__main__':
    main()
