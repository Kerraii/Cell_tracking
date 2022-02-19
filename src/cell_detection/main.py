from animation import *
from src.data_parsing.parse_data import avi_to_frames


def run_one(file_name, log=False, blur=False, blob_mode=False):
    if log:
        print(f"{file_name}: FRAMES")
    # make frames for the avi
    frame_dir_original = avi_to_frames(file_name)

    # create animations
    if log and blur:
        print(f"{file_name}: BLURRED")
    if log and blob_mode:
        print(f"{file_name}: BLOBS ONLY")

    if log:
        print(f"{file_name}: {MODE_LAPLACIAN_OF_GAUSSIAN}")
    create_animation(frame_dir_original, file_name, min_sigma=1, max_sigma=5, threshold=.1,
                     mode=MODE_LAPLACIAN_OF_GAUSSIAN, blur=blur, blur_strength=9, blob_mode=blob_mode)

    if log:
        print(f"{file_name}: {MODE_DIFFERENCE_OF_GAUSSIAN}")
    create_animation(frame_dir_original, file_name, min_sigma=1, max_sigma=5, threshold=.1,
                     mode=MODE_DIFFERENCE_OF_GAUSSIAN, blur=blur, blur_strength=9, blob_mode=blob_mode)

    if log:
        print(f"{file_name}: {MODE_DETERMINANT_OF_HESSIAN}")
    create_animation(frame_dir_original, file_name, min_sigma=0.5, max_sigma=5, threshold=.005,
                     mode=MODE_DETERMINANT_OF_HESSIAN, blur=blur, blur_strength=9, blob_mode=blob_mode)

    # clean frames
    for old_frame in os.listdir(frame_dir_original):
        os.remove(os.path.join(frame_dir_original, old_frame))

    if log:
        print(f"{file_name}: DONE\n")


def run_all(log=False, blur=None, blob_mode=False):
    for f in os.listdir(os.path.join('..', 'data')):
        if f.endswith('.avi'):
            if blur is None:
                run_one(f, log, blur=False, blob_mode=blob_mode)
                run_one(f, log, blur=True, blob_mode=blob_mode)
            else:
                run_one(f, log, blur=blur, blob_mode=blob_mode)


if __name__ == '__main__':
    run_all(True)
    run_all(True, blob_mode=True)
    # test_file_name = 'A2_mo4_1x_5.avi'
    # run_one(test_file_name, log=True, blur=True, blob_mode=False)

