from src.data_parsing.parse_data import frames_to_avi
from src.cell_detection.detect_blob import difference_of_gaussian, laplacian_of_gaussian, determinant_of_hessian
import os


MODE_LAPLACIAN_OF_GAUSSIAN = 'LAPLACIAN_OF_GAUSSIAN'
MODE_DIFFERENCE_OF_GAUSSIAN = 'DIFFERENCE_OF_GAUSSIAN'
MODE_DETERMINANT_OF_HESSIAN = 'DETERMINANT_OF_HESSIAN'


def create_animation(frame_dir, file_name, min_sigma=1, max_sigma=5, threshold=.1,
                     mode=MODE_DIFFERENCE_OF_GAUSSIAN, overwrite=True, blur=False, blur_strength=9, blob_mode=False):
    if not os.path.exists('animation'):
        os.mkdir('animation')
    if not os.path.exists(os.path.join('animation', file_name)):
        os.mkdir(os.path.join('animation', file_name))
    if not os.path.exists(os.path.join('animation', file_name, 'frames')):
        os.mkdir(os.path.join('animation', file_name, 'frames'))

    for old_frame in os.listdir(os.path.join('animation', file_name, 'frames')):
        os.remove(os.path.join('animation', file_name, 'frames', old_frame))

    if not overwrite and os.path.exists(os.path.join('animation', file_name, mode + '.avi')):
        return False

    if mode == MODE_LAPLACIAN_OF_GAUSSIAN:
        for new_frame in os.listdir(frame_dir):
            laplacian_of_gaussian(os.path.join(frame_dir, new_frame),
                                  save_path=os.path.join('animation', file_name, 'frames', new_frame),
                                  min_sigma=min_sigma,
                                  max_sigma=max_sigma,
                                  threshold=threshold,
                                  blur=blur,
                                  blur_strength=blur_strength,
                                  blob_mode=blob_mode
                                  )

    elif mode == MODE_DIFFERENCE_OF_GAUSSIAN:
        for new_frame in os.listdir(frame_dir):
            difference_of_gaussian(os.path.join(frame_dir, new_frame),
                                   save_path=os.path.join('animation', file_name, 'frames', new_frame),
                                   min_sigma=min_sigma,
                                   max_sigma=max_sigma,
                                   threshold=threshold,
                                   blur=blur,
                                   blur_strength=blur_strength,
                                   blob_mode=blob_mode
                                   )

    elif mode == MODE_DETERMINANT_OF_HESSIAN:
        for new_frame in os.listdir(frame_dir):
            determinant_of_hessian(os.path.join(frame_dir, new_frame),
                                   save_path=os.path.join('animation', file_name, 'frames', new_frame),
                                   min_sigma=min_sigma,
                                   max_sigma=max_sigma,
                                   threshold=threshold,
                                   blur=blur,
                                   blur_strength=blur_strength,
                                   blob_mode=blob_mode
                                   )

    else:
        print('Invalid mode')
        return False

    if blur:
        mode += '_BLUR'

    if blob_mode:
        mode = 'BLOB_' + mode

    frames_to_avi(os.path.join('animation', file_name, 'frames'), os.path.join('animation', file_name, mode + '.avi'))

    with open(os.path.join('animation', file_name, mode + ".txt"), "w+") as f:
        f.write(f'min_sigma={str(min_sigma)}\nmax_sigma={str(max_sigma)}\nthreshold={str(threshold)}\nblur={str(blur)}\nblur_strength={blur_strength}')

    for old_frame in os.listdir(os.path.join('animation', file_name, 'frames')):
        os.remove(os.path.join('animation', file_name, 'frames', old_frame))

    return True


if __name__ == '__main__':
    f_name = 'A2_mo4_1x_5.avi'
    f_dir = os.path.join('frames', f_name)
    create_animation(f_dir, f_name)
