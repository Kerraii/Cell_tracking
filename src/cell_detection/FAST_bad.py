import cv2
import os
from src.data_parsing.parse_data import avi_to_frames


# LETS NOT USE THIS :)
# todo uitzoeken waarom dit het niet zo goed doet en in verslag
def FAST(frame):

    img = cv2.imread(frame, 0)

    # Initiate FAST object with default values
    fast = cv2.FastFeatureDetector_create()

    fast.setThreshold(20)

    # find and draw the keypoints
    kp = fast.detect(img, None)
    img2 = cv2.drawKeypoints(img, kp, img, color=(255, 0, 0))

    # Print all default params
    print("Threshold: " + str(fast.getThreshold()))
    print("nonmaxSuppression: " + str(fast.getNonmaxSuppression()))
    print("neighborhood: " + str(fast.getType()))
    print("Total Keypoints with nonmaxSuppression: " + str(len(kp)))

    cv2.imwrite('fast_true.png', img2)

    # Disable nonmaxSuppression
    fast.setNonmaxSuppression(0)
    kp = fast.detect(img, None)

    print("Total Keypoints without nonmaxSuppression: ", str(len(kp)))

    img3 = cv2.drawKeypoints(img, kp, img, color=(255, 0, 0))

    cv2.imwrite('fast_false.png', img3)


if __name__ == '__main__':
    file_name = 'A2_mo4_1x_5.avi'
    frame_dir = avi_to_frames(file_name)
    FAST(os.path.join(frame_dir, '000000.jpg'))