import os
import cv2
import numpy as np

def main():
    print(f"sss")

def get_image_file_paths(directory):
    image_file_paths = []
    for file in os.listdir(directory):
        file_path = os.path.join(directory, file)
        if os.path.isfile(file_path):
            if not cv2.imread(file_path, cv2.IMREAD_UNCHANGED) is None:
                image_file_paths.append(file_path)
    return image_file_paths

def image_expand(image, new_shape):
    result = np.zeros(new_shape, dtype=np.uint8)
    result[:image.shape[0], :image.shape[1], :image.shape[2]] = image
    return result

def shifting(image, d_x, d_y):
    h, w = image.shape[:2]
    translation_matrix = np.float32([[1, 0, d_y], [0, 1, d_x]])
    result = cv2.warpAffine(image, translation_matrix, (w, h))
    return result

def append_image(panorama, image_file_path):
    image = cv2.imread(image_file_path)

    orb = cv2.ORB_create()
    panorama_kp, panorama_des = orb.detectAndCompute(panorama, None)
    image_kp, image_des = orb.detectAndCompute(image, None)
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = matcher.match(panorama_des, image_des)
    matches = sorted(matches, key=lambda x: x.distance)

    (panorama_w, panorama_h) = (panorama.shape[1], panorama.shape[0])
    (image_w, image_h) = (image.shape[1], image.shape[0])
    panorama_kp_pts = panorama_kp[matches[0].queryIdx].pt
    image_kp_pts = image_kp[matches[0].trainIdx].pt
    result_kp_pts = (max(panorama_kp_pts[0], image_kp_pts[0]), max(panorama_kp_pts[1], image_kp_pts[1]))
    result_w = max( result_kp_pts[0] + panorama_w - panorama_kp_pts[0], result_kp_pts[0] + image_w - image_kp_pts[0] )
    result_h = max( result_kp_pts[1] + panorama_h - panorama_kp_pts[1], result_kp_pts[1] + image_h - image_kp_pts[1] )
    result_shape = (int(result_h),int(result_w),3)

    #cv2.imshow("Matches", cv2.resize(cv2.drawMatches(panorama, panorama_kp, image, image_kp, matches[:1], None), (1600, 900)))
    #cv2.waitKey()

    panorama = image_expand(panorama, result_shape)
    panorama = shifting(panorama, result_kp_pts[1]-panorama_kp_pts[1], result_kp_pts[0]-panorama_kp_pts[0])
    image = image_expand(image, result_shape)
    image = shifting(image, result_kp_pts[1] - image_kp_pts[1], result_kp_pts[0] - image_kp_pts[0])

    panorama2gray = cv2.cvtColor(panorama, cv2.COLOR_BGR2GRAY)
    ret, mask = cv2.threshold(panorama2gray, 10, 255, cv2.THRESH_BINARY)
    mask_inv = cv2.bitwise_not(mask)
    image_bg = cv2.bitwise_and(image, image, mask=mask_inv)
    panorama_fg = cv2.bitwise_and(panorama, panorama, mask=mask)

    result = np.zeros(result_shape, np.uint8)
    result = cv2.add(result, image_bg)
    result = cv2.add(result, panorama_fg)
    return result

final_image = []
i = 0;
for image_file_path in get_image_file_paths("1"):
    if i == 0:
        final_image = cv2.imread(image_file_path, cv2.IMREAD_UNCHANGED)
    else:
        final_image = append_image(final_image, image_file_path)
    i = i + 1
final_image = cv2.resize(final_image, (1600, 900))
cv2.imshow("Panorama", final_image)
cv2.waitKey()
