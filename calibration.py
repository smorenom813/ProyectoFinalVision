from typing import List
import numpy as np
import imageio
import cv2
import copy
import glob
import os

def load_images(filenames: List) -> List:
    return [imageio.imread(filename) for filename in filenames]

def show_image(image,i):
    cv2.imshow(f"foto{i}",image)

    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
def write_image(image,i):
    directorio_new = f"{directorio}\corners"
    
    if not os.path.exists(directorio_new):
        os.makedirs(directorio_new)
    filename = f"{directorio_new}\img{i}.jpg"

    if not os.path.exists(filename):
        cv2.imwrite(filename,image)
    
def get_chessboard_points(chessboard_shape, dx, dy):

    points = []
    
    for i in range(chessboard_shape[1]):
        for j in range(chessboard_shape[0]):
            points.append([dx*i,dy*j,0])
    return points 

# TODO Build a list containing the paths of all images from the left camera
directorio = "data"

imgs_path_raw = os.listdir(directorio)
imgs_path = [f"{directorio}/{img}" for img in  imgs_path_raw if img != "corners"]
imgs = load_images(imgs_path)

# TODO Find corners with cv2.findChessboardCorners()
corners = [cv2.findChessboardCorners(img,(7,7))for img in imgs]

corners_copy = copy.deepcopy(corners)
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 24, 0.01)

# TODO To refine corner detections with cv2.cornerSubPix() you need to input grayscale images. Build a list containing grayscale images.

imgs_gray = [cv2.cvtColor(img,cv2.COLOR_BGR2GRAY) for img in imgs]

corners_refined = [cv2.cornerSubPix(i, cor[1], (7, 7), (-1, -1), criteria) if cor[0] else [] for i, cor in zip(imgs_gray, corners_copy)]

imgs_copy = copy.deepcopy(imgs)

# TODO Use cv2.drawChessboardCorners() to draw the cornes
imgs_draw = [cv2.drawChessboardCorners(img,(7,7),np.array(corner),True) for img,corner in zip(imgs_copy,corners_refined)]


for i in range(len(imgs_draw)):
    show_image(imgs_draw[i],i)
    write_image(imgs_draw[i],i)

# TODO You need the points for every image, not just one
chessboard_points = np.asarray(get_chessboard_points((7, 7), 24, 24), dtype=np.float32)
list_chessboard_points = np.asarray([chessboard_points for _ in range(len(corners_refined))], dtype=np.float32)
print(len(list_chessboard_points))

# Filter data and get only those with adequate detections
valid_corners = [cor[1] for cor in corners if cor[0]]
# Convert list to numpy array
valid_corners = np.asarray(valid_corners, dtype=np.float32)
len(valid_corners)

# Como la primera foto no es "valid" tenemos que quitar el primer elemento de list_chessboard_points
rms, intrinsics, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(list_chessboard_points,valid_corners,(imgs[1].shape[1],imgs[1].shape[0]),None,None)

# Obtain extrinsics
extrinsics = list(map(lambda rvec, tvec: np.hstack((cv2.Rodrigues(rvec)[0], tvec)), rvecs, tvecs))

# Print outputs
print("Intrinsics:\n", intrinsics)
print("Distortion coefficients:\n", dist_coeffs)
print("Root mean squared reprojection error:\n", rms)
