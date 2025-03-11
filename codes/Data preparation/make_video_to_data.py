import time
import numpy as np
import matplotlib.pyplot as plt
from utils import get_shape
from pathlib import Path
from tqdm import tqdm
from io import BytesIO
from PIL import Image
import cv2 as cv
from skimage.morphology import skeletonize , thin, medial_axis
from skimage import data
from skimage.util import invert
from skimage.util import img_as_ubyte
from skimage import feature
from skimage.measure import find_contours, approximate_polygon, subdivide_polygon
from skimage.morphology import reconstruction
from skimage.filters import threshold_sauvola
import pickle

def imageprosses(img,fig):

#Open image file

    # edge ditection
    img_gray = cv.cvtColor(img, cv.COLOR_RGB2GRAY)

    ret1, th1 = cv.threshold(img_gray, 128, 255, cv.THRESH_BINARY)

    thresh_sauvola = threshold_sauvola(th1, window_size=15)

    binary_sauvola = th1 > thresh_sauvola
    image1 = invert(binary_sauvola)


    # perform skeletonization
    skeleton = skeletonize(image1, method='lee')

    datax = []
    datay = []
    # approximate subdivided polygon with Douglas-Peucker algorithm
    for contour in find_contours(skeleton, 0,fully_connected='low'):
        coords2 = approximate_polygon(contour, tolerance=1)
        plt.plot(coords2[:, 1], -1*coords2[:, 0], '-k',linewidth = '2')
 
    buf = BytesIO()
    plt.axis('off')
    plt.axis('equal')
    fig.savefig(buf, dpi=75)
    plt.clf()
    buf.seek(0)
    image = Image.open(buf)
    return image




def get_shape_image(L_gal, S0, mu,fig):
    L = 1
    s = np.linspace(start=0, stop=L)
    [x, y, phi] = get_shape(0, 0, 0, mu=mu, L_gal=L_gal, s0=S0, s=s)
    buf = BytesIO()
    # Save the image
    plt.plot(x, y, '-k')
    plt.axis('equal')
    plt.axis('off')
    fig.savefig(buf, dpi=75)
    plt.clf()
    buf.seek(0)
    pil_image = Image.open(buf).convert('RGB')
    open_cv_image = np.array(pil_image)
    # Convert RGB to BGR
    img = open_cv_image[:, :, ::-1].copy()
    return img


def get_shape_image2(L_gal, S0, mu, fig):
    L = 1
    s = np.linspace(start=0, stop=L)
    [x, y, phi] = get_shape(0, 0, 0, mu=mu, L_gal=L_gal, s0=S0, s=s)
    norm_i = (L_gal - L_gal_min) / (L_gal_max - L_gal_min)
    norm_j = (s0 - S0_min) / (L_gal_max - S0_min)
    norm_k = (mu - mu_min) / (mu_max - mu_min)
    image_path = output_path / f'{norm_i}_{norm_j}_{norm_k}.jpg'  # L_gal_S0_MU

    # Save the image
    ax = fig.add_subplot()
    ax.plot(x, y, '-k')
    # Save the image
    ax.axis('equal')
    ax.axis('off')
    ax.figure.savefig(image_path, dpi=150)
    ax.clear()
    ax.axis('off')
    return


def zoom_center(img, zoom_factor=1.5):
    pil_image = img.convert('RGB')
    open_cv_image = np.array(pil_image)
    # Convert RGB to BGR
    img = open_cv_image[:, :, ::-1].copy()

    y_size = img.shape[0]
    x_size = img.shape[1]

    # define new boundaries
    x1 = int(0.5 * x_size * (1 - 1 / zoom_factor))
    x2 = int(x_size - 0.5 * x_size * (1 - 1 / zoom_factor))
    y1 = int(0.5 * y_size * (1 - 1 / zoom_factor))
    y2 = int(y_size - 0.5 * y_size * (1 - 1 / zoom_factor))

    # first crop image then scale
    img_cropped = img[y1:y2, x1:x2]
    return cv.resize(img_cropped, None, fx=zoom_factor, fy=zoom_factor)


def rotate_image(img,degrees):
    pil_image = img.convert('RGB')
    open_cv_image = np.array(pil_image)
    # Convert RGB to BGR
    img = open_cv_image[:, :, ::-1].copy()

    rows, cols = img.shape
    # cols-1 and rows-1 are the coordinate limits.
    M = cv.getRotationMatrix2D(((cols - 1) / 2.0, (rows - 1) / 2.0), degrees, 1)
    dst = cv.warpAffine(img, M, (cols, rows))
    return dst



if __name__ == '__main__':
 
    output_path = Path('./Path_To_Videos_Folder/')
    folder_path = output_path/'Output_folder_For_Data'
    folder_path.mkdir(parents=True, exist_ok=True)
    #Open video file
    video_path = output_path/'Video_name.mp4' 
    fig = plt.figure(num=1,dpi=75)
    cap = cv.VideoCapture(str(video_path))
    num_of_frame = 70
    count = 0
    list_path = './Data_Path_conected_To_Video.pkl'
    with open(list_path, 'rb') as f:
        mynewlist = pickle.load(f)
     # Loop through the video frame
    while cap.isOpened():
        # Read the frame from video
        success, frame = cap.read()
        cap.set(cv.CAP_PROP_POS_FRAMES, num_of_frame)
        if success:
            num_of_frame = num_of_frame + (65 - count)
            start = time.perf_counter()
            end = time.perf_counter()

            total_time = end - start
            fps = 1 / total_time
            gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
            image_path = folder_path/f'{count}_{mynewlist[count][0]}_{mynewlist[count][1]}_{mynewlist[count][2]}:{mynewlist[count][3]}_{mynewlist[count][4]}_{mynewlist[count][5]}_{mynewlist[count][6]}_{mynewlist[count][7]}:{mynewlist[count][8]}_{mynewlist[count][9]}_{mynewlist[count][10]}.jpg' 
            count = count + 1
            cv.imshow('frame', frame)
            cv.imwrite(str(image_path), frame, [cv.IMWRITE_JPEG_QUALITY, 100])
        if cv.waitKey(1) == ord('q'):
            break
    
    cap.release()
    cv.destroyAllWindows()
    







