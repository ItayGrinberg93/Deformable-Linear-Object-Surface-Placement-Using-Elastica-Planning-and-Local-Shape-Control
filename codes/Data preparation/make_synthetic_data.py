import time
import numpy as np
import matplotlib.pyplot as plt
from help_funcs import get_shape
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
import multiprocessing as mp
from itertools import product

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




def get_shape_image1(L_gal, S0, mu):
    L = 1
    s = np.linspace(start=0, stop=L)
    [x, y, phi] = get_shape(0, 0, 0, mu=mu, L_gal=L_gal, s0=S0, s=s)
    buf = BytesIO()
    # Save the image
    plt.plot(x, y, '-k')
    plt.axis('equal')
    plt.axis('off')
    plt.savefig(buf, dpi=75)
    plt.clf()
    buf.seek(0)
    pil_image = Image.open(buf).convert('RGB')
    return pil_image


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
    norm_j = (S0 - S0_min) / (L_gal_max - S0_min)
    norm_k = (mu-mu_min)/(mu_max-mu_min) # 2*((mu - mu_min) / (mu_max - mu_min))-1b
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

def process_shape(args):
    L_gal, s0, mu, output_path = args
    norm_i = (L_gal-L_gal_min)/(L_gal_max-L_gal_min)
    norm_j = (s0-S0_min)/(L_gal_max-S0_min)
    norm_k = (mu-mu_min)/(mu_max-mu_min)
    image_path = output_path/f'{norm_i}_{norm_j}_{norm_k}.jpg'
    
    if image_path.exists():
        return
        
    fig = plt.figure(num=1, dpi=64)
    img = get_shape_image(mu=mu, L_gal=L_gal, S0=s0, fig=fig)
    img = imageprosses(img, fig)
    
    plt.imshow(img, cmap=plt.cm.gray)
    plt.axis('equal')
    plt.axis('off')
    plt.savefig(image_path, dpi=64)
    plt.close(fig)
    return image_path

if __name__ == '__main__':
    # LDO parameters
    L = 1
    s = np.linspace(start=0, stop=L)
    mu_min = -0.462
    mu_max = 1
    L_gal_max = 4*L
    L_gal_min = 1*L
    S0_max = L_gal_max
    S0_min = 0
    index_mu = 150
    index_s0 = 75
    index_L_gal = 75
    mus = np.linspace(start=mu_min, stop=mu_max, num=index_mu, endpoint=True)
    L_gals = np.linspace(start=L_gal_min, stop=L_gal_max, num=index_L_gal, endpoint=True)
    output_path = Path('./synthetic_data_Normalized_pipeline_Large_withutils')
    output_path.mkdir(parents=True, exist_ok=True)

    # Create work items
    work_items = []
    for L_gal in L_gals:
        S0s = np.linspace(start=0, stop=L_gal, num=index_s0, endpoint=True)
        work_items.extend([
            (L_gal, s0, mu, output_path) 
            for s0, mu in product(S0s, mus)
        ])

    # Set up multiprocessing
    num_processes = mp.cpu_count()
    print(f"Starting processing with {num_processes} processes")
    
    # Create pool and process items
    with mp.Pool(processes=num_processes) as pool:
        for _ in tqdm(
            pool.imap_unordered(process_shape, work_items),
            total=len(work_items),
            desc="Generating shapes"
        ):
            pass









