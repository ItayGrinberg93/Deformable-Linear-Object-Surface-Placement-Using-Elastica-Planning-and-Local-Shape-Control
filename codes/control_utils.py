from skimage.morphology import skeletonize , thin, medial_axis
from skimage.util import invert
from skimage.util import img_as_ubyte
from skimage.measure import find_contours, approximate_polygon, subdivide_polygon
from skimage.filters import threshold_sauvola
import time
from pathlib import Path
import seaborn as sns
from scipy.interpolate import make_interp_spline, BSpline
import matplotlib.pyplot as plt
import numpy as np
from fastai.vision.all import *
from pathlib import Path
from help_funcs import get_shape as get_shape_hf
import cv2 as cv
from sklearn.metrics import mean_squared_error 
from PIL import Image
from fastai.vision.all import *
import dill
from ultralytics import YOLO
from PIL import Image


output_path = Path('path_to_buffer_folder/data_real_shapes/test_buffer')
output_path.mkdir(parents=True, exist_ok=True)
buf_path = output_path.joinpath('buf_image.jpg')
buf_path1 = output_path / f'buf_image1.jpg'
buf_path2 = output_path.joinpath('buf_image2.jpg')
buf_path3 = output_path / f'buf_image3.jpg'
buf_path4 = output_path.joinpath('buf_image4.jpg')


def metric_get_shape(inp, targ):
    "The error between each shape of `inp` and `targ`."
    f_tot = 0
    s = np.linspace(start=0, stop=1)
    for inp1, out1 in zip(inp, targ):
        L_gal_in, S0_in, mu_in = unormlized(inp1[0].cpu(), inp1[1].cpu(), inp1[2].cpu())
        L_gal_out, S0_out, mu_out = unormlized(out1[0].cpu(), out1[1].cpu(), out1[2].cpu())
        [x_in, y_in, phi_in] = get_shape_hf(0, 0, 0, mu=mu_in, L_gal=L_gal_in, s0=S0_in, s=s)
        [x_out, y_out, phi_out] = get_shape_hf(0, 0, 0, mu=mu_out, L_gal=L_gal_out, s0=S0_out, s=s)
        temp = F.mse_loss(x_in, x_out) + F.mse_loss(y_in, y_out)
        if temp.isnan():
            continue
        else:
            f_tot = f_tot + temp
    f_tot = f_tot / len(inp)
    return f_tot.cuda()

def unormlized(L_gal, S0, mu):
    mu_min = -0.462
    mu_max = 1
    L_gal_max = 4
    L_gal_min = 1
    S0_max = L_gal_max
    S0_min = 0
    not_norm_L_gal = L_gal * (L_gal_max - L_gal_min) + L_gal_min
    not_norm_S0 = S0 * (S0_max - S0_min) + S0_min
    not_norm_mu = mu * (mu_max - mu_min) + mu_min
    
    return not_norm_L_gal, not_norm_S0, not_norm_mu

def rotate_image(image, angle):
  image_center = tuple(np.array(image.shape[1::-1]) / 2)
  rot_mat = cv.getRotationMatrix2D(image_center, angle, 1.0)
  result = cv.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv.INTER_LINEAR)
  
  return result

def largest_rotated_rect(w, h, angle):
    """
    Given a rectangle of size wxh that has been rotated by 'angle' (in
    radians), computes the width and height of the largest possible
    axis-aligned rectangle within the rotated rectangle.

    Original JS code by 'Andri' and Magnus Hoff from Stack Overflow

    Converted to Python by Aaron Snoswell
    """

    quadrant = int(math.floor(angle / (math.pi / 2))) & 3
    sign_alpha = angle if ((quadrant & 1) == 0) else math.pi - angle
    alpha = (sign_alpha % math.pi + math.pi) % math.pi

    bb_w = w * math.cos(alpha) + h * math.sin(alpha)
    bb_h = w * math.sin(alpha) + h * math.cos(alpha)

    gamma = math.atan2(bb_w, bb_w) if (w < h) else math.atan2(bb_w, bb_w)

    delta = math.pi - alpha - gamma

    length = h if (w < h) else w

    d = length * math.cos(alpha)
    a = d * math.sin(alpha) / math.sin(delta)

    y = a * math.cos(gamma)
    x = y * math.tan(gamma)

    return (
        bb_w - 2 * x,
        bb_h - 2 * y
    )

def MSE_model_error_test(inp,out,Length, base_frame):
# Get predictions and calculate errors
    labels = inp[:3]
    pred = out[:3]
    shape_error = []
    tangent_error = []
    elastica_true = labels
    elastica_pred = unormlized(*pred)
    elastica_error = np.square(np.subtract(labels,pred)).mean() 
    # Generate shape coordinates
    s_true = np.linspace(start=0, stop=Length)
    [x_true, y_true, phi_true] = get_shape_hf(base_frame[0], base_frame[1], base_frame[2],
                                            elastica_true[0], elastica_true[1], elastica_true[2], 
                                            s=s_true)

    s_pred = np.linspace(start=0, stop=1)
    [x_pred, y_pred, phi_pred] = get_shape_hf(base_frame[0], base_frame[1], base_frame[2],
                                            elastica_pred[0], elastica_pred[1], elastica_pred[2], 
                                            s=s_pred)
    x_pred=(x_pred-x_pred[0])*Length
    y_pred=(y_pred-y_pred[0])*Length
    x_true=(x_true-x_true[0])
    y_true=(y_true-y_true[0])

    plt.plot(x_pred,y_pred, label = 'predicted shape')
    plt.plot(x_true,y_true, label = 'elastica shape')
    # plt.title("Planned DLO shape")
    plt.legend()
    plt.axis('equal')
    plt.axis('off')
    plt.show()

    fig = plt.figure(num=1, dpi=64)
    plt.plot(x_pred,y_pred)
    plt.axis('equal')
    plt.axis('off')
    plt.savefig(buf_path4, dpi=64)
    plt.clf()

    # Print error metrics
    # print(f"Elastica Error: {elastica_error:.4f}")
    for i in range(len(s_true)):
        shape_error.append(np.square((x_true[i]-x_pred[i])**2 +(y_true[i]-y_pred[i])**2))
        tangent_error.append(np.square((phi_true[i]-phi_pred[i])**2))
    shape_error = np.array(shape_error).mean()
    tangent_error = np.array(tangent_error).mean()
    return shape_error, elastica_error, tangent_error

def crop_around_center(image, width, height):
    """
    Given a NumPy / OpenCV 2 image, crops it to the given width and height,
    around it's centre point
    """

    image_size = (image.shape[1], image.shape[0])
    image_center = (int(image_size[0] * 0.5), int(image_size[1] * 0.5))

    if(width > image_size[0]):
        width = image_size[0]

    if(height > image_size[1]):
        height = image_size[1]

    x1 = int(image_center[0] - width * 0.5)
    x2 = int(image_center[0] + width * 0.5)
    y1 = int(image_center[1] - height * 0.5)
    y2 = int(image_center[1] + height * 0.5)

    return image[y1:y2, x1:x2]

def get_label_inx(file_path):
    temp, temp1 = file_path.stem.split(':')
    inx, x0, y0, phi_0 = temp.split('_')
    L_gal, s0, mu, L, l = temp1.split('_')
    labels = [round(float(L_gal), 3), round(float(s0), 3), round(float(mu), 3)]
    base_frame = [float(x0), float(y0), float(phi_0), float(L), float(l)]
    
    return inx, labels, base_frame

def get_label(file_path):
    L_gal, s0, mu = file_path.stem.split('_')
    results = [round(float(L_gal), 3), round(float(s0), 3), round(float(mu), 3)]
    
    return results

def get_label_inx_real(file_path):
    temp, temp1, temp2 = file_path.stem.split(':')
    inx, x0, y0, phi_0 = temp.split('_')
    L_gal, s0, mu, L, l = temp1.split('_')
    x_tcp, y_tcp, phi_tcp = temp2.split('_')
    labels = [round(float(L_gal), 3), round(float(s0), 3), round(float(mu), 3)]
    base_frame = [float(x0), float(y0), float(phi_0), float(L), float(l)]
    TCP_base_frame = [float(x_tcp), float(y_tcp), float(phi_tcp)]
    
    return inx, labels, base_frame, TCP_base_frame 

def TCP_get_shape(xs,ys,phis, L_gal, S0, mu, L):
    Llim = L_gal/4
    if S0 + L <=Llim:
        S0_2 = 2*Llim - (S0+L)
    else:
        while S0 + L >= Llim:
            Llim = Llim + L_gal/2
        Llim = Llim - L_gal/2
        S0_2 = 2*Llim - (S0+L)
    if isinstance(xs, list):
        temp = get_shape_hf(xs[-1], ys[-1], phis[-1]-np.pi, L_gal=L_gal, s0=S0_2, mu=mu, s=np.linspace(start=0, stop=L, num=100))
    else:
        temp = get_shape_hf(xs, ys, phis-np.pi, L_gal=L_gal, s0=S0_2, mu=mu, s=np.linspace(start=0, stop=L, num=100))
    return temp

def crop_model(img,yolo_model):
   frame=img

   # # Run YOLO model for crop image

   start = time.perf_counter()
   # Run YOLOv8 interface on the frame:

   im_pil = Image.fromarray(img)
   # For reversing the operation:

   results = yolo_model(im_pil) 
   end = time.perf_counter()
   total_time = end - start
   fps = 1 / total_time

   # Visualize the results on the frame:
   annotated_frame = results[0].plot()
   # plt.imshow(annotated_frame)
   # plt.axis('off')
   # plt.show()
   conf_ldo = 0
   pix = 20
   if len(results[0].boxes.conf) == 0:
       flag1 = True
       return flag1
   for inx,conf in enumerate(results[0].boxes.conf):
         if conf >= conf_ldo:
              conf_ldo=conf
              inx_box=inx
   x1_lim, y1_lim, x2_lim, y2_lim =results[0].boxes.xyxy[inx_box]
   crop_img = frame[int(y1_lim)-pix:int(y2_lim)+pix, int(x1_lim)-pix:int(x2_lim)+pix]
   # plt.axis('off')
   # plt.imshow(crop_img, cmap=plt.cm.gray)
   # plt.show()
   for inx,conf in enumerate(results[0].boxes.conf):
         if conf >= conf_ldo:
              conf_ldo=conf
              inx_box=inx
   x1_lim, y1_lim, x2_lim, y2_lim =results[0].boxes.xyxy[inx_box]
   x=[]
   y=[]
   for inx,mask in enumerate(results[0].masks.xy):
            for point in mask:
                if (point[0]>=int(x1_lim)-pix)and(point[0]<=int(x2_lim)+pix) and (point[1]>=int(y1_lim)-pix)and(point[1]<int(y2_lim)+pix):
                    x.append(point[0])
                    y.append(-point[1])
   x = np.array(x)
   y = np.array(y)

   plt.fill_between(x,y1=y,y2=int(np.max(y))) 
   plt.axis('off')
   plt.axis('equal')
   plt.savefig(buf_path, dpi=64)
   plt.clf()
   return crop_img


def pipeline(img_src, model):

    # # edge ditection
    img_gray = cv.cvtColor(img_src, cv.COLOR_BGR2GRAY)
    ret1, th1 = cv.threshold(img_gray, 128, 255, cv.THRESH_BINARY)

    thresh_sauvola = threshold_sauvola(th1, window_size=15)


    # Invert the horse image
    binary_sauvola = th1 > thresh_sauvola
    image1 = invert(binary_sauvola)


    # perform skeletonization
    thinned = thin(image1)

    fig = plt.figure(num=1, dpi=64)
    plt.imshow(255-thinned, cmap=plt.cm.gray)
    plt.axis('equal')
    plt.axis('off')
    plt.savefig(buf_path2, dpi=64)
    plt.clf()

    fig1, ax2 = plt.subplots(ncols=1, figsize=(7.2, 9.6)) #
    # approximate subdivided polygon with Douglas-Peucker algorithm
    for contour in find_contours(thinned,0,fully_connected='low'):
        coords = approximate_polygon(contour, tolerance=1)
        coords2 = approximate_polygon(contour, tolerance=1)
        ax2.plot(coords2[:, 1], -1*coords2[:, 0],'-k',linewidth = '2')
    ax2.axis('off')
    ax2.axis('equal')
    ax2.figure.savefig(buf_path1 , dpi=64)
    # ax2.set_title('contour of the shape', fontsize=20)
    # plt.show()
    plt.clf()

    
    img_test1 =cv.imread(str(buf_path1), cv.IMREAD_GRAYSCALE)

    temp = model.predict(img_test1)[0]
    [L_gal,S0,mu]=unormlized(L_gal=temp[0], S0=temp[1], mu=temp[2])
    
    
 # ----------- image for check --------------#  
    fig = plt.figure(num=1, dpi=64)
    [x, y, phi1] = get_shape_hf(0, 0, 0, L_gal=L_gal, s0=S0, mu=mu, s=np.linspace(start=0, stop=1))
    plt.plot(x, y)
    plt.axis('equal')
    plt.axis('off')
    plt.savefig(buf_path3 , dpi=64)
    plt.clf()

# ----------- end --------------# 

    return temp

def align_and_resize_images(image, reference_image, debug=False):
    """Align images while handling both RGB and grayscale"""
    if image is None:
        raise ValueError("Input image is None")
    
    # Get reference dimensions
    ref_height, ref_width = reference_image.shape[:2]
    
    # Calculate aspect ratios
    img_aspect = image.shape[1] / image.shape[0]
    ref_aspect = ref_width / ref_height
    
    # Always scale based on height for aligned_rotated
    if 'rotated' in str(image.shape):  # This will make rotated image wider
        scale = image.shape[0] / ref_height
        new_height = ref_height
        new_width = int(image.shape[1] / scale * 1.5)  # Increase width by 50%
    else:
        if img_aspect > ref_aspect:
            scale = image.shape[1] / ref_width
            new_width = ref_width 
            new_height = int(image.shape[0] / scale)
        else:
            scale = image.shape[0] / ref_height
            new_height = ref_height
            new_width = int(image.shape[1] / scale)
    
    if debug:
        print(f"Original shape: {image.shape}")
        print(f"Reference shape: {reference_image.shape}")
        print(f"New dimensions (w,h): {new_width}, {new_height}")
        print(f"Aspect ratios - Image: {img_aspect:.2f}, Reference: {ref_aspect:.2f}")
    
    # Resize image
    resized = cv.resize(image, (new_width, new_height))
    
    # Create output canvas with appropriate channels
    if len(image.shape) == 3:
        final_image = np.zeros((ref_height, ref_width, image.shape[2]), dtype=np.uint8)
    else:
        final_image = np.zeros((ref_height, ref_width), dtype=np.uint8)
    
    # Calculate position to place the resized image
    x_offset = (ref_width - new_width) // 2
    y_offset = (ref_height - new_height) // 2
    
    # Handle cases where the resized image is wider than the canvas
    if new_width > ref_width:
        crop_x = (new_width - ref_width) // 2
        if len(image.shape) == 3:
            final_image = resized[y_offset:y_offset+new_height, crop_x:crop_x+ref_width, :]
        else:
            final_image = resized[y_offset:y_offset+new_height, crop_x:crop_x+ref_width]
    else:
        # Place the resized image in the center
        if len(image.shape) == 3:
            final_image[y_offset:y_offset+new_height, x_offset:x_offset+new_width, :] = resized
        else:
            final_image[y_offset:y_offset+new_height, x_offset:x_offset+new_width] = resized
    
    return final_image

























