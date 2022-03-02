import matplotlib.pyplot as plt
import math
import numpy as np
from skimage.color import rgb2gray
from skimage import feature

def print_img(title, image):
    plt.title(title)
    if len(image.shape) >= 3 and min(image.shape) > 1:
        plt.imshow(image)
    else:
        plt.imshow(image, cmap=plt.cm.gray)
    plt.show()


def calc_euclidean(image1, image2):
    max_row_img2 = image2.shape[0]
    max_col_img2 = image2.shape[1]

    euclid = 0.0

    for row in range(0, image1.shape[0]):
        for col in range(0, image1.shape[1]):
            if (row < max_row_img2 and col < max_col_img2):
                p = (image1[row][col] - image2[row][col]) ** 2
                euclid = euclid + p/100

    return 10 * math.sqrt(euclid)

def compute_brightness(image):
    img = np.float64(image)

    Cr = img[:, :, 2]
    Cg = img[:, :, 1]
    Cb = img[:, :, 0]

    Width = img.shape[0]
    Height = img.shape[1]

    return np.sqrt((0.241 * (Cr**2)) + (0.691 * (Cg**2)) + (0.068 * (Cb**2))) / (Width * Height)


def contrast_equalization(image, threshold=125, adjust_if_lower=True):
    contrast = 1
    img = image
    brightness = compute_brightness(img).sum()

    while (adjust_if_lower and brightness < threshold):
        contrast += 0.01
        img = image.copy() * contrast
        brightness = compute_brightness(img).sum()

    while (brightness > threshold):
        contrast -= 0.01
        img = image.copy() * contrast
        brightness = compute_brightness(img).sum()

    return np.uint8(img)


def show_histogram(image):
    fig, ax = plt.subplots(2, 3)
    bins = 256
    for ci, c in enumerate('rgb'):
        ax[0, ci].imshow(image[:, :, ci], cmap='gray')
        ax[1, ci].hist(image[:, :, ci].flatten(),  bins=256, density=True)

def specularity_pixels_count(image):
    img = np.float64(image)
    r = img[:, :, 2]
    g = img[:, :, 1]
    b = img[:, :, 0]

    m = (1/3)*(r+g+b)
    m_max = m.max()  

    if ((b+r) - 2*g).sum() >= 0:
        s = r-(0.5)*g-b
    else:
        s = r+g-2*b
    
    s_max = s.max()

    p = img > 0.5*m_max
    q = img < (1/3)*s_max

    return (p & q).sum()

def is_strong_solid_opacity(image, padding=70, threshold=100, sigma=0.01) -> bool:
    """A strong manifestatiom of opacity will probably be manifested by a low
       border detection on the image, because the eye is so opaque that the light
       of the fundoscope cannot show any veins

    Parameters
    ----------
    image : 2D array
        Grayscale input image to detect edges on; can be of any dtype.

    padding: how much of the image should be cut from its edges, in order to avoid
             counting the fundos circle format as a border. This is better than having
             to detect a ciclic border and exclude it. You could also set padding to zero 
             and try to compensate it increasing the threshold
    
    threshold: how many "border pixels" should be considered as a minimal for having 
               veins detected.

    Returns
    -------
    output : True/False"""


    if len(image.shape) > 2:
        img = rgb2gray(image)
    else:
        img = image

    img = img[padding:img.shape[0]-padding,padding:img.shape[1]-padding]
    img =  feature.canny(img, sigma=sigma)
    img[img > 0] = 255
    return (img > 0).sum() < threshold

def find_hog(image):
    return feature.hog(image, orientations=8, pixels_per_cell=(16, 16),
        cells_per_block=(1, 1), visualize=False, feature_vector=True, channel_axis=-1)
    