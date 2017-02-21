
import numpy as np
import matplotlib.image as mpimg
from PIL import Image

# Function to load the images
def load_image(infilename):
    data = mpimg.imread(infilename)
    return data


# Function to crop images into patches
def img_crop_padded(im, w, h, pad):
    list_patches = []
    
    imgwidth = im.shape[0] - pad
    imgheight = im.shape[1] - pad
    
    is_2d = len(im.shape) < 3
    for i in range(pad,imgheight,h):
        for j in range(pad,imgwidth,w):
            if is_2d:
                im_patch = im[j-pad:j+w+pad, i-pad:i+h+pad]
            else:
                im_patch = im[j-pad:j+w+pad, i-pad:i+h+pad, :]
            list_patches.append(im_patch)
    return list_patches


# Function to deflatten the output
# Reshape it an array of size (patch_size x patch_size)
def deflatten(img_flat, patch_size):
    img = []
    for i in range(len(img_flat)):
        img.append(img_flat[i].reshape((patch_size,patch_size)))
    return img


# Function to ransform patches into a complete black and white image
def label_to_img(imgwidth, imgheight, w, h, labels):
    im = np.zeros([imgwidth, imgheight])
    idx = 0
    for i in range(0,imgheight,h):
        for j in range(0,imgwidth,w):
            im[j:j+w, i:i+h] = labels[idx]
            idx = idx + 1
    return im



# Given function that allow the superposition of mask over image
def make_img_overlay(img, predicted_img):
    w = img.shape[0]
    h = img.shape[1]
    color_mask = np.zeros((w, h, 3), dtype=np.uint8)
    color_mask[:,:,0] = predicted_img*255

    img8 = img_float_to_uint8(img)
    background = Image.fromarray(img8, 'RGB').convert("RGBA")
    overlay = Image.fromarray(color_mask, 'RGB').convert("RGBA")
    new_img = Image.blend(background, overlay, 0.2)
    return new_img



# Helper functions


# Concatenate an image and its groundtruth
def concatenate_images(img, gt_img):
    nChannels = len(gt_img.shape)
    w = gt_img.shape[0]
    h = gt_img.shape[1]
    if nChannels == 3:
        cimg = np.concatenate((img, gt_img), axis=1)
    else:
        gt_img_3c = np.zeros((w, h, 3), dtype=np.uint8)
        gt_img8 = img_float_to_uint8(gt_img)          
        gt_img_3c[:,:,0] = gt_img8
        gt_img_3c[:,:,1] = gt_img8
        gt_img_3c[:,:,2] = gt_img8
        img8 = img_float_to_uint8(img)
        cimg = np.concatenate((img8, gt_img_3c), axis=1)
    return cimg

# Basic given image cropping function
def img_crop(im, w, h):
    list_patches = []
    imgwidth = im.shape[0]
    imgheight = im.shape[1]
    is_2d = len(im.shape) < 3
    for i in range(0,imgheight,h):
        for j in range(0,imgwidth,w):
            if is_2d:
                im_patch = im[j:j+w, i:i+h]
            else:
                im_patch = im[j:j+w, i:i+h, :]
            list_patches.append(im_patch)
    return list_patches


# Function to crop images into patches for the output. Simply flattened the output 
def img_crop_output(im, w, h):
    list_patches = []
    imgwidth = im.shape[0]
    imgheight = im.shape[1]
    is_2d = len(im.shape) < 3
    for i in range(0,imgheight,h):
        for j in range(0,imgwidth,w):
            if is_2d:
                im_patch = im[j:j+w, i:i+h].flatten() # Here we add the flatten function
            else:
                im_patch = im[j:j+w, i:i+h, :].flatten() # Here we add the flatten function
            list_patches.append(im_patch)
    return list_patches


def img_float_to_uint8(img):
    rimg = img - np.min(img)
    rimg = (rimg / np.max(rimg) * 255).round().astype(np.uint8)
    return rimg
