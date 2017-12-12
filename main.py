# coding: utf-8
import numpy as np
import scipy.misc
import matplotlib.pyplot as plt
from scipy import fftpack

img_path = 'factory.bmp'
chrominance_table = np.array([
    [17, 18, 24, 47, 99, 99, 99, 99],
    [18, 21, 26, 66, 99, 99, 99, 99],
    [24, 26, 56, 99, 99, 99, 99, 99],
    [47, 66, 99, 99, 99, 99, 99, 99],
    [99, 99, 99, 99, 99, 99, 99, 99],
    [99, 99, 99, 99, 99, 99, 99, 99],
    [99, 99, 99, 99, 99, 99, 99, 99],
    [99, 99, 99, 99, 99, 99, 99, 99],
    ]) 
luminance_table = np.array([
    [16, 11, 10, 16, 24, 40, 51, 61],
    [12, 12, 14, 19, 26, 58, 60, 55],
    [14, 13, 16, 24, 40, 57, 69, 56],
    [14, 17, 22, 29, 51, 87, 80, 62],
    [18, 22, 37, 56, 68, 109, 103, 77],
    [24, 36, 55, 64, 81, 104, 113, 92],
    [49, 64, 78, 87, 103, 121, 120, 101],
    [72, 92, 95, 98, 112, 100, 103, 99]
    ])


### Color Space Transform
def rgb2yuv(image:'[0,255]'):
    image = image.astype(np.float32)
    r = image[:,:,0]
    g = image[:,:,1]
    b = image[:,:,2]
    
    y = 0.299 * r + 0.587 * g + 0.114 * b
    u = 128 -0.16875 * r  -0.33126 * g + 0.5 * b
    v = 128 + 0.5 * r - 0.41869 * g - 0.08131 * b
    
    img_yuv = np.stack((y,u,v), axis =2)
    
    return img_yuv

def yuv2rgb(image):
    image = image.astype(np.float32)
    y = image[:,:,0]
    u = image[:,:,1]
    v = image[:,:,2]
    r = y + 1.402 * (v - 128)
    g = y - 0.34413 * (u-128) - 0.71414 * (v-128)
    b = y + 1.772 * (u-128)
    
    img_rgb = np.stack((r,g,b), axis = 2)
    return img_rgb

### Color Sampling and color separate
def yuv_sepa(yuv_image): # 4:2:2
    height = yuv_image.shape[0]
    width = yuv_image.shape[1]
    lumi = yuv_image[:,:,0]
    sampling_column = [i for i in range(0, width, 2)]
    replace_column = [i + 1 for i in range(0, width, 2)]
    u_channel = yuv_image[:,:,1]
    u_channel[:, replace_column] = u_channel[:, sampling_column] # Sampling
    v_channel = yuv_image[:,:,2]
    v_channel[:, replace_column] = v_channel[:, replace_column]
    return lumi, u_channel, v_channel

def dct_trans(subimage):
    tmp = fftpack.dct(subimage, type = 2, axis = 0, norm = 'ortho') # Horizontal
    subimg_dct = fftpack.dct(tmp, type = 2, axis = 1, norm = 'ortho') # Vertical
    return subimg_dct

def idct_trans(subimg_dct):
    tmp = fftpack.idct(subimg_dct, type = 2, axis = 1, norm = 'ortho')
    sub_img = fftpack.idct(tmp, type = 2, axis = 0, norm = 'ortho')
    return sub_img

def img_dct_quan(img, channel):
    height = img.shape[0]
    width = img.shape[1]
    new_img = np.zeros([height, width])
    if channel == 'lumi':
        for i in range(0, height, 8):
            for j in range(0, width, 8):
                new_img[i:i+8, j:j+8] = np.round(dct_trans(img[i:i+8, j:j+8]) / luminance_table)
    else:
        for i in range(0, height, 8):
            for j in range(0, width, 8):
                new_img[i:i+8, j:j+8] = np.round(dct_trans(img[i:i+8, j:j+8]) / chrominance_table)
    
    return new_img

def img_idct_dequan(img_dct, channel, ):
    height = img_dct.shape[0]
    width = img_dct.shape[1]
    new_img = np.zeros([height, width])
    if channel == 'lumi':
        for i in range(0, height, 8):
            for j in range(0, width, 8):
                new_img[i:i+8, j:j+8] = idct_trans(img_dct[i:i+8, j:j+8] * luminance_table)
                
    else:
        for i in range(0, height, 8):
            for j in range(0, width, 8):
                new_img[i:i+8, j:j+8] = idct_trans(img_dct[i:i+8, j:j+8] * chrominance_table)        
    
    return new_img

def mse_error(img1, img2):
    '''
    img1 and img2 should have same size
    
    '''
    height = img1.shape[0]
    width = img1.shape[1]
    img1 = img1.astype(np.float32)
    img2 = img2.astype(np.float32)
    error = np.sum((img1 - img2) ** 2) / (height * width)
    return error

def get_img(img_path):
    img = scipy.misc.imread(img_path, mode = 'RGB')
    return img

def main():
    img = get_img(img_path)
    plt.figure(1)
    plt.imshow(img)
    img_yuv = rgb2yuv(img)
    lumi, u_channel, v_channel = yuv_sepa(img_yuv)
    # Dct and quantization
    lumi_dct_quan = img_dct_quan(lumi, 'lumi')
    u_dct_quan = img_dct_quan(u_channel,'u')
    v_dct_Quan = img_dct_quan(v_channel,'v')

    # Dequantization and inverse DCT
    lumi_idct = img_idct_dequan(lumi_dct_quan,'lumi')
    u_idct = img_idct_dequan(u_dct_quan,'u')
    v_idct = img_idct_dequan(v_dct_Quan,'v')
    jpeg = np.stack((lumi_idct,u_idct, v_idct), axis = 2)

    jpeg = yuv2rgb(jpeg)
    jpeg = np.clip(jpeg, 0, 255)
    plt.figure(2)
    plt.imshow(jpeg.astype(np.uint8))
    plt.show()
    print('Loss is: ' + str(mse_error(img, jpeg)))

if __name__ == '__main__':
    main()
