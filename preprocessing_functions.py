'''
Date: 02/23/2018
Objective:
  process the ucsd dataset to get
    - image: with mask; reshape
    - density map: plain image with mixed 2D Gaussian density; with mask; reshape
  data augment options include:
    - horizontal flip
    - random crop
    - adjust of brightness
    - adjust of shaprness
    - add rondom noise ??? (only image)
Notice:
    - all images are from the same camera: only one mask
'''

from __future__ import print_function
from scipy.stats import multivariate_normal
from PIL import Image, ImageChops, ImageEnhance
from scipy.io import loadmat
import glob
import numpy as np
import cv2


# get mask from mat file (for ground truth and density)
def get_mask_map(filepath):
    mask = loadmat(filepath).get('roi')[0][0][2]
    return mask


# get perspective from mat file (for density)
def get_perspective_map(filepath):
    p_map = loadmat(filepath).get('dmap')[0][0][0]
    return p_map


# find the global mean of all images
def get_global_mean(img_path, img_type, file_num):
    # initialize mean array
    mean_arr = np.zeros(file_num)
    folder_list = glob.glob(img_path)
    # get folder number, number of image in each folder
    folder_num = len(folder_list)
    folder_img_num = int(file_num/folder_num)
    for i, folder in enumerate(folder_list):
        image_list = glob.glob(folder + '/*.' + img_type)
        for j, img_path in enumerate(image_list):
            img = Image.open(img_path)
            img_arr = np.array(img, dtype=np.float32)
            idx = folder_img_num * i + j
            mean_arr[idx] = np.mean(img_arr)
    return np.sum(mean_arr)/file_num


'''
get the ground truth image
Procedure: subtract mean, apply mask, then resize to (224,224)
Input:
    - img_path: './image/*'
    - image_type: 'png'
    - global_mean: 96.026
    - mask-arr: return from get_mask_map()
    - resize_dim = (224,224)
    - file_num: 2000
    - number_of_channel: 1
'''


def get_ground_truth_img(img_path, img_type, global_mean, mask_arr, resize_dim, file_num, num_of_channel):
    # initial data array
    data_arr = np.zeros((file_num, resize_dim[0], resize_dim[1], num_of_channel))
    # get sub folders under `image'
    folder_list = glob.glob(img_path)
    # get folder number, number of image in each folder
    folder_num = len(folder_list)
    folder_img_num = int(file_num/folder_num)
    for i, folder in enumerate(folder_list):
        image_list = glob.glob(folder + '/*.' + img_type)
        for j, img_path in enumerate(image_list):
            img = Image.open(img_path)
            img_arr = np.array(img, dtype=np.float32)
            img_arr -= global_mean
            # expand dimension if number_of_channel <=1
            if num_of_channel <= 1:
                img_arr = np.expand_dims(img_arr, axis=-1)
            # apply mask
            for channel in range(num_of_channel):
                img_arr[:,:,channel] = np.multiply(img_arr[:,:,channel], mask_arr)
            # resize
            img_arr = cv2.resize(img_arr, resize_dim)
            # RGB to BGR (if three channel)
            if num_of_channel == 3:
                img_arr = img_arr[..., ::-1]
            # expand dimension if number_of_channel <=1
            if num_of_channel <= 1:
                img_arr = np.expand_dims(img_arr, axis=-1)
            # add to img data array
            idx = folder_img_num*i + j
            data_arr[idx,:,:,:] = img_arr
            print(idx)
    return data_arr


'''
get ground truth counts (sum up counts in left, right directions)
Input: 
    count_path: './count/*'
    file_num: 2000
Output: 
    loc_list
    count_arr
'''
def get_ground_truth_count(count_path, img_num):
    # initial count array
    count_arr = np.zeros(img_num)
    # get mat under `count'
    file_list = glob.glob(count_path+'.mat')
    # number of image in each mat file
    file_img_num = int(img_num / len(file_list))
    for i, mat_file in enumerate(file_list):
        count_total = loadmat(mat_file).get('count')
        counts = count_total[0][0][0] + count_total[0][1][0]
        for j, count in enumerate(counts):
            idx = file_img_num*i + j
            count_arr[idx] = counts[j]
    return count_arr.astype(dtype=np.float32)


'''
get ground truth location of people in each image
Input: 
    - loc_path: './location/*'
    - img_num: 2000
'''
def get_ground_truth_loc(loc_path, img_num):
    # initialize location list
    loc_list = []
    file_list = glob.glob(loc_path+'.mat')
    file_num = len(file_list)
    sub_file_num = int(img_num/file_num)
    for i, mat_file in enumerate(file_list):
        loc_data = loadmat(mat_file).get('frame')
        for j in range(sub_file_num):
            loc_arr = loc_data[0][j][0][0][0]
            loc_list.append(loc_arr[:,0:2])
    return loc_list



'''
generate 2D density (Gaussian) given location (mean), perspective ratio (variance)
Input: 
    - mask_arr: mask
    - loc: location (1*2)
    - p_arr: perspective map
    - const: constant as in paper, choose 0.3 as default   
'''
def get_single_density(mask_arr, loc, p_arr, const=0.3):
    dim = mask_arr.shape
    x, y = np.mgrid[0:dim[0]:1, 0:dim[1]:1]
    pos = np.empty(x.shape + (2,))
    pos[:, :, 0] = x
    pos[:, :, 1] = y
    # notice: location is indexed as (column, row)
    idx_row = min(max(0, int(np.round(loc[1]) - 1)), dim[0]-1)
    idx_col = min(max(0, int(np.round(loc[0]) - 1)), dim[1]-1)
    sigma = p_arr[idx_row, idx_col] * const
    rv = multivariate_normal([loc[1], loc[0]], [[sigma, 0.0], [0.0, sigma]])
    return rv.pdf(pos)


'''
get the ground truth density
Procedure: first generate the mixed 2D Gaussian density, apply mask, and resize
also return the ground truth of people counting (after mask is applied)
Input: 
    - loc_list: return from get_ground_truth_loc()
    - mask: return from get_mask_map()
    - p_arr: return from get_perspective_map()
    - resize_dim: (224, 224)
    - img_num: 2000
    - resize_factor: 158*238 / (224*224) --> 0.7446
Output:
    - density_arr:
    - count_arr: 
'''



def get_ground_truth_density(loc_list, mask, p_arr, resize_dim, img_num, resize_factor=0.7446):
    # initialize data
    density_arr = np.zeros((img_num, resize_dim[0], resize_dim[1]))
    count_arr = np.zeros(img_num)
    for i in range(img_num):
        print(i)
        temp_density = np.zeros((mask.shape[0], mask.shape[1]))
        temp_loc_arr = loc_list[i]
        for j in range(temp_loc_arr.shape[0]):
            temp_density += get_single_density(mask, temp_loc_arr[j], p_arr, const=0.3)
        # apply mask
        temp_density = np.multiply(temp_density, mask)
        # resize
        density_arr[i] = cv2.resize(temp_density, resize_dim) * resize_factor
        # update count
        count_arr[i] = np.sum(density_arr[i])
    return density_arr, count_arr
'''
options
'''
def data_agument():
    pass



