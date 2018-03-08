'''
preprocessing to get:
    - raw image: meet VGG 16 requirements
    - density map
'''
from preprocessing_functions import *


# variables
img_type = 'png'
img_num = 2000
resize_dim = (224,224)
num_of_channel = 1

# path
img_path = './image/*'
loc_path = './location/*'
mask_path = './mask/vidf1_33_roi_mainwalkway.mat'
perspective_path = './perspective/vidf1_33_dmap3.mat'


# get mask, perspective map, global_mean
mask = get_mask_map(mask_path)
p_arr = get_perspective_map(perspective_path)
global_mean = get_global_mean(img_path, img_type, img_num)

print('----------------- image processing start --------------------')
# get ground truth image
image_arr = get_ground_truth_img(img_path, img_type, global_mean, mask, resize_dim, img_num, num_of_channel)


# get ground truth location
loc_list = get_ground_truth_loc(loc_path, img_num)
# get ground truth density and count
print('----------------- density processing start --------------------')
gt_density, gt_count = get_ground_truth_density(loc_list, mask, p_arr, resize_dim, img_num, resize_factor=0.7446)


# save
np.save('./data/X.npy', image_arr)
np.save('./data/Y_gt_density.npy', gt_density)
np.save('./data/Y_gt_count.npy', gt_count)
