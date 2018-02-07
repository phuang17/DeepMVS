import os
import re
import json

import numpy as np
from numpy import linalg as la
import cv2
import imageio

def max_disparity_adjust(d, max_disparity_mu = 1.2, max_disparity_sigma = 0.2):
	return d * np.random.normal(max_disparity_mu, max_disparity_sigma)

def generate_volume_train(shared_data, max_num_neighbors, num_depths, patch_height, patch_width, dataset_path = None):
	ready_e = shared_data["ready_e"]
	start_e = shared_data["start_e"]
	DATASET_LIST = ([
		"GTAV_540", "GTAV_720",
		"mvs_achteck_turm", "mvs_breisach", "mvs_citywall", 
		"rgbd_10_to_20_3d_train", "rgbd_10_to_20_handheld_train", "rgbd_10_to_20_simple_train", "rgbd_20_to_inf_3d_train", "rgbd_20_to_inf_handheld_train", "rgbd_20_to_inf_simple_train",
		"scenes11_train", 
		"sun3d_train_0.01m_to_0.1m", "sun3d_train_0.1m_to_0.2m", "sun3d_train_0.2m_to_0.4m", "sun3d_train_0.4m_to_0.8m", "sun3d_train_0.8m_to_1.6m", "sun3d_train_1.6m_to_infm"
	])
	if dataset_path is None:
		DATASET_DIR_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "dataset", "train")
	else:
		DATASET_DIR_ROOT = dataset_path
	# Assign probability weights to each sequence according to the total number of images.
	d_idx_choices = [[] for i in range(0, max_num_neighbors)]
	s_idx_choices = [[] for i in range(0, max_num_neighbors)]
	weights = [[] for i in range(0, max_num_neighbors)]
	num_images = []
	for (d_idx, dataset) in enumerate(DATASET_LIST):
		with open(os.path.join(DATASET_DIR_ROOT, dataset, "num_images.json")) as f:
			num_images.append(json.load(f))
		for neighbor_idx in range(0, max_num_neighbors):
			# Select only the sequences with at least (neighbor_idx + 1) frames.
			choices = np.flatnonzero(np.greater(num_images[-1], neighbor_idx + 1))
			s_idx_choices[neighbor_idx].extend(choices)
			d_idx_choices[neighbor_idx].extend([d_idx for i in range(0, len(choices))])
			weights[neighbor_idx].extend(np.array(num_images[-1])[choices])
	# Normalize probability distribution.
	for neighbor_idx in range(0, max_num_neighbors):
		weights[neighbor_idx] = np.array(weights[neighbor_idx]).astype(np.float32) / np.sum(np.array(weights[neighbor_idx]))
	# Allocate buffers.
	ref_img = np.zeros((patch_height, patch_width, 3), dtype = np.float32)
	sweep_volume = np.zeros((max_num_neighbors, num_depths, patch_height, patch_width, 3), dtype = np.float32)
	ref_depth = np.zeros((patch_height, patch_width), dtype = np.float32)
	valid_mask = np.zeros((patch_height, patch_width), dtype = np.uint8)
	x_map = np.zeros((patch_height, patch_width), dtype = np.float32)
	y_map = np.zeros((patch_height, patch_width), dtype = np.float32)
	coord_buffer = np.zeros((patch_height, patch_width, 4), dtype = np.float32)
	# Fetch image dimensions of each dataset.
	actual_width_list = []
	actual_height_list = []
	for dataset in DATASET_LIST:
		img = imageio.imread(os.path.join(DATASET_DIR_ROOT, dataset, "{:04d}".format(0), "images", "{:04d}.png".format(0)))
		actual_width_list.append(img.shape[1])
		actual_height_list.append(img.shape[0])
	# Keep generating until stop signal.
	while not shared_data["stop"]:
		# Wait for start_flag.
		start_e.wait()
		start_e.clear()
		if shared_data["stop"]:
			break
		# Randomly choose number of neighbors from [1, max_num_neighbors].
		num_neighbors = np.random.randint(0, max_num_neighbors)
		# Randomly select a sequence until finding an available one.
		while True:
			# Select a dataset.
			random_idx = np.random.choice(len(d_idx_choices[num_neighbors]), p = weights[num_neighbors])
			d_idx = d_idx_choices[num_neighbors][random_idx]
			DATASET_DIR = os.path.join(DATASET_DIR_ROOT, DATASET_LIST[d_idx])
			actual_width = actual_width_list[d_idx]
			actual_height = actual_height_list[d_idx]
			# Select a sequence, a reference image, and location of the patch.
			s_idx = s_idx_choices[num_neighbors][random_idx]
			r_idx = np.random.randint(0, num_images[d_idx][s_idx])
			target_x = np.random.randint(0, actual_width - patch_width)
			target_y = np.random.randint(0, actual_height - patch_height)
			# Load ground truth depths.
			ref_depth_full = imageio.imread(os.path.join(DATASET_DIR, "{:04d}".format(s_idx), "depths", "{:04d}.exr".format(r_idx)))
			valid_mask_full = ref_depth_full != 0.0
			ref_depth_full = np.where(valid_mask_full, 1.0 / ref_depth_full, 0.0)
			max_disparity = max_disparity_adjust(np.max(ref_depth_full))
			ref_depth[...] = ref_depth_full[target_y:target_y + patch_height, target_x:target_x + patch_width]
			valid_mask[...] = valid_mask_full[target_y:target_y + patch_height, target_x:target_x + patch_width]
			# If too many invalid ground truths, skip it.
			if np.count_nonzero(valid_mask) < patch_width * patch_height * 0.80:
				continue
			# Choose a random pixel to determine overlapping neighbor images. 
			while True:
				sample_x = np.random.randint(0, patch_width)
				sample_y = np.random.randint(0, patch_height)
				if valid_mask[sample_y, sample_x]:
					break
			# Load camera pose of reference image.
			with open(os.path.join(DATASET_DIR, "{:04d}".format(s_idx), "poses", "{:04d}.json".format(r_idx))) as f:
				r_info = json.load(f)
				r_c_x = r_info["c_x"]
				r_c_y = r_info["c_y"]
				r_f_x = r_info["f_x"]
				r_f_y = r_info["f_y"]
				r_extrinsic = np.array(r_info["extrinsic"])
			# Select neighbors which have overlap with reference image.
			selected_neighbor_count = 0
			valid_neighbors = np.flatnonzero(np.not_equal(np.array(range(0, num_images[d_idx][s_idx])), r_idx))
			n_idx_list = []
			while len(valid_neighbors) > 0:
				n_idx = valid_neighbors[np.random.randint(0, len(valid_neighbors))]
				valid_neighbors = valid_neighbors[valid_neighbors != n_idx]
				with open(os.path.join(DATASET_DIR, "{:04d}".format(s_idx), "poses", "{:04d}.json".format(n_idx))) as f:
					n_info = json.load(f)
					n_c_x = n_info["c_x"]
					n_c_y = n_info["c_y"]
					n_f_x = n_info["f_x"]
					n_f_y = n_info["f_y"]
					n_extrinsic = np.array(n_info["extrinsic"])
				x = (target_x + sample_x - r_c_x) / r_f_x
				y = (target_y + sample_y - r_c_y) / r_f_y
				d = ref_depth[sample_y, sample_x]
				if d == 0.0:
					coord = np.array([x, y, 1.0, 0.0])
				else:
					coord = np.array([x/d, y/d, 1.0/d, 1.0])
				coord = n_extrinsic.dot(la.inv(r_extrinsic)).dot(coord)
				x = coord[0] / coord[2] * n_f_x + n_c_x
				y = coord[1] / coord[2] * n_f_y + n_c_y
				if x >= 0 and x < actual_width and y >= 0 and y < actual_height and coord[2] > 0:
					n_idx_list.append(n_idx)
				if len(n_idx_list) == num_neighbors + 1:
					break
			# Check if enough number of neighbors are collected.
			if len(n_idx_list) < num_neighbors + 1:
				continue
			else:
				break
		# Load reference RGB image.
		d_step = max_disparity / (num_depths - 1)
		d_list = [d_step * i for i in range(0, num_depths)]
		ref_img_full = imageio.imread(os.path.join(DATASET_DIR, "{:04d}".format(s_idx), "images", "{:04d}.png".format(r_idx))).astype(np.float32) / 255.0
		ref_img[...] = ref_img_full[target_y:target_y + patch_height, target_x:target_x + patch_width, ...] - 0.5
		# Generate plane-sweep volume.
		for neighbor_idx in range(0, num_neighbors + 1):
			# Load camera pose of neighbor image.
			n_idx = n_idx_list[neighbor_idx]
			with open(os.path.join(DATASET_DIR, "{:04d}".format(s_idx), "poses", "{:04d}.json".format(n_idx))) as f:
				n_info = json.load(f)
				n_c_x = n_info["c_x"]
				n_c_y = n_info["c_y"]
				n_f_x = n_info["f_x"]
				n_f_y = n_info["f_y"]
				n_extrinsic = np.array(n_info["extrinsic"])
			neighbor_img = imageio.imread(os.path.join(DATASET_DIR, "{:04d}".format(s_idx), "images", "{:04d}.png".format(n_idx))).astype(np.float32) / 255.0 - 0.5
			trans_matrix = n_extrinsic.dot(la.inv(r_extrinsic))
			for (disparity_idx, disparity) in enumerate(d_list):
				coord_buffer[..., 0] = (np.mgrid[0:patch_height, 0:patch_width][1] + target_x - r_c_x) / r_f_x
				coord_buffer[..., 1] = (np.mgrid[0:patch_height, 0:patch_width][0] + target_y - r_c_y) / r_f_y
				coord_buffer[..., 2] = 1.0
				if disparity == 0.0:
					coord_buffer[..., 3] = 0.0
				else:
					coord_buffer[..., 0:3] /= disparity
					coord_buffer[..., 3] = 1.0
				coord_buffer[...] = np.moveaxis(trans_matrix.dot(coord_buffer[..., np.newaxis])[..., 0], 0, -1)
				x_map[...] = np.where(coord_buffer[..., 2] >= 0.0, coord_buffer[..., 0] / coord_buffer[..., 2] * n_f_x + n_c_x, -1.0)
				y_map[...] = np.where(coord_buffer[..., 2] >= 0.0, coord_buffer[..., 1] / coord_buffer[..., 2] * n_f_y + n_c_y, -1.0)
				cv2.remap(neighbor_img, x_map, y_map, cv2.INTER_LINEAR, sweep_volume[neighbor_idx, disparity_idx, ...], cv2.BORDER_CONSTANT, (0.0, 0.0, 0.0))
		# Send plane-sweep volume to main thread.
		shared_data["ref_img"] = ref_img
		shared_data["ref_img_full"] = ref_img_full
		shared_data["target_x"] = target_x
		shared_data["target_y"] = target_y
		shared_data["sweep_volume"] = sweep_volume[0:num_neighbors + 1, ...]
		shared_data["ref_depth"] = ref_depth / max_disparity
		shared_data["valid_mask"] = valid_mask
		shared_data["num_neighbors"] = num_neighbors + 1
		ready_e.set()



