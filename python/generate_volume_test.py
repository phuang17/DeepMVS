import os
import re
import json

import numpy as np
from numpy import linalg as la
import cv2
import imageio

def generate_volume_test(shared_data):
	# Get event handles.
	ready_e = shared_data["ready_e"]
	start_e = shared_data["start_e"]
	# Read paramters.
	patch_width = shared_data["patch_width"]
	patch_height = shared_data["patch_height"]
	num_depths = shared_data["num_depths"]
	max_num_neighbors = shared_data["max_num_neighbors"]
	sparse_model = shared_data["sparse_model"]
	# Continue generate pland sweep volume.
	while True:
		start_e.wait()
		start_e.clear()
		if shared_data["stop"]:
			break
		# Get reference image index and location of the patch.
		ref_image_idx = shared_data["image_idx"]
		target_x = shared_data["target_x"]
		target_y = shared_data["target_y"]
		# Read the reference RGB image. 
		ref_image = sparse_model.image_list.images[ref_image_idx]
		ref_img_full = ref_image.rgb
		ref_img = ref_img_full[target_y:target_y + patch_height, target_x:target_x + patch_width, :] - 0.5
		# Get neighbor images and disparity step.
		ref_camera = sparse_model.camera_list.get_by_id(ref_image.camera_id)
		neighbor_list = ref_image.neighbor_list
		num_neighbors = len(neighbor_list)
		d_step = ref_image.estimated_max_disparity / num_depths
		# Allocate buffers.
		sweep_volume = np.zeros((num_neighbors, num_depths, patch_height, patch_width, 3), dtype = np.float32)
		x_map = np.zeros((patch_height, patch_width), dtype = np.float32)
		y_map = np.zeros((patch_height, patch_width), dtype = np.float32)
		coord_buffer = np.zeros((patch_height, patch_width, 4), dtype = np.float32)
		# Loop through neighbor images.
		n_idx = 0
		for n_image_idx in neighbor_list:
			# Load the neighbor image.
			n_image = sparse_model.image_list.images[n_image_idx]
			n_camera = sparse_model.camera_list.get_by_id(n_image.camera_id)
			n_img = n_image.rgb - 0.5
			trans_matrix = n_image.extrinsic.dot(la.inv(ref_image.extrinsic))
			# Warp the neighbor image for each disparity level.
			for d_idx in range(0, num_depths):
				disparity = d_idx * d_step
				coord_buffer[..., 0] = (np.mgrid[0:patch_height, 0:patch_width][1] + target_x - ref_camera.cx) / ref_camera.fx
				coord_buffer[..., 1] = (np.mgrid[0:patch_height, 0:patch_width][0] + target_y - ref_camera.cy) / ref_camera.fy
				coord_buffer[..., 2] = 1.0
				if disparity == 0.0:
					coord_buffer[..., 3] = 0.0
				else:
					coord_buffer[..., 0:3] /= disparity
					coord_buffer[..., 3] = 1.0
				coord_buffer[...] = np.moveaxis(trans_matrix.dot(coord_buffer[..., np.newaxis])[..., 0], 0, -1)
				x_map[...] = np.where(coord_buffer[..., 2] > 0.0, coord_buffer[..., 0] / coord_buffer[..., 2] * n_camera.fx + n_camera.cx, -1.0) 
				y_map[...] = np.where(coord_buffer[..., 2] > 0.0, coord_buffer[..., 1] / coord_buffer[..., 2] * n_camera.fy + n_camera.cy, -1.0) 
				cv2.remap(n_img, x_map, y_map, cv2.INTER_LINEAR, sweep_volume[n_idx, d_idx, ...], cv2.BORDER_CONSTANT, (0.0, 0.0, 0.0))
			# Make sure there is overlap between reference image patch and this neighbor image.
			if np.any(sweep_volume[n_idx, ...] != 0.0):
				n_idx += 1
		# Update num_neighbors based on actual overlapping neighbor images.
		num_neighbors = n_idx
		# If there is no overlap at all, pass zeros as the plane-sweep volume and hope for the best.
		if num_neighbors == 0:
			num_neighbors = 1
		# Send the data to main thread.
		shared_data["ref_img"] = ref_img
		shared_data["ref_img_full"] = ref_img_full
		shared_data["sweep_volume"] = sweep_volume[0:num_neighbors, ...]
		shared_data["num_neighbors"] = num_neighbors
		ready_e.set()

