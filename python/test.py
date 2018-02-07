import os
import sys
import argparse
import threading

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torchvision as vision
import pydensecrf.densecrf as dcrf
import imageio

from model import DeepMVS
from generate_volume_test import generate_volume_test
from colmap_helpers import ColmapSparse

# Parse arguments
parser = argparse.ArgumentParser(description = "Run DeepMVS on a sequence.")
parser.add_argument("--image_width", dest = "image_width", type = int, default = -1, help = "Image width (<0 means to derive from image_height).")
parser.add_argument("--image_height", dest = "image_height", type = int, default = 540, help = "Image height (<0 means to derive from image_width).")
parser.add_argument("--patch_width", dest = "patch_width", type = int, default = 128, help = "Width of patches.")
parser.add_argument("--patch_height", dest = "patch_height", type = int, default = 128, help = "Height of patches.")
parser.add_argument("--stride_width", dest = "stride_width", type = int, default = 64, help = "Width of the stride.")
parser.add_argument("--stride_height", dest = "stride_height", type = int, default = 64, help = "Height of the stride.")
parser.add_argument("--num_depths", dest = "num_depths", type = int, default = 100, help = "Number of disparity levels.")
parser.add_argument("--max_num_neighbors", dest = "max_num_neighbors", type = int, default = 16, help = "Maximum number of neighbor images.")
parser.add_argument("--no_gpu", dest = "use_gpu", action = "store_false", default = True, help = "Disable use of GPU.")
parser.add_argument("--image_path", dest = "image_path", help = "Path to the images.", required = True)
parser.add_argument("--sparse_path", dest = "sparse_path", help = "Path to the sparse reconstruction.", required = True)
parser.add_argument("--output_path", dest = "output_path", help = "Path to store the results.", required = True)
parser.add_argument("--model_path", dest = "model_path", help = "Path to the trained model.", default = (
		os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "model", "DeepMVS_final.model")
	))
parser.add_argument("--overwrite", dest = "overwrite", action = "store_true", default = False, help = "Overwrite existing results.")
parser.add_argument("--log_file", dest = "log_file", default = None, help = "Path to log file. (Default: sys.stdout)")

args = parser.parse_args()

image_width = args.image_width
image_height = args.image_height
patch_width = args.patch_width
patch_height = args.patch_height
stride_width = args.stride_width
stride_height = args.stride_height
num_depths = args.num_depths
max_num_neighbors = args.max_num_neighbors
use_gpu = args.use_gpu
image_path = args.image_path
sparse_path = args.sparse_path
output_path = args.output_path
model_path = args.model_path
overwrite = args.overwrite
log_file = args.log_file
batch_size = 1

# Constants for DenseCRF.
sigma_xy = 80.0
sigma_rgb = 15.0
sigma_d = 10.0
iteration_num = 5
compat = np.zeros((num_depths, num_depths), dtype = np.float32)
for row in range(0, num_depths):
	for col in range(0, num_depths):
		compat[row, col] = (row - col) ** 2 / sigma_d ** 2 / 2

# Create log file and output directory if needed.
if log_file is None:
	log_file = sys.stdout
else:
	log_file = open(log_file, "w")
if not os.path.exists(output_path):
	os.makedirs(output_path)

# Check if model exists.
if not os.path.exists(model_path):
	raise ValueError("Cannot find the trained model. Please download it first or specify the path to the model.")

# Load trained model.
print >> log_file, "Loading the trained model..."
log_file.flush()
model = DeepMVS(num_depths, use_gpu)
model.load_state_dict(torch.load(os.path.join(model_path)))
print >> log_file, "Successfully loaded the trained model."
log_file.flush()

# Load COLMAP sparse model.
print >> log_file, "Loading the sparse model..."
log_file.flush()
sparse_model = ColmapSparse(sparse_path, image_path, image_width, image_height)
print >> log_file, "Successfully loaded the sparse model."
log_file.flush()

# Launch plane-sweep volume generating thread.
shared_data = ({
		"ready_e": threading.Event(),
		"start_e": threading.Event(),
		"stop": False,
		"patch_width": patch_width,
		"patch_height": patch_height,
		"num_depths": num_depths,
		"max_num_neighbors": max_num_neighbors,
		"sparse_model": sparse_model
	})
worker_thread = threading.Thread(name = "generate_volume", target = generate_volume_test, args = (shared_data,))
worker_thread.start()

# Prepare VGG model and normalizer.
print >> log_file, "Creating VGG model..."
log_file.flush()
if use_gpu:
	VGG_model = vision.models.vgg19(pretrained = True).cuda()
else:
	VGG_model = vision.models.vgg19(pretrained = True)
VGG_normalize = vision.transforms.Normalize(mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225])
print >> log_file, "Successfully created VGG model."
log_file.flush()

# Loop through all images.
for (ref_image_idx, ref_image) in enumerate(sparse_model.image_list.images):
	# Check if output already exists.
	if not overwrite and os.path.exists(os.path.join(output_path, "{:}.output.npy".format(ref_image.filename))):
		print >> log_file, "Skipped {:} since the output already exists.".format(ref_image.filename)
		log_file.flush()
		continue
	# Start generating plane-sweep volume of the first patch.
	print >> log_file, "Start working on image {:d}/{:d}.".format(ref_image_idx, sparse_model.image_list.length)
	log_file.flush()
	shared_data["image_idx"] = ref_image_idx
	shared_data["target_x"] = 0
	shared_data["target_y"] = 0
	shared_data["ready_e"].clear()
	shared_data["start_e"].set()
	# Generate VGG features.
	ref_camera = sparse_model.camera_list.get_by_id(ref_image.camera_id)
	image_width = ref_camera.width
	image_height = ref_camera.height
	shared_data["ready_e"].wait()
	ref_img_full = shared_data["ref_img_full"]
	VGG_tensor = Variable(VGG_normalize(torch.FloatTensor(ref_img_full)).permute(2, 0, 1).unsqueeze(0), volatile = True)
	if use_gpu:
		VGG_tensor = VGG_tensor.cuda()
	VGG_scaling_factor = 0.01
	for i in range(0, 4):
		VGG_tensor = VGG_model.features[i].forward(VGG_tensor)
	if use_gpu:
		feature_input_1x_whole = VGG_tensor.data.cpu().clone() * VGG_scaling_factor
	else:
		feature_input_1x_whole = VGG_tensor.data.clone() * VGG_scaling_factor
	for i in range(4, 9):
		VGG_tensor = VGG_model.features[i].forward(VGG_tensor)
	if use_gpu:
		feature_input_2x_whole = VGG_tensor.data.cpu().clone() * VGG_scaling_factor
	else:
		feature_input_2x_whole = VGG_tensor.data.clone() * VGG_scaling_factor
	for i in range(9, 14):
		VGG_tensor = VGG_model.features[i].forward(VGG_tensor)
	if use_gpu:
		feature_input_4x_whole = VGG_tensor.data.cpu().clone() * VGG_scaling_factor
	else:
		feature_input_4x_whole = VGG_tensor.data.clone() * VGG_scaling_factor
	for i in range(14, 23):
		VGG_tensor = VGG_model.features[i].forward(VGG_tensor)
	if use_gpu:
		feature_input_8x_whole = VGG_tensor.data.cpu().clone() * VGG_scaling_factor
	else:
		feature_input_8x_whole = VGG_tensor.data.clone() * VGG_scaling_factor
	for i in range(23, 32):
		VGG_tensor = VGG_model.features[i].forward(VGG_tensor)
	if use_gpu:
		feature_input_16x_whole = VGG_tensor.data.cpu().clone() * VGG_scaling_factor
	else:
		feature_input_16x_whole = VGG_tensor.data.clone() * VGG_scaling_factor
	del VGG_tensor
	# Stride through entire reference image.
	predict_raw = torch.zeros(num_depths, image_height, image_width)
	border_x = (patch_width - stride_width) / 2
	border_y = (patch_height - stride_height) / 2
	col_total = (image_width - 2 * border_x - 1) / stride_width + 1
	row_total = (image_height - 2 * border_y - 1) / stride_height + 1
	for row_idx in range(0, row_total):
		for col_idx in range(0, col_total):
			print >> log_file, "Working on patch at row = {:d}/{:d} col = {:d}/{:d}".format(row_idx, row_total, col_idx, col_total)
			log_file.flush()
			# Compute patch location for this patch and next patch.
			if col_idx != col_total - 1:
				start_x = col_idx * stride_width
			else:
				start_x = image_width - patch_width
			if row_idx != row_total - 1:
				start_y = row_idx * stride_height
			else:
				start_y = image_height - patch_height
			next_col_idx = (col_idx + 1) % col_total
			next_row_idx = row_idx if col_idx != col_total - 1 else row_idx + 1
			if next_col_idx != col_total - 1:
				next_start_x = next_col_idx * stride_width
			else:
				next_start_x = image_width - patch_width
			if next_row_idx != row_total - 1:
				next_start_y = next_row_idx * stride_height
			else:
				next_start_y = image_height - patch_height
			# Read plane-sweep volume and start next patch.
			shared_data["ready_e"].wait()
			ref_img = shared_data["ref_img"].copy()
			sweep_volume = shared_data["sweep_volume"].copy()
			num_neighbors = shared_data["num_neighbors"]
			if next_row_idx < row_total:
				shared_data["target_x"] = next_start_x
				shared_data["target_y"] = next_start_y
				shared_data["ready_e"].clear()
				shared_data["start_e"].set()
			# Prepare the inputs.
			data_in_tensor = torch.FloatTensor(batch_size, 1, num_depths, 2, 3, patch_height, patch_width)
			ref_img_tensor = torch.FloatTensor(ref_img).permute(2, 0, 1).unsqueeze(0)
			data_in_tensor[0, 0, :, 0, ...] = ref_img_tensor.expand(num_depths, -1, -1, -1)
			feature_input_1x = Variable(feature_input_1x_whole[... , start_y:start_y + patch_height, start_x:start_x + patch_width], volatile = True)
			feature_input_2x = Variable(feature_input_2x_whole[... , start_y / 2:start_y / 2 + patch_height / 2, start_x / 2:start_x / 2 + patch_width / 2], volatile = True)
			feature_input_4x = Variable(feature_input_4x_whole[... , start_y / 4:start_y / 4 + patch_height / 4, start_x / 4:start_x / 4 + patch_width / 4], volatile = True)
			feature_input_8x = Variable(feature_input_8x_whole[... , start_y / 8:start_y / 8 + patch_height / 8, start_x / 8:start_x / 8 + patch_width / 8], volatile = True)
			feature_input_16x = Variable(feature_input_16x_whole[... , start_y / 16:start_y / 16 + patch_height / 16, start_x / 16:start_x / 16 + patch_width / 16], volatile = True)
			if use_gpu:
				feature_input_1x = feature_input_1x.cuda()
				feature_input_2x = feature_input_2x.cuda()
				feature_input_4x = feature_input_4x.cuda()
				feature_input_8x = feature_input_8x.cuda()
				feature_input_16x = feature_input_16x.cuda()
			# Loop through all neighbor images.
			for neighbor_idx in range(0, num_neighbors):
				data_in_tensor[0, 0, :, 1, ...] = torch.FloatTensor(np.moveaxis(sweep_volume[neighbor_idx, ...], -1, -3))
				data_in = Variable(data_in_tensor, volatile = True)
				if use_gpu:
					data_in = data_in.cuda()
				if neighbor_idx == 0:
					cost_volume = model.forward_feature(data_in, [feature_input_1x, feature_input_2x, feature_input_4x, feature_input_8x, feature_input_16x]).data[...]
				else:
					cost_volume = torch.max(cost_volume, model.forward_feature(data_in, [feature_input_1x, feature_input_2x, feature_input_4x, feature_input_8x, feature_input_16x]).data[...])
			# Make final prediction.
			predict = model.forward_predict(Variable(cost_volume[:, 0, ...], volatile = True))
			# Compute copy range.
			if col_idx == 0:
				copy_x_start = 0
				copy_x_end = patch_width - border_x
			elif col_idx == col_total - 1:
				copy_x_start = border_x + col_idx * stride_width
				copy_x_end = image_width
			else:
				copy_x_start = border_x + col_idx * stride_width
				copy_x_end = copy_x_start + stride_width
			if row_idx == 0:
				copy_y_start = 0
				copy_y_end = patch_height - border_y
			elif row_idx == row_total - 1:
				copy_y_start = border_y + row_idx * stride_height
				copy_y_end = image_height
			else:
				copy_y_start = border_y + row_idx * stride_height
				copy_y_end = copy_y_start + stride_height
			# Copy the prediction to buffer.
			predict_raw[..., copy_y_start:copy_y_end, copy_x_start:copy_x_end] = predict.data[0, :, copy_y_start - start_y:copy_y_end - start_y, copy_x_start - start_x:copy_x_end - start_x]
	# Pass through DenseCRF.
	print >> log_file, "Running DenseCRF..."
	log_file.flush()
	unary_energy = F.log_softmax(Variable(predict_raw, volatile = True), dim = 0).data.numpy()
	crf = dcrf.DenseCRF2D(image_width, image_height, num_depths)
	crf.setUnaryEnergy(-unary_energy.reshape(num_depths, image_height * image_width))
	ref_img_full = (ref_img_full * 255.0).astype(np.uint8)
	crf.addPairwiseBilateral(sxy=(sigma_xy, sigma_xy), srgb=(sigma_rgb, sigma_rgb, sigma_rgb), rgbim=ref_img_full, compat=compat, kernel=dcrf.FULL_KERNEL, normalization=dcrf.NORMALIZE_SYMMETRIC)
	new_raw = crf.inference(iteration_num)
	new_raw = np.array(new_raw).reshape(num_depths, image_height, image_width)
	new_predict = np.argmax(new_raw, 0).astype(np.float32) / (num_depths - 1) * ref_image.estimated_max_disparity
	# Store the results.
	output_dir = os.path.dirname(os.path.join(output_path, ref_image.filename))
	if not os.path.exists(output_dir):
		os.makedirs(output_dir)
	np.save(os.path.join(output_path, "{:}.output.npy".format(ref_image.filename)), new_predict)
	imageio.imwrite(os.path.join(output_path, "{:}.output.png".format(ref_image.filename)), (new_predict / ref_image.estimated_max_disparity).clip(0.0, 1.0))
	print >> log_file, "Result has been saved to {:}.".format(os.path.join(output_path, "{:}.output.npy".format(ref_image.filename)))
	log_file.flush()

# Terminate worker threads.
shared_data["stop"] = True
shared_data["start_e"].set()

# Finished.
print >> log_file, "Finished running DeepMVS."
print >> log_file, "Results can be found in {:}".format(output_path)