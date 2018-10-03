import os
import argparse
import cv2
import imageio
import numpy as np
from scipy import linalg as la

# Parse arguments.
parser = argparse.ArgumentParser(description = "Evaluate predicted disparity maps.")
parser.add_argument("--image_path", dest = "image_path", help = "Path to the images.", required = True)
parser.add_argument("--sparse_path", dest = "sparse_path", help = "Path to the sparse reconstruction.", required = True)
parser.add_argument("--output_path", dest = "output_path", help = "Path to store the predicted results.", required = True)
parser.add_argument("--gt_path", dest = "gt_path", help = "Path to store the ground truth results. The gt depth filenames should be <image_name>.depth.npy", required = True)
parser.add_argument("--load_bin", dest = "load_bin", type = bool, default = False, help = "Set if you want to load COLMAP .bin files")
# TODO: Support --gt_type = "colmap_bin"
parser.add_argument("--gt_type", dest = "gt_type", choices = ["depth", "disparity"], default = "disparity", help = "Specify whether the ground truth depth files store depth or disparity (=1/depth).")
parser.add_argument("--output_type", dest = "output_type", choices = ["depth", "disparity"], default = "disparity", help = "Specify whether the predicted depth files store depth value or disparity (=1/depth).")
parser.add_argument("--image_width", dest = "image_width", type = int, help = "Image width (>0).", required = True)
parser.add_argument("--image_height", dest = "image_height", type = int, help = "Image height (>0).", required = True)
parser.add_argument("--skip_rephoto", dest = "skip_rephoto", action = "store_true", default = False, help = "Skip rephoto error to speed up evaluation.")
parser.add_argument("--store_rephoto", dest = "store_rephoto", action = "store_true", default = False, help = "Store the rephotography result using the predicted depths to <output_path>/rephoto.")
# Set border = 6 since COLMAP predicts unuseful values for the borders.
parser.add_argument("--border", dest = "border", type = int, default = 6, help = "Width of the borders to ignore from evaluation.")
parser.add_argument("--size_mismatch", dest = "size_mismatch", choices = ["throw", "crop_pad", "resize"], default = "throw", 
	help = 
"""Specify what to do if the size of depth maps do not match the specified image_width and image_height.
"throw": throw an error.
"crop_pad": crop or pad the depth maps.
"resize": resample the depth maps using nearest neighbor sampling.
"""
	)

args = parser.parse_args()

image_path = args.image_path
sparse_path = args.sparse_path
output_path = args.output_path
gt_path = args.gt_path
load_bin = args.load_bin
gt_type = args.gt_type
output_type = args.output_type
image_width = args.image_width
image_height = args.image_height
skip_rephoto = args.skip_rephoto
store_rephoto = args.store_rephoto
border = args.border
size_mismatch = args.size_mismatch

if args.load_bin:
	from colmap_helpers_for_bin import ColmapSparse
else:
	from colmap_helpers import ColmapSparse

# Crop or pad the depth map to specific size.
def crop_pad(img, w, h):
	img_w = img.shape[1]
	img_h = img.shape[0]
	if img_w > w:
		padding = (img_w - w) / 2
		img = img[:,padding:padding+w]
	elif img_w < w:
		padding_pre = (w - img_w) / 2
		padding_post = w - img_w - padding_pre
		img = np.pad(img, ((0,0), (padding_pre,padding_post)), "edge")
	if img_h > h:
		padding = (img_h - h) / 2
		img = img[padding:padding+h,:]
	elif img_h < h:
		padding_pre = (h - img_h) / 2
		padding_post = h - img_h - padding_pre
		img = np.pad(img, ((padding_pre,padding_post), (0,0)), "edge")
	return img

# Resize the depth map to specific size.
def resize(img, w, h):
	img = cv2.resize(img, (w, h), interpolation = cv2.INTER_NEAREST)
	return img

# Compute rephoto error.
def get_rephoto_diff(rephoto_path, sparse_model, frame_idx, predict_depth, mask = None):

	# Get reference image and camera pose.
	target_image = sparse_model.image_list.images[frame_idx]
	target_camera = sparse_model.camera_list.get_by_id(target_image.camera_id)
	num_neighbors = len(target_image.neighbor_list)

	# Ignore borders.
	gt_img = target_image.rgb
	if border > 0:
		gt_img = gt_img[border:-border, border:-border, :]
	input_width = gt_img.shape[1]
	input_height = gt_img.shape[0]

	# Store color candidates for each pixel.
	rephoto_volume = -np.ones((num_neighbors, input_height, input_width, 3), dtype = np.float32)

	# Loop through all neighbors.
	for (idx, neighbor_idx) in enumerate(target_image.neighbor_list):
		print "  Warping neighbor No. {:d}/{:d}".format(idx, num_neighbors)
		# Get neighbor image and camera pose.
		n_image = sparse_model.image_list.images[neighbor_idx]
		n_camera = sparse_model.camera_list.get_by_id(n_image.camera_id)
		rgb_dst = n_image.rgb
		trans_matrix = n_image.extrinsic.dot(la.inv(target_image.extrinsic))
		# Warp the neighbor image to reference view.
		coord_buffer = np.zeros((input_height, input_width, 4), dtype = np.float32)
		coord_buffer[..., 0] = (border + np.mgrid[0:input_height, 0:input_width][1] - target_camera.cx) / target_camera.fx
		coord_buffer[..., 1] = (border + np.mgrid[0:input_height, 0:input_width][0] - target_camera.cy) / target_camera.fy
		coord_buffer[..., 2] = 1.0
		coord_buffer[..., 3] = np.where(predict_depth == 0.0, 0.0, 1.0)
		coord_buffer[..., 0:3] = np.where(predict_depth[..., np.newaxis] == 0.0, coord_buffer[..., 0:3], coord_buffer[..., 0:3] / predict_depth[..., np.newaxis])
		coord_buffer[...] = np.moveaxis(trans_matrix.dot(coord_buffer[..., np.newaxis])[..., 0], 0, -1)
		x_map = np.where(coord_buffer[..., 2] > 0.0, 
			coord_buffer[..., 0] / coord_buffer[..., 2] * n_camera.fx + n_camera.cx, -1) 
		y_map = np.where(coord_buffer[..., 2] > 0.0,
			coord_buffer[..., 1] / coord_buffer[..., 2] * n_camera.fy + n_camera.cy, -1) 
		cv2.remap(rgb_dst, x_map, y_map, cv2.INTER_LINEAR, rephoto_volume[idx, ...], cv2.BORDER_CONSTANT, (-1.0, -1.0, -1.0))

	# Select the median color for each pixel.
	rephoto_volume = np.sort(rephoto_volume, axis = 0)[::-1, ...]
	valid_count = num_neighbors - np.sum(rephoto_volume < 0.0, axis = 0)
	valid_mask = valid_count != 0
	chosen_indices = (valid_count) / 2
	rephoto_img = np.zeros((input_height, input_width, 3), dtype = np.float32)
	row_grid = np.mgrid[0:input_height, 0:input_width, 0:3][0]
	col_grid = np.mgrid[0:input_height, 0:input_width, 0:3][1]
	channel_grid = np.mgrid[0:input_height, 0:input_width, 0:3][2]
	rephoto_img = rephoto_volume[chosen_indices, row_grid, col_grid, channel_grid]
	rephoto_img[np.logical_not(valid_mask)] = 0.5

	# Compute rephoto error.
	valid_mask = valid_mask[..., 0]
	diff = np.sum(np.abs(gt_img - rephoto_img), axis = -1)
	rephoto_img = np.where(valid_mask[..., np.newaxis], rephoto_img, np.array([0.0, 1.0, 0.0])[np.newaxis, np.newaxis, :])

	# Apply masks if needed.
	if not mask is None:
		rephoto_img = np.where(mask[..., np.newaxis], rephoto_img, np.array([1.0, 0.0, 0.0])[np.newaxis, np.newaxis, :])
		valid_mask = np.logical_and(valid_mask, mask)

	# Store rephoto images if needed.
	if store_rephoto:
		output_dir = os.path.dirname(os.path.join(rephoto_path, target_image.filename))
		if not os.path.exists(output_dir):
			os.makedirs(output_dir)
		imageio.imwrite("{:}/{:}.rephoto.png".format(rephoto_path, target_image.filename), rephoto_img.clip(0.0, 1.0))
		imageio.imwrite("{:}/{:}.gt_rgb.png".format(rephoto_path, target_image.filename), gt_img.clip(0.0, 1.0))

	return diff[valid_mask]

# Load COLMAP sparse model.
print "Loading the sparse model..."
sparse_model = ColmapSparse(sparse_path, image_path, image_width, image_height)
print "Successfully loaded the sparse model."

# Loop through all reference images.
errors_L1 = []
errors_rephoto = []
for (frame_idx, frame) in enumerate(sparse_model.image_list.images):
	print "Processing reference image No. {:d}/{:d}".format(frame_idx, sparse_model.image_list.length)

	# Load ground truth depths.
	gt_depth = np.load("{:}/{:}.depth.npy".format(gt_path, frame.filename))
	gt_depth = np.pad(gt_depth, ((0,0), (15,15)), "edge")
	if gt_depth.shape[0] != image_height or gt_depth.shape[1] != image_width:
		if size_mismatch == "throw":
			raise RuntimeError("Invalid size of gt_depth. gt_depth has size = {:} but the specified image size = ({:d}, {:d}).".format(gt_depth.shape, image_height, image_width))
		elif size_mismatch == "crop_pad":
			gt_depth = crop_pad(gt_depth, image_width, image_height)
		elif size_mismatch == "resize":
			gt_depth = resize(gt_depth, image_width, image_height)
		else:
			raise ValueError("size_mismatch is not supported")
	if border > 0:
		gt_depth = gt_depth[border:-border, border:-border]
	if gt_type == "depth":
		gt_valid = gt_depth > 0.0
		gt_depth = np.where(gt_valid, 1.0 / gt_depth, 0.0)
	elif gt_type == "disparity":
		# In ground truth, disparity = 0 represents invalid values.
		gt_valid = gt_depth > 0.0
	else:
		raise ValueError("gt_type is not supported")
	
	# Load predicted depths.
	output_depth = np.load("{:}/{:}.output.npy".format(output_path, frame.filename))
	if output_depth.shape[0] != image_height or output_depth.shape[1] != image_width:
		if size_mismatch == "throw":
			raise RuntimeError("Invalid size of output_depth. output_depth has size = {:} but the specified image size = ({:d}, {:d}).".format(output_depth.shape, image_height, image_width))
		elif size_mismatch == "crop_pad":
			output_depth = crop_pad(output_depth, image_width, image_height)
		elif size_mismatch == "resize":
			output_depth = resize(output_depth, image_width, image_height)
		else:
			raise ValueError("size_mismatch is not supported")
	if border > 0:
		output_depth = output_depth[border:-border, border:-border]
	if output_type == "depth":
		output_valid = output_depth > 0.0
		output_depth = np.where(output_valid, 1.0 / output_depth, 0.0)
	elif output_type == "disparity":
		# In predicted results, disparity = 0 is a valid value.
		output_valid = output_depth >= 0.0
	else:
		raise ValueError("output_type is not supported")
	
	
	
	# Compute L1 error.
	valid_mask = np.logical_and(gt_valid, output_valid)
	error_L1 = np.abs(output_depth - gt_depth)[valid_mask]
	errors_L1.extend(error_L1.flatten().tolist())

	# Compute rephoto error.
	if not skip_rephoto:
		error_rephoto = get_rephoto_diff(os.path.join(output_path, "rephoto"), sparse_model, frame_idx, output_depth, output_valid)
		errors_rephoto.extend(error_rephoto.tolist())

# Report errors:
mean_L1 = np.mean(errors_L1)
print "Disparity L1 error = {:f}, number of valid pixels = {:d}".format(mean_L1, len(errors_L1))
if not skip_rephoto:
	mean_rephoto = np.mean(errors_rephoto)
	print "Rephoto error = {:f}, number of valid pixels = {:d}".format(mean_rephoto, len(errors_rephoto))
