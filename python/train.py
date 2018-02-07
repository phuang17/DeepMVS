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

from model import DeepMVS_PT, DeepMVS, weights_init
from generate_volume_train import generate_volume_train

# Parse arguments
parser = argparse.ArgumentParser(description = "Train DeepMVS.")
parser.add_argument("--patch_width", dest = "patch_width", type = int, default = 64, help = "Width of patches.")
parser.add_argument("--patch_height", dest = "patch_height", type = int, default = 64, help = "Height of patches.")
parser.add_argument("--num_depths", dest = "num_depths", type = int, default = 100, help = "Number of disparity levels.")
parser.add_argument("--max_num_neighbors", dest = "max_num_neighbors", type = int, default = 4, help = "Maximum number of neighbor images.")
parser.add_argument("--num_threads", dest = "num_threads", type = int, default = 4, help = "Number of threads for plane-sweep volume generation.")
parser.add_argument("--no_gpu", dest = "use_gpu", action = "store_false", default = True, help = "Disable use of GPU.")
parser.add_argument("--dataset_path", dest = "dataset_path", default = (
		os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "dataset", "train")
	), help = "Path to the training dataset.")
parser.add_argument("--model_path", dest = "model_path", default = (
		os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "model")
	), help = "Path to store models.")
parser.add_argument("--snapshot_period", dest = "snapshot_period", type = int, default = 10000, help = "Take snapshot every n iterations. (0: no snapshot)")
parser.add_argument("--retrain", dest = "retrain", action = "store_true", default = False, help = "Retrain the network from scratch.")
parser.add_argument("--log_file", dest = "log_file", default = None, help = "Path to log file. (Default: sys.stdout)")

args = parser.parse_args()

patch_width = args.patch_width
patch_height = args.patch_height
num_depths = args.num_depths
max_num_neighbors = args.max_num_neighbors
num_threads = args.num_threads
use_gpu = args.use_gpu
dataset_path = args.dataset_path
model_path = args.model_path
snapshot_period = args.snapshot_period
retrain = args.retrain
log_file = args.log_file
batch_size = 1

# Check if training dataset is downloaded.
dataset_downloaded = True
if os.path.exists(dataset_path):
	DATASET_LIST = ([
			"GTAV_540", "GTAV_720",
			"mvs_achteck_turm", "mvs_breisach", "mvs_citywall", 
			"rgbd_10_to_20_3d_train", "rgbd_10_to_20_handheld_train", "rgbd_10_to_20_simple_train", "rgbd_20_to_inf_3d_train", "rgbd_20_to_inf_handheld_train", "rgbd_20_to_inf_simple_train",
			"scenes11_train", 
			"sun3d_train_0.01m_to_0.1m", "sun3d_train_0.1m_to_0.2m", "sun3d_train_0.2m_to_0.4m", "sun3d_train_0.4m_to_0.8m", "sun3d_train_0.8m_to_1.6m", "sun3d_train_1.6m_to_infm"
		])
	for dataset in DATASET_LIST:
		if not os.path.exists(os.path.join(dataset_path, dataset)):
			print >> log_file, "Cannot find dataset '{:}'".format(dataset)
			dataset_downloaded = False
			break
else:
	os.makedirs(dataset_path)
	dataset_downloaded = False

# Try downloading training datasets if it has not been done.
if not dataset_downloaded:
	print >> log_file, "Training datasets must be downloaded before training DeepMVS."
	print >> log_file, "Run 'python python/download_training_datasets.py' to download the training datasets."
	sys.exit()

# Create model directory and log file.
if not os.path.exists(model_path):
	os.makedirs(model_path)
if log_file is None:
	log_file = sys.stdout
else:
	log_file = open(log_file, "w")

# Create worker threads for volume generation.
shared_datas = []
threads = []
for i in range(0, num_threads):
	shared_datas.append({
		"ready_e": threading.Event(),
		"start_e": threading.Event(),
		"stop": False
	})
	threads.append(threading.Thread(name = "generate_volume_{:d}".format(i), target = generate_volume_train, args = 
		(shared_datas[i], max_num_neighbors, num_depths, patch_height, patch_width, dataset_path)
	))
	threads[i].start()
	shared_datas[i]["ready_e"].clear()
	shared_datas[i]["start_e"].set()

# Check if we can resume from last snapshot.
iteration_stop = 320000
if retrain:
	need_pretrain = True
	iteration_start = 0
elif os.path.exists(os.path.join(model_path, "DeepMVS_PT_final.model")):
	need_pretrain = False
elif snapshot_period != 0:
	need_pretrain = True
	iter_idx = 0
	while os.path.exists(os.path.join(model_path, "DeepMVS_PT_snapshot_{:d}.model".format(iter_idx + snapshot_period))):
		iter_idx += snapshot_period
	iteration_start = iter_idx
else:
	need_pretrain = True
	iteration_start = 0

# Train the DeepMVS pre-train network.
# Use function to ensure the local variables are cleaned up upon exiting the scope.
def train_DeepMVS_PT():
	# Initialization.
	model = DeepMVS_PT(num_depths, use_gpu)
	lr = 1e-5
	grad_clip = 1.0
	optimizer = torch.optim.Adam(model.parameters(), lr = lr)
	if iteration_start == 0:
		model.apply(weights_init)
	else:
		model.load_state_dict(torch.load(os.path.join(model_path, "DeepMVS_PT_snapshot_{:d}.model".format(iteration_start))))
		optimizer.load_state_dict(torch.load(os.path.join(model_path, "DeepMVS_PT_snapshot_{:d}.optimizer".format(iteration_start))))
	# Start training.
	data_in = torch.FloatTensor(batch_size, max_num_neighbors, num_depths, 2, 3, patch_height, patch_width)
	data_gt = torch.LongTensor(batch_size, patch_height, patch_width)
	invalid_mask = torch.ByteTensor(batch_size, patch_height, patch_width)
	thread_idx = 0
	print >> log_file, "Start training DeepMVS_PT from iteration {:d}.".format(iteration_start)
	for iteration_idx in range(iteration_start, iteration_stop):
		# Load a plane-sweep volume.
		while not shared_datas[thread_idx]["ready_e"].wait(1e-3):
			thread_idx = (thread_idx + 1) % num_threads
		ref_img = shared_datas[thread_idx]["ref_img"].copy()
		sweep_volume = shared_datas[thread_idx]["sweep_volume"].copy()
		ref_depth = shared_datas[thread_idx]["ref_depth"].copy()
		valid_mask = shared_datas[thread_idx]["valid_mask"].copy()
		num_neighbors = shared_datas[thread_idx]["num_neighbors"]
		shared_datas[thread_idx]["ready_e"].clear()
		shared_datas[thread_idx]["start_e"].set()
		# Prepare inputs.
		optimizer.zero_grad()
		ref_img_tensor = torch.FloatTensor(np.moveaxis(ref_img, -1, 0))
		data_in[0, 0:num_neighbors, :, 0, ...] = ref_img_tensor.expand(num_neighbors, num_depths, -1, -1, -1)
		data_in[0, 0:num_neighbors, :, 1, ...] = torch.FloatTensor(np.moveaxis(sweep_volume, -1, -3))
		data_gt[...] = torch.LongTensor(np.round(ref_depth * (num_depths - 1.0)).clip(0, num_depths - 1.0).astype(int))
		invalid_mask[...] = torch.ByteTensor(1 - valid_mask)
		data_gt.masked_fill_(invalid_mask, -1)
		if use_gpu:
			data_in_var = Variable(data_in[:, 0:num_neighbors, ...], requires_grad = True).cuda()
			data_gt_var = Variable(data_gt, requires_grad = False).cuda()
		else:
			data_in_var = Variable(data_in[:, 0:num_neighbors, ...], requires_grad = True)
			data_gt_var = Variable(data_gt, requires_grad = False)
		# Update parameters.
		predict = model.forward(data_in_var)
		loss = model.layer_loss(predict, data_gt_var)
		loss.backward()
		nn.utils.clip_grad_norm(model.parameters(), grad_clip)
		optimizer.step()
		# Save snapshot if needed.
		if snapshot_period != 0:
			if((iteration_idx + 1) % snapshot_period == 0):
				torch.save(model.state_dict(), os.path.join(model_path, "DeepMVS_PT_snapshot_{:d}.model".format(iteration_idx + 1)))
				torch.save(optimizer.state_dict(), os.path.join(model_path, "DeepMVS_PT_snapshot_{:d}.optimizer".format(iteration_idx + 1)))
		# Print loss to log file.
		print >> log_file, "Iter {:d}: loss = {:.6e}".format(iteration_idx, loss.data[0])
		log_file.flush()
	# Save final trained model.
	torch.save(model.state_dict(), os.path.join(model_path, "DeepMVS_PT_final.model"))
	torch.save(optimizer.state_dict(), os.path.join(model_path, "DeepMVS_PT_final.optimizer"))

if need_pretrain:
	train_DeepMVS_PT()

# Check if we can resume from last snapshot.
iteration_stop = 320000
if retrain or snapshot_period == 0:
	iteration_start = 0
else:
	iter_idx = 0
	while os.path.exists(os.path.join(model_path, "DeepMVS_snapshot_{:d}.model".format(iter_idx + snapshot_period))):
		iter_idx += snapshot_period
	iteration_start = iter_idx

# Train the complete DeepMVS network.
# Use function to ensure the local variables are cleaned up upon exiting the scope.
def train_DeepMVS():
	# Initialization.
	model = DeepMVS(num_depths, use_gpu)
	lr = 1e-6
	grad_clip = 0.1
	optimizer = torch.optim.Adam(model.parameters(), lr = lr)
	if iteration_start == 0:
		model.apply(weights_init)
		model.load_state_dict(torch.load(os.path.join(model_path, "DeepMVS_PT_final.model")), strict = False)
	else:
		model.load_state_dict(torch.load(os.path.join(model_path, "DeepMVS_snapshot_{:d}.model".format(iteration_start))))
		optimizer.load_state_dict(torch.load(os.path.join(model_path, "DeepMVS_snapshot_{:d}.optimizer".format(iteration_start))))
	if use_gpu:
		VGG_model = vision.models.vgg19(pretrained = True).cuda()
	else:
		VGG_model = vision.models.vgg19(pretrained = True)
	VGG_normalize = vision.transforms.Normalize(mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225])
	# Start training.
	data_in = torch.FloatTensor(batch_size, max_num_neighbors, num_depths, 2, 3, patch_height, patch_width)
	data_gt = torch.LongTensor(batch_size, patch_height, patch_width)
	invalid_mask = torch.ByteTensor(batch_size, patch_height, patch_width)
	thread_idx = 0
	print >> log_file, "Start training DeepMVS from iteration {:d}.".format(iteration_start)
	for iteration_idx in range(iteration_start, iteration_stop):
		# Load a plane-sweep volume.
		while not shared_datas[thread_idx]["ready_e"].wait(1e-3):
			thread_idx = (thread_idx + 1) % num_threads
		ref_img = shared_datas[thread_idx]["ref_img"].copy()
		ref_img_full = shared_datas[thread_idx]["ref_img_full"].copy()
		target_x = shared_datas[thread_idx]["target_x"]
		target_y = shared_datas[thread_idx]["target_y"]
		sweep_volume = shared_datas[thread_idx]["sweep_volume"].copy()
		ref_depth = shared_datas[thread_idx]["ref_depth"].copy()
		valid_mask = shared_datas[thread_idx]["valid_mask"].copy()
		num_neighbors = shared_datas[thread_idx]["num_neighbors"]
		shared_datas[thread_idx]["ready_e"].clear()
		shared_datas[thread_idx]["start_e"].set()
		# Prepare inputs.
		optimizer.zero_grad()
		ref_img_tensor = torch.FloatTensor(np.moveaxis(ref_img, -1, 0))
		data_in[0, 0:num_neighbors, :, 0, ...] = ref_img_tensor.expand(num_neighbors, num_depths, -1, -1, -1)
		data_in[0, 0:num_neighbors, :, 1, ...] = torch.FloatTensor(np.moveaxis(sweep_volume, -1, -3))
		data_gt[...] = torch.LongTensor(np.round(ref_depth * (num_depths - 1.0)).clip(0, num_depths - 1.0).astype(int))
		invalid_mask[...] = torch.ByteTensor(1 - valid_mask)
		data_gt.masked_fill_(invalid_mask, -1)
		if use_gpu:
			data_in_var = Variable(data_in[:, 0:num_neighbors, ...], requires_grad = True).cuda()
			data_gt_var = Variable(data_gt, requires_grad = False).cuda()
		else:
			data_in_var = Variable(data_in[:, 0:num_neighbors, ...], requires_grad = True)
			data_gt_var = Variable(data_gt, requires_grad = False)
		# Compute VGG Features.
		VGG_scaling_factor = 0.01
		if use_gpu:
			VGG_temp_var = Variable(VGG_normalize(torch.FloatTensor(ref_img_full)).permute(2, 0, 1).unsqueeze(0), volatile = True).cuda()
		else:
			VGG_temp_var = Variable(VGG_normalize(torch.FloatTensor(ref_img_full)).permute(2, 0, 1).unsqueeze(0), volatile = True)
		for i in range(0, 4): # conv_1_2
			VGG_temp_var = VGG_model.features[i].forward(VGG_temp_var)
		feature_input_1x = Variable(VGG_temp_var.data[... , target_y:target_y + patch_height, target_x:target_x + patch_width].clone() * VGG_scaling_factor, requires_grad = True)
		for i in range(4, 9): # conv_2_2
			VGG_temp_var = VGG_model.features[i].forward(VGG_temp_var)
		feature_input_2x = Variable(VGG_temp_var.data[... , target_y / 2:target_y / 2 + patch_height / 2, target_x / 2:target_x / 2 + patch_width / 2].clone() * VGG_scaling_factor, requires_grad = True)
		for i in range(9, 14): # conv_3_2
			VGG_temp_var = VGG_model.features[i].forward(VGG_temp_var)
		feature_input_4x = Variable(VGG_temp_var.data[... , target_y / 4:target_y / 4 + patch_height / 4, target_x / 4:target_x / 4 + patch_width / 4].clone() * VGG_scaling_factor, requires_grad = True)
		for i in range(14, 23): # conv_4_2
			VGG_temp_var = VGG_model.features[i].forward(VGG_temp_var)
		feature_input_8x = Variable(VGG_temp_var.data[... , target_y / 8:target_y / 8 + patch_height / 8, target_x / 8:target_x / 8 + patch_width / 8].clone() * VGG_scaling_factor, requires_grad = True)
		for i in range(23, 32): # conv_5_2
			VGG_temp_var = VGG_model.features[i].forward(VGG_temp_var)
		feature_input_16x = Variable(VGG_temp_var.data[... , target_y / 16:target_y / 16 + patch_height / 16, target_x / 16:target_x / 16 + patch_width / 16].clone() * VGG_scaling_factor, requires_grad = True)
		del VGG_temp_var
		# Update parameters.
		predict = model.forward(data_in_var, [feature_input_1x, feature_input_2x, feature_input_4x, feature_input_8x, feature_input_16x])
		loss = model.layer_loss(predict, data_gt_var)
		loss.backward()
		nn.utils.clip_grad_norm(model.parameters(), grad_clip)
		optimizer.step()
		# Save snapshot if needed.
		if snapshot_period != 0:
			if((iteration_idx + 1) % snapshot_period == 0):
				torch.save(model.state_dict(), os.path.join(model_path, "DeepMVS_snapshot_{:d}.model".format(iteration_idx + 1)))
				torch.save(optimizer.state_dict(), os.path.join(model_path, "DeepMVS_snapshot_{:d}.optimizer".format(iteration_idx + 1)))
		# Print loss to log file.
		print >> log_file, "Iter {:d}: loss = {:.6e}".format(iteration_idx, loss.data[0])
		log_file.flush()
	# Save final trained model.
	torch.save(model.state_dict(), os.path.join(model_path, "DeepMVS_final.model"))
	torch.save(optimizer.state_dict(), os.path.join(model_path, "DeepMVS_final.optimizer"))

train_DeepMVS()

# Terminate worker threads.
for i in range(0, num_threads):
	shared_datas[i]["stop"] = True
	shared_datas[i]["start_e"].set()
# Finished.
print >> log_file, "Finished training DeepMVS."
print >> log_file, "Trained model can be found at {:}".format(os.path.join(model_path, "DeepMVS_final.model"))
