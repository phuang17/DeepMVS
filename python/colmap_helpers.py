import os
import re

import numpy as np
import cv2
import imageio
from pyquaternion import Quaternion

class PointList:

	class Point:
		def __init__(self, id, coord):
			self.id = id
			self.coord = coord

	def __init__(self, line):
		words = line.split()
		self.points = []
		for i in range(0, len(words) / 3):
			id = int(words[3 * i + 2])
			if id == -1:
				continue
			coord = np.array([float(words[3 * i + 0]), float(words[3 * i + 1])])
			self.points.append(self.Point(id, coord))
		self.length = len(self.points)

	def get_by_id(self, id):
		for idx in range(0, self.length):
			if id == self.points[idx].id:
				return self.points[idx]
		return None

class ImageList:

	class Image:
		def __init__(self, id, extrinsic, camera_id, filename, point_list):
			self.id = id
			self.extrinsic = extrinsic
			self.camera_id = camera_id
			self.filename = filename
			self.point_list = point_list

	def __init__(self, path):
		self.images = []
		with open(path) as f:
			lines = f.readlines()
		self.length = int(re.match(r"# Number of images: (\d+),",lines[3]).groups()[0])
		for i in range(0, self.length):
			words = lines[4 + i * 2].split()
			id = int(words[0])
			extrinsic = Quaternion(float(words[1]), float(words[2]), float(words[3]), float(words[4])).transformation_matrix
			extrinsic[0,3] = float(words[5])
			extrinsic[1,3] = float(words[6])
			extrinsic[2,3] = float(words[7])
			camera_id = int(words[8])
			filename = words[9]
			point_list = PointList(lines[4 + i * 2 + 1])
			self.images.append(self.Image(id, extrinsic, camera_id, filename, point_list))

	def get_by_id(self, id):
		for idx in range(0, self.length):
			if id == self.images[idx].id:
				return self.images[idx]
		return None

class CameraList:

	class Camera:
		def __init__(self, id, width, height, fx, fy, cx, cy):
			self.id = id
			self.width = width
			self.height = height
			self.fx = fx
			self.fy = fy
			self.cx = cx
			self.cy = cy

	def __init__(self, path):
		self.cameras = []
		with open(path) as f:
			lines = f.readlines()
		self.length = int(re.match(r"# Number of cameras: (\d+)",lines[2]).groups()[0])
		for camera_idx in range(0, self.length):
			words = lines[3 + camera_idx].split()
			id = int(words[0])
			width = int(words[2])
			height = int(words[3])
			fx = float(words[4])
			fy = float(words[5])
			cx = float(words[6])
			cy = float(words[7])
			self.cameras.append(self.Camera(id, width, height, fx, fy, cx, cy))

	def get_by_id(self, id):
		for idx in range(0, self.length):
			if id == self.cameras[idx].id:
				return self.cameras[idx]
		return None

class PointCloud:

	class Point:
		def __init__(self, id, coord):
			self.id = id
			self.coord = coord

	def __init__(self, path):
		self.points = []
		with open(path) as f:
			lines = f.readlines()
		self.length = int(re.match(r"# Number of points: (\d+),",lines[2]).groups()[0])
		for i in range(0, self.length):
			words = lines[3 + i].split()
			id = int(words[0])
			x = float(words[1])
			y = float(words[2])
			z = float(words[3])
			coord = np.array([x, y, z, 1.0])
			self.points.append(self.Point(id, coord))

	def get_by_id(self, id):
		for idx in range(0, self.length):
			if id == self.points[idx].id:
				return self.points[idx]
		return None

class ColmapSparse:
	def __init__(self, sparse_path, image_path, image_width = -1, image_height = -1, max_num_neighbors = 16):
		image_list_path = os.path.join(sparse_path, "images.txt")
		camera_list_path = os.path.join(sparse_path, "cameras.txt")
		point_cloud_path = os.path.join(sparse_path, "points3D.txt")
		if not os.path.exists(image_list_path):
			raise ValueError("{:} does not exist.".format(image_list_path))
		if not os.path.exists(camera_list_path):
			raise ValueError("{:} does not exist.".format(camera_list_path))
		if not os.path.exists(point_cloud_path):
			raise ValueError("{:} does not exist.".format(point_cloud_path))
		self.image_list = ImageList(image_list_path)
		self.camera_list = CameraList(camera_list_path)
		self.point_cloud = PointCloud(point_cloud_path)
		self.load_images(image_path)
		self.resize(image_width, image_height)
		self.estimate_max_disparities()
		self.generate_neighbor_list(max_num_neighbors)

	def load_images(self, image_path):
		for image_idx in range(self.image_list.length):
			self.image_list.images[image_idx].rgb = imageio.imread(os.path.join(image_path, self.image_list.images[image_idx].filename)).astype(np.float32) / 255.0

	def resize(self, image_width, image_height):
		if image_width < 0 and image_height < 0:
			return
		for camera_idx in range(self.camera_list.length):
			orig_image_width = self.camera_list.cameras[camera_idx].width
			orig_image_height = self.camera_list.cameras[camera_idx].height
			if image_width < 0:
				target_image_width = image_height * orig_image_width / orig_image_height
				target_image_height = image_height
			elif image_height < 0:
				target_image_width = image_width
				target_image_height = image_width * orig_image_height / orig_image_width
			else:
				target_image_width = image_width
				target_image_height = image_height
			width_ratio = float(target_image_width) / orig_image_width
			height_ratio = float(target_image_height) / orig_image_height
			self.camera_list.cameras[camera_idx].width = target_image_width
			self.camera_list.cameras[camera_idx].height = target_image_height
			self.camera_list.cameras[camera_idx].fx *= width_ratio
			self.camera_list.cameras[camera_idx].fy *= height_ratio
			self.camera_list.cameras[camera_idx].cx *= width_ratio
			self.camera_list.cameras[camera_idx].cy *= height_ratio
			self.camera_list.cameras[camera_idx].width_ratio = width_ratio
			self.camera_list.cameras[camera_idx].height_ratio = height_ratio

		for image_idx in range(self.image_list.length):
			camera = self.camera_list.get_by_id(self.image_list.images[image_idx].camera_id)
			self.image_list.images[image_idx].rgb = cv2.resize(self.image_list.images[image_idx].rgb, (camera.width, camera.height), interpolation = cv2.INTER_AREA)
			for point_idx in range(self.image_list.images[image_idx].point_list.length):
				self.image_list.images[image_idx].point_list.points[point_idx].coord[0] *= camera.width_ratio
				self.image_list.images[image_idx].point_list.points[point_idx].coord[1] *= camera.height_ratio

	def estimate_max_disparities(self, percentile = 0.99, stretch = 1.333333):
		for (img_idx, image) in enumerate(self.image_list.images):
			camera = self.camera_list.get_by_id(image.camera_id)
			disparity_list = []
			for (point_idx, point) in enumerate(self.point_cloud.points):
				coord = image.extrinsic.dot(point.coord)
				new_x = (coord[0] / coord[2] * camera.fx + camera.cx) 
				new_y = (coord[1] / coord[2] * camera.fy + camera.cy) 
				new_d = 1.0 / coord[2]
				if new_x >= 0.0 and new_x < camera.width and new_y >= 0.0 and new_y < camera.height and new_d > 0.0:
					disparity_list.append(new_d)
			disparity_list = np.sort(np.array(disparity_list))
			self.image_list.images[img_idx].estimated_max_disparity = disparity_list[int(disparity_list.shape[0] * percentile)] * stretch

	def generate_neighbor_list(self, max_num_neighbors):
		point_id_list = []
		for (ref_idx, ref_image) in enumerate(self.image_list.images):
			point_id_set = set()
			for (ref_point_idx, ref_point) in enumerate(ref_image.point_list.points):
				point_id_set.add(ref_point.id)
			point_id_list.append(point_id_set)
		for (ref_idx, ref_image) in enumerate(self.image_list.images):
			shared_feature_list = []
			for (n_idx, n_image) in enumerate(self.image_list.images):
				if n_idx == ref_idx:
					shared_feature_list.append(0)
					continue
				shared_feature_list.append(len(point_id_list[ref_idx] & point_id_list[n_idx]))
			index_order = np.argsort(np.array(shared_feature_list))[::-1]
			neighbor_list = []
			for idx in index_order:
				if shared_feature_list[idx] == 0:
					break
				neighbor_list.append(idx)
				if len(neighbor_list) == max_num_neighbors:
					break
			self.image_list.images[ref_idx].neighbor_list = neighbor_list
