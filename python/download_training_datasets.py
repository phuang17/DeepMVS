import os
import sys
import subprocess
import json

import numpy as np
import imageio
from imageio.plugins import freeimage
import h5py
from lz4.block import decompress

def download_demon_datasets(path):
    print """
===================================
Note: DeMoN dataset is not created by us. If you use these datasets in your work, please visit the url below for information about citations. 
https://github.com/lmb-freiburg/demon/blob/master/datasets/download_traindata.sh
===================================
    """
    DATASET_NAMES = ["sun3d", "rgbd", "mvs", "scenes11"]
    for dataset_name in DATASET_NAMES:
        if not os.path.exists(os.path.join(path, "{:}_train.tgz".format(dataset_name))):
            print "Downloading {:} dataset...".format(dataset_name)
            subprocess.call(
                    "cd {:} ;".format(path) +
                    "wget https://lmb.informatik.uni-freiburg.de/data/demon/traindata/{:}_train.tgz ;".format(dataset_name) +
                    "tar -xvzf {:}_train.tgz ;".format(dataset_name),
                    shell = True
                )

    print "Converting DeMoN dataset into the format required by DeepMVS..."
    SUB_DATASET_NAMES = ([
        "mvs_achteck_turm", "mvs_breisach", "mvs_citywall", 
        "rgbd_10_to_20_3d_train", "rgbd_10_to_20_handheld_train", "rgbd_10_to_20_simple_train", "rgbd_20_to_inf_3d_train", "rgbd_20_to_inf_handheld_train", "rgbd_20_to_inf_simple_train",
        "scenes11_train", 
        "sun3d_train_0.01m_to_0.1m", "sun3d_train_0.1m_to_0.2m", "sun3d_train_0.2m_to_0.4m", "sun3d_train_0.4m_to_0.8m", "sun3d_train_0.8m_to_1.6m", "sun3d_train_1.6m_to_infm"
    ])
    for dataset_name in SUB_DATASET_NAMES:
        print "Converting {:}.h5 ...".format(dataset_name)
        if not os.path.isdir(os.path.join(path, dataset_name)):
            os.mkdir(os.path.join(path, dataset_name))
        file = h5py.File(os.path.join(path, "{:}.h5".format(dataset_name)), "r")
        
        num_images = []
        for (seq_idx, seq_name) in enumerate(file):
            print "Processing sequence {:d}/{:d}".format(seq_idx, len(file))
            if not os.path.isdir(os.path.join(path, dataset_name, "{:04d}".format(seq_idx))):
                os.mkdir(os.path.join(path, dataset_name, "{:04d}".format(seq_idx)))
            if not os.path.isdir(os.path.join(path, dataset_name, "{:04d}".format(seq_idx), "images")):
                os.mkdir(os.path.join(path, dataset_name, "{:04d}".format(seq_idx), "images"))
            if not os.path.isdir(os.path.join(path, dataset_name, "{:04d}".format(seq_idx), "depths")):
                os.mkdir(os.path.join(path, dataset_name, "{:04d}".format(seq_idx), "depths"))
            if not os.path.isdir(os.path.join(path, dataset_name, "{:04d}".format(seq_idx), "poses")):
                os.mkdir(os.path.join(path, dataset_name, "{:04d}".format(seq_idx), "poses"))
            sequence = file[seq_name]["frames"]["t0"]
            num_images.append(len(sequence))
            for (f_idx, f_name) in enumerate(sequence):
                frame = sequence[f_name]
                for dt_type in frame:
                    dataset = frame[dt_type]
                    img = dataset[...]
                    if dt_type == "camera":
                        camera = ({
                            "extrinsic": [[img[5],img[8],img[11],img[14]], [img[6],img[9],img[12],img[15]], [img[7],img[10],img[13],img[16]], [0.0,0.0,0.0,1.0]],
                            "f_x": img[0],
                            "f_y": img[1],
                            "c_x": img[3],
                            "c_y": img[4]
                        })
                        with open(os.path.join(path, dataset_name, "{:04d}".format(seq_idx), "poses", "{:04d}.json".format(f_idx)), "w") as output_file:
                            json.dump(camera, output_file)
                    elif dt_type == "depth":
                        dimension = dataset.attrs["extents"]
                        depth = np.array(np.frombuffer(decompress(img.tobytes(), dimension[0] * dimension[1] * 2), dtype = np.float16)).astype(np.float32)
                        depth = depth.reshape(dimension[0], dimension[1])
                        imageio.imwrite(os.path.join(path, dataset_name, "{:04d}".format(seq_idx), "depths", "{:04d}.exr".format(f_idx)), depth, flags = freeimage.IO_FLAGS.EXR_ZIP)
                    elif dt_type == "image":
                        try:
                            img = imageio.imread(img.tobytes(), format = "RAW-FI")
                        except:
                            img = imageio.imread(img.tobytes())
                        imageio.imwrite(os.path.join(path, dataset_name, "{:04d}".format(seq_idx), "images", "{:04d}.png".format(f_idx)), img)
        with open(os.path.join(path, dataset_name, "num_images.json"), "w") as output_file:
            json.dump(num_images, output_file)

def download_GTAV_datasets(path):
    if not os.path.exists(os.path.join(path, "GTAV_720.tar.gz")):
        print "Downloading GTAV_720 dataset..."
        subprocess.call(
                "cd {:} ;".format(path) +
                "wget -O GTAV_720.tar.gz https://filebox.ece.vt.edu/~jbhuang/project/deepmvs/mvs-syn/GTAV_720.tar.gz ;" +
                "tar -xvzf GTAV_720.tar.gz ;",
                shell = True
            )
    if not os.path.exists(os.path.join(path, "GTAV_540.tar.gz")):
        print "Downloading GTAV_540 dataset..."
        subprocess.call(
                "cd {:} ;".format(path) + 
                "wget -O GTAV_540.tar.gz https://filebox.ece.vt.edu/~jbhuang/project/deepmvs/mvs-syn/GTAV_540.tar.gz ;" +
                "tar -xvzf GTAV_540.tar.gz ;",
                shell = True
            )

def download_training_datasets(path = None):
    if path is None:
        ROOT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
        if not os.path.isdir(os.path.join(ROOT_DIR, "dataset")):
            os.mkdir(os.path.join(ROOT_DIR, "dataset"))
        if not os.path.isdir(os.path.join(ROOT_DIR, "dataset", "train")):
            os.mkdir(os.path.join(ROOT_DIR, "dataset", "train"))
        download_demon_datasets(os.path.join(ROOT_DIR, "dataset", "train"))
        download_GTAV_datasets(os.path.join(ROOT_DIR, "dataset", "train"))
    else:
        download_demon_datasets(path)
        download_GTAV_datasets(path)
    print "Finished downloading training datasets."

if __name__ == "__main__":
    download_training_datasets()
