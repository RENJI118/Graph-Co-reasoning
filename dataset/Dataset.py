import numpy as np
import cv2
import random


import torch
import torch.utils.data
from torchvision import datasets, models, transforms




# def get_datasets(seed, debug, no_seg=False, on="train", full=False,
#                  fold_number=0, normalisation="minmax"):
#     base_folder = pathlib.Path(get_brats_folder(on)).resolve()
#     print(base_folder)
#     import os
#     print(os.path.exists(str(base_folder)))
#     assert base_folder.exists()
#     patients_dir = sorted([x for x in base_folder.iterdir() if x.is_dir()])
#     kfold = KFold(5, shuffle=True, random_state=seed)
#     splits = list(kfold.split(patients_dir))
#     train_idx, val_idx = splits[fold_number]
#     print("first idx of train", train_idx[0])
#     print("first idx of test", val_idx[0])
#     train = [patients_dir[i] for i in train_idx]
#     val = [patients_dir[i] for i in val_idx]
#     # return patients_dir
#     train_dataset = Brats(train, training=True,  debug=debug,
#                           normalisation=normalisation)
#     val_dataset = Brats(val, training=False, data_aug=False,  debug=debug,
#                         normalisation=normalisation)
#     bench_dataset = Brats(val, training=False, benchmarking=True, debug=debug,
#                           normalisation=normalisation)  
#     train_dataset = Dataset(args, train_img_paths, train_mask_paths, args.aug)
#     val_dataset = Dataset(args, val_img_paths, val_mask_paths)
    
#     return train_dataset, val_dataset




class Dataset(torch.utils.data.Dataset):

    def __init__(self, args, img_paths, mask_paths, aug=False):
        self.args = args
        self.img_paths = img_paths
        self.mask_paths = mask_paths
        self.aug = aug

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        mask_path = self.mask_paths[idx]
        #读numpy数据(npy)的代码
        npimage = np.load(img_path)
        npmask = np.load(mask_path)
        npimage = npimage.transpose((2, 0, 1))

        WT_Label = npmask.copy()
        WT_Label[npmask == 1] = 1.
        WT_Label[npmask == 2] = 1.
        WT_Label[npmask == 4] = 1.
        TC_Label = npmask.copy()
        TC_Label[npmask == 1] = 1.
        TC_Label[npmask == 2] = 0.
        TC_Label[npmask == 4] = 1.
        ET_Label = npmask.copy()
        ET_Label[npmask == 1] = 0.
        ET_Label[npmask == 2] = 0.
        ET_Label[npmask == 4] = 1.
        nplabel = np.empty((160, 160, 3))
        nplabel[:, :, 0] = WT_Label
        nplabel[:, :, 1] = TC_Label
        nplabel[:, :, 2] = ET_Label
        nplabel = nplabel.transpose((2, 0, 1))

        nplabel = nplabel.astype("float32")
        npimage = npimage.astype("float32")

        return npimage,nplabel
    
    def get_file_name(self, index):
        return self.img_paths[index]

