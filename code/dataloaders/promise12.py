import torch
from torch.utils.data import Dataset
import SimpleITK as sitk
from skimage.exposure import equalize_adapthist
import torchvision.transforms as transforms
import os
import numpy as np
import cv2
from scipy import ndimage
import random
from torch.utils.data.sampler import Sampler
import itertools
import torchvision

def color_jitter(image):
    if not torch.is_tensor(image):
        np_to_tensor = transforms.ToTensor()
        image = np_to_tensor(image)
    # s is the strength of color distortion.
    s = 1.0
    jitter = transforms.ColorJitter(0.8 * s, 0.8 * s, 0.8 * s, 0.2 * s)
    return jitter(image)

def random_rot_flip(image, label=None):
    k = np.random.randint(0, 4)  
    image = np.rot90(image, k)  
    axis = np.random.randint(0, 2)  
    image = np.flip(image, axis=axis).copy()  
    if label is not None:  
        label = np.rot90(label, k) 
        label = np.flip(label, axis=axis).copy() 
        return image, label
    else:
        return image

def random_rotate(image, label):
    angle = np.random.randint(-20, 20)  
    image = ndimage.rotate(image, angle, order=0, reshape=False) 
    label = ndimage.rotate(label, angle, order=0, reshape=False)  
    return image, label

def cutout_gray(img, mask, p=0.5, size_min=0.02, size_max=0.4, ratio_1=0.3, ratio_2=1/0.3, value_min=0, value_max=1, pixel_level=True):
    if random.random() < p:
        img = np.array(img)
        mask = np.array(mask)
        img_h, img_w = img.shape

        while True:
            size = np.random.uniform(size_min, size_max) * img_h * img_w
            ratio = np.random.uniform(ratio_1, ratio_2)
            erase_w = int(np.sqrt(size / ratio))
            erase_h = int(np.sqrt(size * ratio))
            x = np.random.randint(0, img_w)
            y = np.random.randint(0, img_h)
            if x + erase_w <= img_w and y + erase_h <= img_h:
                break

        if pixel_level:
            value = np.random.randint(value_min, value_max + 1, (erase_h, erase_w))
        else:
            value = np.random.randint(value_min, value_max + 1)

        img[y:y + erase_h, x:x + erase_w] = value
        mask[y:y + erase_h, x:x + erase_w] = 0

    return img, mask

class Promise12(Dataset):

    def __init__(self, data_dir, mode, out_size):
        # store data in the npy file
        self.out_size = out_size
        np_data_path = os.path.join(data_dir, 'npy_image')
        if not os.path.exists(np_data_path):
            os.makedirs(np_data_path)
            data_to_array(data_dir, np_data_path, self.out_size, self.out_size)
        else:
            print('read the data from: {}'.format(np_data_path))
        self.data_dir = data_dir
        self.sample_list = []
        self.mode = mode
        # read the data from npy
        if self.mode == 'train':
            self.X_train = np.load(os.path.join(np_data_path, 'X_train.npy'))
            self.y_train = np.load(os.path.join(np_data_path, 'y_train.npy'))
        elif self.mode == "val":
            with open(data_dir + "/val.list", "r") as f:
                self.sample_list = f.readlines()
            self.sample_list = [item.replace("\n", "") for item in self.sample_list]
          # self.X_val = np.load(os.path.join(np_data_path, 'X_val.npy'))
          # self.y_val = np.load(os.path.join(np_data_path, 'y_val.npy'))
        
    def __getitem__(self, i):

        np_data_path = os.path.join(self.data_dir, 'npy_image')
        if self.mode == 'train':
            img, mask = self.X_train[i], self.y_train[i]  # [224,224] [224,224]
            if random.random() > 0.5:  
                img, mask = random_rot_flip(img, mask)
            elif random.random() > 0.5:
                img, mask = random_rotate(img, mask)

            image_strong, label_strong = cutout_gray(img,mask,p=0.5)
            image_strong = color_jitter(img).type("torch.FloatTensor")
            img_tensor = torch.from_numpy(img).unsqueeze(0)
            mask_tensor = torch.from_numpy(mask)
            mask_strong = torch.from_numpy(label_strong)
            sample = {"image": img_tensor, 
                      "image_strong": image_strong,
                      "mask": mask_tensor, 
                      'mask_strong': mask_strong,
                      }

        elif self.mode == 'val':
            case = self.sample_list[i]
            img = np.load(os.path.join(np_data_path, '{}.npy'.format(case)))
            mask = np.load(os.path.join(np_data_path, '{}_segmentation.npy'.format(case)))
            img_tensor = torch.from_numpy(img)
            mask_tensor = torch.from_numpy(mask)
            sample = {"image": img_tensor, "mask": mask_tensor}
        return sample

    def __len__(self):
        if self.mode == 'train':
             return self.X_train.shape[0]
        elif self.mode == 'val':
             return len(self.sample_list)


def data_to_array(base_path, store_path, img_rows, img_cols):
    global min_val, max_val
    fileList = os.listdir(base_path)
    fileList = sorted((x for x in fileList if '.mhd' in x))

    val_list = [35, 36, 37, 38, 39]
    test_list = [40, 41, 42, 43, 44, 45, 46, 47, 48, 49]
    train_list = list(set(range(50)) - set(val_list) - set(test_list))

    for the_list in [train_list]:
        images = []
        masks = []

        filtered = [file for file in fileList for ff in the_list if str(ff).zfill(2) in file]

        for filename in filtered:

            itkimage = sitk.ReadImage(os.path.join(base_path, filename))
            imgs = sitk.GetArrayFromImage(itkimage)

            if 'segm' in filename.lower():
                imgs = img_resize(imgs, img_rows, img_cols, equalize=False)
                masks.append(imgs)
            else:
                imgs = img_resize(imgs, img_rows, img_cols, equalize=False)
                imgs_norm = np.zeros([len(imgs), img_rows, img_cols])
                for mm, img in enumerate(imgs):
                    min_val = np.min(img)  # Min-Max归一化
                    max_val = np.max(img)
                    imgs_norm[mm] = (img - min_val) / (max_val - min_val)
                images.append(imgs_norm)

        # images: slices x w x h ==> total number x w x h
        images = np.concatenate(images, axis=0).reshape(-1, img_rows, img_cols)  # (1250,256,256)
        masks = np.concatenate(masks, axis=0).reshape(-1, img_rows, img_cols)
        masks = masks.astype(np.uint8)

        # Smooth images using CurvatureFlow
        images = smooth_images(images)
        images = images.astype(np.float32)

        np.save(os.path.join(store_path, 'X_train.npy'), images)
        np.save(os.path.join(store_path, 'y_train.npy'), masks)
    for the_list in [val_list, test_list]:
        filtered = [file for file in fileList for ff in the_list if str(ff).zfill(2) in file]

        for filename in filtered:

            itkimage = sitk.ReadImage(os.path.join(base_path, filename))
            imgs = sitk.GetArrayFromImage(itkimage)

            if 'segm' in filename.lower():
                imgs = img_resize(imgs, img_rows, img_cols, equalize=False)
                imgs = imgs.astype(np.uint8)
                np.save(os.path.join(store_path, '{}.npy'.format(filename[:-4])), imgs)
            else:
                imgs = img_resize(imgs, img_rows, img_cols, equalize=False)
                imgs_norm = np.zeros([len(imgs), img_rows, img_cols])
                for mm, img in enumerate(imgs):
                    min_val = np.min(img)  # Min-Max归一化
                    max_val = np.max(img)
                    imgs_norm[mm] = (img - min_val) / (max_val - min_val)
                images = smooth_images(imgs_norm)
                images = images.astype(np.float32)
                np.save(os.path.join(store_path, '{}.npy'.format(filename[:-4])), images)



def img_resize(imgs, img_rows, img_cols, equalize=True):
    new_imgs = np.zeros([len(imgs), img_rows, img_cols])
    for mm, img in enumerate(imgs):
        if equalize:
            img = equalize_adapthist(img, clip_limit=0.05)
        new_imgs[mm] = cv2.resize(img, (img_rows, img_cols), interpolation=cv2.INTER_NEAREST)

    return new_imgs


def smooth_images(imgs, t_step=0.125, n_iter=5):
    """
    Curvature driven image denoising.
    In my experience helps significantly with segmentation.
    """

    for mm in range(len(imgs)):
        img = sitk.GetImageFromArray(imgs[mm])
        img = sitk.CurvatureFlow(image1=img,
                                 timeStep=t_step,
                                 numberOfIterations=n_iter)

        imgs[mm] = sitk.GetArrayFromImage(img)

    return imgs