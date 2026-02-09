import os
from PIL import Image
import torchvision.transforms as transforms
import shutil
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets
from torchvision.datasets import CIFAR10, CIFAR100, MNIST, ImageNet
import numpy as np
import torch


class Cutout(object):

    def __init__(self, n_holes, length):
        self.n_holes = n_holes
        self.length = length

    def __call__(self, img):

        h = img.size(1)
        w = img.size(2)

        mask = np.ones((h, w), np.float32)

        for n in range(self.n_holes):
            y = np.random.randint(h)
            x = np.random.randint(w)

            y1 = np.clip(y - self.length // 2, 0, h)
            y2 = np.clip(y + self.length // 2, 0, h)
            x1 = np.clip(x - self.length // 2, 0, w)
            x2 = np.clip(x + self.length // 2, 0, w)

            mask[y1: y2, x1: x2] = 0.

        mask = torch.from_numpy(mask)
        mask = mask.expand_as(img)
        img = img * mask
        return img

def cifar10(cutout=True, download=True):
    aug = [transforms.RandomCrop(32, padding=4), transforms.RandomHorizontalFlip(),transforms.ToTensor()]
    if cutout:
        aug.append(Cutout(n_holes=1, length=16))
    transform_train = transforms.Compose(aug)
    transform_test = transforms.Compose([
        transforms.ToTensor(),
    ])
    train_dataset = CIFAR10(root='datasets/cifar10',
                            train=True, download=download, transform=transform_train)
    val_dataset = CIFAR10(root='datasets/cifar10',
                            train=False, download=download, transform=transform_test)
    norm = ((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    return train_dataset, val_dataset, norm


def cifar100(cutout=True, download=True):
    aug = [transforms.RandomCrop(32, padding=4), transforms.RandomHorizontalFlip(),transforms.ToTensor()]
    if cutout:
        aug.append(Cutout(n_holes=1, length=16))
    transform_train = transforms.Compose(aug)
    transform_test = transforms.Compose([
        transforms.ToTensor(),
    ])
    train_dataset = CIFAR100(root='dataset/cifar100/',
                                train=True, download=download, transform=transform_train)
    val_dataset = CIFAR100(root='dataset/cifar100/',
                            train=False, download=download, transform=transform_test)
    norm = ((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
    return train_dataset, val_dataset, norm

def mnist(download=True):
    aug = [transforms.RandomCrop(28, padding=4), transforms.RandomHorizontalFlip(),transforms.ToTensor()]
    transform_train = transforms.Compose(aug)
    transform_test = transforms.Compose([
        transforms.ToTensor(),
    ])
    train_dataset = MNIST(root='dataset/mnist',
                                train=True, download=download, transform=transform_train)
    val_dataset = MNIST(root='dataset/mnist',
                            train=False, download=download, transform=transform_test)
    norm = ((0), (1))
    return train_dataset, val_dataset, norm

class TinyImageNetVal(Dataset):

    def __init__(self, val_dir, class_to_idx, transform=None):
        super().__init__()
        self.transform = transform
        self.val_dir = val_dir
        self.class_to_idx = class_to_idx

        images_folder = os.path.join(val_dir, 'images')
        if os.path.isdir(images_folder):
            self._prepare_val(val_dir)

        self.images = []
        self.labels = []
        anno_file = os.path.join(val_dir, 'val_annotations.txt')
        with open(anno_file, 'r') as f:
            for line in f:
                img_name, cls_id = line.strip().split('	')[:2]
                img_path = os.path.join(val_dir, cls_id, img_name)
                self.images.append(img_path)
                self.labels.append(self.class_to_idx[cls_id])

    def _prepare_val(self, val_dir):
        images_folder = os.path.join(val_dir, 'images')
        anno_file = os.path.join(val_dir, 'val_annotations.txt')
        with open(anno_file, 'r') as f:
            for line in f:
                img_name, cls_id = line.strip().split('	')[:2]
                cls_folder = os.path.join(val_dir, cls_id)
                os.makedirs(cls_folder, exist_ok=True)
                src = os.path.join(images_folder, img_name)
                dst = os.path.join(cls_folder, img_name)
                if os.path.exists(src) and not os.path.exists(dst):
                    shutil.move(src, dst)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx]
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        label = self.labels[idx]
        return image, label



def tiny_imagenet(dataroot='tiny-imagenet-200'):
    image_size = 64
    norm = ((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))

    transform_train = transforms.Compose([
        transforms.RandomResizedCrop(image_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=norm[0], std=norm[1]),
    ])

    transform_test = transforms.Compose([
        transforms.Resize(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=norm[0], std=norm[1]),
    ])

    traindir = os.path.join(dataroot, 'train')
    valdir = os.path.join(dataroot, 'val')
    train_dataset = datasets.ImageFolder(traindir, transform_train)
    class_to_idx = train_dataset.class_to_idx
    val_dataset = TinyImageNetVal(valdir, class_to_idx=class_to_idx, transform=transform_test)

    return train_dataset, val_dataset, norm

