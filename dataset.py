import os
from glob import glob

import torch
import torch.utils.data
from PIL import Image
from torchvision import transforms


class MVTecDataset(torch.utils.data.Dataset):
    def __init__(self, root, category, input_size, is_train=True, return_filename=False):
        self.image_transform = transforms.Compose(
            [
                transforms.Resize(input_size),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )
        if is_train:
            self.image_files = glob(
                os.path.join(root, category, "train", "good", "*.png")
            ) + glob(os.path.join(root, category, "train", "good", "*.jpg"))
        else:
            self.image_files = glob(
                os.path.join(root, category, "test", "*", "*.png")
            ) + glob(os.path.join(root, category, "test", "*", "*.jpg"))
            self.target_transform = transforms.Compose(
                [
                    transforms.Resize(input_size),
                    transforms.ToTensor(),
                ]
            )
        self.is_train = is_train
        self.return_filename = return_filename

    def __getitem__(self, index):
        image_file = self.image_files[index]
        image = Image.open(image_file).convert("RGB")
        image = self.image_transform(image)
        if self.is_train:
            if self.return_filename:
                return image, os.path.basename(image_file)
            return image
        else:
            if os.path.dirname(image_file).endswith("good"):
                target = torch.zeros([1, image.shape[-2], image.shape[-1]])
            else:
                test_token = f"{os.sep}test{os.sep}"
                gt_token = f"{os.sep}ground_truth{os.sep}"
                gt_image_file = image_file.replace(test_token, gt_token)
                target = Image.open(gt_image_file).convert("L")
                target = self.target_transform(target)
            if self.return_filename:
                return image, target, os.path.basename(image_file)
            return image, target

    def __len__(self):
        return len(self.image_files)
