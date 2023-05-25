'''import torch
import pandas as pd
import os
import cv2
from torchvision import transforms

class StartingDataset(torch.utils.data.Dataset):
    """
    Dataset that contains 100000 3x224x224 black images (all zeros).
    """

    def __init__(self, training_bool):
        data_csv = pd.read_csv("~/Downloads/cassava-leaf-disease-classification/train.csv")
        
        self.image_ids = data_csv["image_id"]
        self.labels = data_csv["label"]
        if training_bool:
            self.image_ids = self.image_ids[:17118]
            self.labels = self.labels[:17118]
        else:
            self.image_ids = self.image_ids[17118:]
            self.labels = self.labels[17118:]



    def __getitem__(self, index):
        path_image = "~/Downloads/cassava-leaf-disease-classification/train_images" #path to the training image folder, change to whatever

        image_id = self.image_ids[index]
        label = self.labels[index]
        both = (image_id, label)
        image = cv2.imread(os.path.join(path_image, image_id))
        image_array = torch.Tensor(image)

        return (image_array, label)

    def __len__(self):
        return len(self.image_ids)

'''
import torch
import pandas as pd
from PIL import Image
from torchvision import transforms

class StartingDataset(torch.utils.data.Dataset):
    """
    Dataset that contains 100000 3x224x224 black images (all zeros).
    """

    def __init__(self, images_path, csv_path, train_or_test):
        self.images_path = images_path
        images_labels = pd.read_csv(csv_path)
        if train_or_test:
            self.image_ids = images_labels['image_id'][:17118]
            self.labels = images_labels['label'][:17118]
        else:
            self.image_ids = images_labels['image_id'][17118:]
            self.labels = images_labels['label'][17118:]

    def __getitem__(self, index):
        image_id = self.image_ids[index]
        label = self.labels[index]
        #both = (image_id, label)
        image_path = self.images_path+ "/" +image_id
        image = Image.open(image_path)
        transform = transforms.Compose([
            transforms.PILToTensor()
        ])
        return (transform(image), label)

    def __len__(self):
        return len(self.image_ids)