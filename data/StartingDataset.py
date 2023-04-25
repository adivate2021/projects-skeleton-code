import torch
import pandas as pd
import os
import cv2
from torchvision import transforms

wow = pd.read_csv("train.csv")
print(wow)
print(wow.groupby(["label"]).count())

class StartingDataset(torch.utils.data.Dataset):
    """
    Dataset that contains 100000 3x224x224 black images (all zeros).
    """

    def __init__(self):
        data_csv = pd.read_csv("/cassava-leaf-disease-classification/train.csv")
        
        self.image_ids = data_csv["image_id"]
        self.labels = data_csv["label"]



    def __getitem__(self, index):
        path_image = "~/Desktop/cassava-leaf-disease-classification/train_images" #path to the training image folder

        image_id = self.image_ids[index]
        label = self.labels[index]
        both = (image_id, label)
        image = cv2.imread(os.path.join(path_image, image_id))
        image_array = torch.Tensor(image)

        return image_array

    def __len__(self):
        return len(self.image_ids)
