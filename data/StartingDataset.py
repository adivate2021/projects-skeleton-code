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

    def __init__(self, image_ids, labels):
        pd.read_csv("/cassava-leaf-disease-classification/train.csv")
        
        self.image_ids = image_ids
        self.labels = labels

        
        # inputs = torch.zeros([3, 224, 224])
        #label = 0
        # convert_tensor = transforms.ToTensor()
        # path_image = "~/Desktop/cassava-leaf-disease-classification/train_images"
        # inputs = []
        # for img in os.path(path_image):
        #     img_array = cv2.imread(os.path.join(path_image, img))
        #     new_array = cv2.resize(img_array, (224, 224))
        #     b,g,r = cv2.split(new_array)
        #     inputs.append([b,g,r])
        # print(inputs[0])



    def __getitem__(self, index):
        image_id = self.image_ids[index]
        label = self.labels[index]
        both = (image_id, label)
        
        return both

    def __len__(self):
        return len(self.image_ids)
