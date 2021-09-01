import pandas as pd
import numpy as np
from torch.utils.data import Dataset
from PIL import Image
import torch
import os


class CheXpertDataset(Dataset):
    def __init__(
            self,
            path_to_images,
            fold,
            include_uncertainty=False,
            transform=None,
            sample=0,
            finding="any",
            starter_images=False):

        self.transform = transform
        self.path_to_images = os.path.dirname(path_to_images)
        self.df = pd.read_csv(f"{path_to_images}/{fold}.csv")
        self.df['No Finding'].fillna(0, inplace=True)

        if starter_images:
            starter_images = pd.read_csv("starter_images.csv")
            self.df = pd.merge(left=self.df,right=starter_images, how="inner",on="Image Index")

        if not include_uncertainty:
            self.df = self.remove_unknown(self.df)
        
        self.targets = self.df['No Finding'].values
            
        # can limit to sample, useful for testing
        # if fold == "train" or fold =="val": sample=500
        if(sample > 0 and sample < len(self.df)):
            self.df = self.df.sample(sample)

        if not finding == "any":  # can filter for positive findings of the kind described; useful for evaluation
            if finding in self.df.columns:
                if len(self.df[self.df[finding] == 1]) > 0:
                    self.df = self.df[self.df[finding] == 1]
                else:
                    print(f"No positive cases exist for {finding}, returning all unfiltered cases")
            else:
                print("cannot filter on finding " + finding +
                      " as not in data - please check spelling")

        self.df = self.df.set_index("Path")
        self.PRED_LABEL = [
            'No Finding',
            'Enlarged Cardiomediastinum',
            'Cardiomegaly',
            'Lung Opacity',
            'Lung Lesion',
            'Edema',
            'Consolidation',
            'Pneumonia',
            'Atelectasis',
            'Pneumothorax',
            'Pleural Effusion',
            'Pleural Other',
            'Fracture',
            'Support Devices']

        # RESULT_PATH = "results/"

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):

        image = Image.open(os.path.join(self.path_to_images, self.df.index[idx]))
        # image = image.convert('RGB')

        labels = np.zeros(len(self.PRED_LABEL), dtype=int)
        for i in range(0, len(self.PRED_LABEL)):
            # can leave zero if zero, else make one
            if self.df[self.PRED_LABEL[i].strip()].iloc[idx].astype('int') >= -1:
                labels[i] = self.df[self.PRED_LABEL[i].strip()].iloc[idx].astype('int')

        label = np.array(labels[0])   # only No finding

        if self.transform:
            image = self.transform(image)
            label = torch.tensor(label)

        return image, label

    def remove_unknown(self, df):
        anomalies = ['Enlarged Cardiomediastinum', 'Cardiomegaly', 'Lung Opacity',
                     'Lung Lesion', 'Edema', 'Consolidation', 'Pneumonia', 'Atelectasis',
                     'Pneumothorax', 'Pleural Effusion', 'Pleural Other', 'Fracture']

        idxs = ~((df[anomalies] == -1).any(axis=1))
        new_df = df[idxs]
        return new_df
