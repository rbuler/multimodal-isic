import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as T

class DermDataset(Dataset):
    def __init__(self, df, radiomics, transform=None, is_train=True):
        self.df = df.reset_index(drop=True)
        self.transform = transform
        self.is_train = is_train

        self.artifact_cols = ['hair', 'ruler_marks', 'bubbles', 'vignette', 'frame', 'other']

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        img = Image.open(row['img_path']).convert('RGB')
        if self.transform:
            img = self.transform(img)
        else:
            img = T.ToTensor()(img)

        age = torch.tensor(row['age_normalized'], dtype=torch.float)
        sex = torch.tensor(row['sex_encoded'], dtype=torch.long)
        loc = torch.tensor(row['loc_encoded'], dtype=torch.long)
        artifacts = torch.tensor(row[self.artifact_cols].values, dtype=torch.long)


        # radiomic_features = 


        target = torch.tensor(row['dx'], dtype=torch.long)
        
        return {
            'target': target,
            'image': img,
            'age': age,
            'sex': sex,
            'loc': loc,
            'artifacts': artifacts}