import cv2
import torch
import numpy as np
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as T

class DermDataset(Dataset):
    def __init__(self, df, radiomics, transform=None, is_train=True, crop_size=450):
        self.df = df
        self.radiomics = radiomics
        self.transform = transform
        self.is_train = is_train
        self.crop_size = crop_size
        self.artifact_cols = ['hair', 'ruler_marks', 'bubbles', 'vignette', 'frame', 'other']

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        radiomic_features = self.radiomics.iloc[idx]

        cropped_image, cropped_mask = self.preprocess_image_and_mask(
            row['image_path'], row['segmentation_path'], self.crop_size
        )

        if self.transform:
            cropped_image = np.array(cropped_image)
            augmented = self.transform(image=cropped_image)
            cropped_image = augmented['image']
        else:
            cropped_image = T.ToTensor()(cropped_image)
        
        age = torch.tensor(row['age_normalized'], dtype=torch.float)
        sex = torch.tensor(row['sex_encoded'], dtype=torch.long)
        loc = torch.tensor(row['loc_encoded'], dtype=torch.long)
        artifacts = torch.tensor(row[self.artifact_cols].values.astype(int), dtype=torch.long)
        radiomic_features = torch.tensor(radiomic_features.values, dtype=torch.float)
        target = torch.tensor(row['dx'], dtype=torch.long)

        return {
            'image': cropped_image,
            'mask': T.ToTensor()(cropped_mask),
            'radiomics': radiomic_features,
            'age': age,
            'sex': sex,
            'loc': loc,
            'artifacts': artifacts,
            'target': target
        }

    def crop_centered_on_mask(self, image, mask, crop_size=450):
        h, w = image.shape[:2]

        if mask.shape[:2] != (h, w):
            mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)

        ys, xs = np.where(mask > 0)

        if len(xs) == 0 or len(ys) == 0:
            cx, cy = w // 2, h // 2
        else:
            cx = int(np.mean(xs))
            cy = int(np.mean(ys))

        half_crop = crop_size // 2
        x1 = max(cx - half_crop, 0)
        y1 = max(cy - half_crop, 0)

        x1 = min(x1, w - crop_size)
        y1 = min(y1, h - crop_size)

        x2 = x1 + crop_size
        y2 = y1 + crop_size

        cropped_image = image[y1:y2, x1:x2]
        cropped_mask = mask[y1:y2, x1:x2]

        return cropped_image, cropped_mask

    def preprocess_image_and_mask(self, image_path, mask_path, crop_size=450):
        image = cv2.imread(image_path)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

        cropped_image, cropped_mask = self.crop_centered_on_mask(image, mask, crop_size)

        cropped_image = Image.fromarray(cv2.cvtColor(cropped_image, cv2.COLOR_BGR2RGB))
        cropped_mask = Image.fromarray(cropped_mask)

        return cropped_image, cropped_mask