import os
import sys
import torch
import numpy as np
import pandas as pd
import albumentations as A
from dataset import DermDataset
from sklearn.decomposition import PCA
from torch.utils.data import DataLoader
from albumentations.pytorch import ToTensorV2


def extract_latents(config, path, remove_background=False):

    if not hasattr(np, 'float'):
        np.float = float
    sys.path.append("ConvMAE")
    from ConvMAE.models_convmae import convmae_convvit_base_patch16_dec512d8b

    root = os.getcwd()

    device = torch.device(config['device'] if torch.cuda.is_available() else 'cpu')
    df_train_val = pd.read_pickle(config['dir']['df'])
    df_test = pd.read_pickle(config['dir']['df_test'])

    transform = A.Compose([
        A.Resize(224,224),
        A.Normalize(mean=(0.485,0.456,0.406), std=(0.229,0.224,0.225)),
        ToTensorV2(),
    ])

    seed = config['seed']
    np.random.seed(seed)
    torch.manual_seed(seed)

    train_val_dataset = DermDataset(df_train_val, radiomics=None, transform=transform)
    test_dataset = DermDataset(df_test, radiomics=None, transform=transform)

    train_val_loader = DataLoader(train_val_dataset, batch_size=1000, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False)

    ae_model = convmae_convvit_base_patch16_dec512d8b(with_decoder=False)
    ae_model = ae_model.to(device)

    model_name = path

    checkpoint_path = os.path.join(root, "models", model_name)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    ae_model.load_state_dict(checkpoint, strict=False)

    ae_model.eval()

    with torch.no_grad():
        def extract_latents(loader):
            pooled_list = []
            raw_list = []
            for batch in loader:
                images = batch['image'].to(device)
                image_path, segmentation_path = batch['image_path'], batch['segmentation_path']
                latent, _, ids_restore = ae_model(images, mask_ratio=0)

                latent_pooled_max = torch.max(latent, dim=1).values
                latent_pooled_mean = torch.mean(latent, dim=1)

                pooled_df = pd.DataFrame({
                    'image_path': image_path,
                    'segmentation_path': segmentation_path,
                    'target': batch['target'].numpy(),
                    'latent_pooled_max': list(latent_pooled_max.cpu().numpy()),
                    'latent_pooled_mean': list(latent_pooled_mean.cpu().numpy()),
                    'ids_restore': list(ids_restore.cpu().numpy())
                })
                pooled_list.append(pooled_df)

                # skin lesion mask (supports (B, H, W) or (B, 1, H, W))
                mask = batch['mask'].to(device)
                if mask.dim() == 3:
                    mask = mask.unsqueeze(1)  # (B,1,H,W)

                # divide image into 16x16 patches and find which patches overlap with lesion mask
                B, _, H, W = mask.shape
                patch_size = 16
                mask_patches = mask.unfold(2, patch_size, patch_size).unfold(3, patch_size, patch_size)  # (B,1,H/16,W/16,16,16)
                mask_patches = mask_patches.contiguous().view(B, 1, H // patch_size, W // patch_size, -1)  # (B,1,H/16,W/16,256)
                mask_patches = mask_patches.sum(dim=-1)  # (B,1,H/16,W/16)
                mask_patches = (mask_patches > 0).squeeze(1)  # (B,H/16,W/16), bool tensor indicating if patch overlaps with lesion

                raw_df = pd.DataFrame({
                    'image_path': image_path,
                    'segmentation_path': segmentation_path,
                    'target': batch['target'].numpy(),
                    'latent': list(latent.cpu().numpy()),
                    'ids_restore': list(ids_restore.cpu().numpy()),
                    'lesion_mask_patches': list(mask_patches.cpu().numpy())
                })
                raw_list.append(raw_df)

            pooled_df_all = pd.concat(pooled_list, ignore_index=True) if len(pooled_list) > 0 else pd.DataFrame()
            raw_df_all = pd.concat(raw_list, ignore_index=True) if len(raw_list) > 0 else pd.DataFrame()
            return pooled_df_all, raw_df_all

        # extract for train/val and test
        latent_pooled_train, latent_raw_train = extract_latents(train_val_loader)
        latent_pooled_test, latent_raw_test = extract_latents(test_loader)

    # build patch-level latents from raw df
    def build_patch_level_df(latent_raw_df, remove=True):
        patch_level_latents = []
        count = 0
        for idx, row in latent_raw_df.iterrows():
            image_path = row['image_path']
            segmentation_path = row['segmentation_path']
            target = row['target']
            latent = row['latent']  # shape (num_patches, dim)
            ids_restore = row['ids_restore']  # shape (num_patches,)
            mask_patches = row['lesion_mask_patches']  # shape (H/16, W/16) or flattened

            mask_flat = np.asarray(mask_patches).ravel()  # (num_patches,)
            for patch_idx in range(len(latent)):
                patch_latent = np.asarray(latent[patch_idx])  # (dim,)
                patch_id = int(ids_restore[patch_idx])

                # determine if this patch overlaps the lesion mask (safe check for out-of-bounds)
                inside = False
                if patch_id < mask_flat.size:
                    inside = bool(mask_flat[patch_id])
                    patch_in_mask = int(inside)  # 1 if inside, 0 if not

                if remove:
                    if inside:
                        patch_level_latents.append({
                            'image_path': image_path,
                            'segmentation_path': segmentation_path,
                            'target': target,
                            'patch_id': patch_id,
                            'patch_latent': patch_latent,
                            'patch_in_mask': patch_in_mask
                        })
                        count += 1
                else:
                    patch_level_latents.append({
                        'image_path': image_path,
                        'segmentation_path': segmentation_path,
                        'target': target,
                        'patch_id': patch_id,
                        'patch_latent': patch_latent,
                        'patch_in_mask': patch_in_mask
                    })
        df = pd.DataFrame(patch_level_latents)
        return df, count


    patch_level_train_df, train_count = build_patch_level_df(latent_raw_train, remove=remove_background)
    patch_level_test_df, test_count = build_patch_level_df(latent_raw_test, remove=remove_background)

    print(f"Total lesion-overlapping patches (train_val): {train_count}")
    print(f"Total lesion-overlapping patches (test): {test_count}")

    if len(patch_level_train_df) > 0:
        X_patches_train = np.vstack(patch_level_train_df["patch_latent"].values)
        pca = PCA(n_components=0.90, whiten=False)
        X_pca_train = pca.fit_transform(X_patches_train)
        print(f"PCA reduced dimensions from {X_patches_train.shape[1]} to {X_pca_train.shape[1]}")
        patch_level_train_df['patch_latent_pca'] = list(X_pca_train)
    else:
        patch_level_train_df['patch_latent_pca'] = []

    if len(patch_level_test_df) > 0:
        if len(patch_level_train_df) == 0:
            raise RuntimeError("No train patches to fit PCA. Cannot transform test patches.")
        X_patches_test = np.vstack(patch_level_test_df["patch_latent"].values)
        X_pca_test = pca.transform(X_patches_test)
        patch_level_test_df['patch_latent_pca'] = list(X_pca_test)
    else:
        patch_level_test_df['patch_latent_pca'] = []

    # SAVE DATAFRAMES ------------------------------------------------------
    save_files = True
    folder = 'dataframes_latents' 
    if not os.path.exists(folder):
        os.makedirs(folder)

    if save_files:
        patch_level_train_df.to_pickle(f"{folder}/patch_level_latents_train_df.pkl")
        patch_level_test_df.to_pickle(f"{folder}/patch_level_latents_test_df.pkl")
        latent_pooled_train.to_pickle(f"{folder}/latent_pooled_train_df.pkl")
        latent_pooled_test.to_pickle(f"{folder}/latent_pooled_test_df.pkl")
        latent_raw_train.to_pickle(f"{folder}/latent_raw_train_df.pkl")
        latent_raw_test.to_pickle(f"{folder}/latent_raw_test_df.pkl")

    print("Finished saving train_val and test patch-level and pooled latents.")

    return (patch_level_train_df, patch_level_test_df,
            latent_pooled_train, latent_pooled_test,
            latent_raw_train, latent_raw_test)