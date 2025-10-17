import os
import io
import gc
import torch
import typing
import neptune
import umap
import argparse
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.preprocessing import normalize


def concat_patch_moments(latent, eps=1e-6, unbiased=False):

    mean = latent.mean(dim=1)
    maxv = latent.max(dim=1).values
    std = latent.std(dim=1, unbiased=unbiased)
    median = latent.median(dim=1).values
    centered = latent - mean.unsqueeze(1)
    m3 = (centered ** 3).mean(dim=1)
    m4 = (centered ** 4).mean(dim=1)

    sigma = std.clamp(min=eps)
    skew = m3 / (sigma ** 3)
    kurtosis = m4 / (sigma ** 4) - 3.0
    feats = torch.cat([mean, maxv, std, median, skew, kurtosis], dim=1)  # (B, 6*D)
    
    return feats


def visualize_latent_space(config, run, seed, num_epochs, epoch, latent_feats_list, target_list, balance_classes=False):
    if epoch % 10 == 0 or epoch == num_epochs - 1:
        latent_feats_all = torch.cat(latent_feats_list, dim=0).numpy()  # (N_images, 6*D)
        targets_all = torch.cat(target_list, dim=0).numpy()

            # optional: balanced sampling
        if balance_classes:
            counts = np.bincount(targets_all)
            num_samples_per_class = min(100, int(counts.min()))
            selected_indices = []
            for cls in np.unique(targets_all):
                cls_indices = np.where(targets_all == cls)[0]
                if len(cls_indices) > num_samples_per_class:
                    sel = np.random.choice(cls_indices, num_samples_per_class, replace=False)
                else:
                    sel = cls_indices
                selected_indices.extend(sel)
            selected_indices = np.array(selected_indices, dtype=int)

            latent_feats_sel = latent_feats_all[selected_indices]
            targets_sel = targets_all[selected_indices]
        else:
            latent_feats_sel = latent_feats_all
            targets_sel = targets_all

        scaler = StandardScaler()
        latent_scaled = scaler.fit_transform(latent_feats_sel)  # shape (M, 6*D)

            # PCA to 512 (but no more than n_samples and no more than n_features)
        n_samples, n_features = latent_scaled.shape
        pca_n_components = min(512, n_samples, n_features)
        pca = PCA(n_components=pca_n_components, random_state=seed)
        latent_pca = pca.fit_transform(latent_scaled)  # (M, pca_n_components)

        latent_pca = normalize(latent_pca, norm='l2')

            # UMAP on PCA output
        reducer = umap.UMAP(n_components=2, random_state=seed)
        emb = reducer.fit_transform(latent_pca)  # (M, 2)

        fig, ax = plt.subplots(figsize=(6, 6))
        unique_labels = np.unique(targets_sel)
        cmap = plt.get_cmap('tab10')
        for i, lbl in enumerate(unique_labels):
            mask = targets_sel == lbl
            ax.scatter(emb[mask, 0], emb[mask, 1], s=5, color=cmap(i % 10), label=str(int(lbl)), alpha=0.8)
        ax.set_title(f"MomentsConcat PCA{pca_n_components} UMAP (epoch {epoch})")
        ax.axis('off')
        ax.legend(title='class', markerscale=3, fontsize='small', bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()

        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=150)
        buf.seek(0)
        if config.get("neptune"):
            run[f"visuals/umap/moments_concat"].append(neptune.types.File.from_content(buf.getvalue(), extension='png'))
            run.sync()
        buf.close()
        plt.close(fig)

        del latent_feats_all, latent_feats_sel, latent_scaled, latent_pca, emb
        del reducer, pca, scaler
        gc.collect()


def visualize_model_outputs(run, device, val_loader, ae_model, mask_ratio, num_epochs, epoch):
    if epoch % 10 == 0 or epoch == num_epochs - 1:
        with torch.no_grad():
            batch = next(iter(val_loader))
            for i in range(4):
                img = batch['image'][i:i+1].to(device)
                loss, pred, mask = ae_model(img, mask_ratio=mask_ratio)
                recon = ae_model.unpatchify(pred) if hasattr(ae_model, 'unpatchify') else pred
                img_vis = img.squeeze().cpu().numpy().transpose(1, 2, 0)
                recon_vis = recon.squeeze().cpu().numpy().transpose(1, 2, 0)
                mask_vis = mask.cpu().numpy()
                    
                image_patches = ae_model.patchify(img).cpu().numpy() if hasattr(ae_model, 'patchify') else img.cpu().numpy()
                mask_expanded = mask_vis[..., None]
                unmasked_patches = image_patches * (1 - mask_expanded)
                binary_patches = mask_expanded * np.ones_like(image_patches)
                binary_image = ae_model.unpatchify(torch.tensor(binary_patches, device=device)).cpu().numpy()

                    
                img_vis = img_vis * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
                recon_vis = recon_vis * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
                    
                binary_image_vis = binary_image.squeeze().transpose(1, 2, 0)
                overlay_vis = recon_vis * binary_image_vis + img_vis * (1 - binary_image_vis)
                
                img_vis = np.clip(img_vis, 0, 1)
                binary_image_vis = np.clip(binary_image_vis, 0, 1)
                recon_vis = np.clip(recon_vis, 0, 1)
                overlay_vis = np.clip(overlay_vis, 0, 1)

                fig, axs = plt.subplots(1, 4, figsize=(16, 4))
                axs[0].imshow(img_vis)
                axs[0].set_title("Original")
                axs[0].axis('off')
                axs[1].imshow(binary_image_vis, cmap='gray')
                axs[1].set_title("Mask")
                axs[1].axis('off')
                axs[2].imshow(recon_vis)
                axs[2].set_title("Reconstruction")
                axs[2].axis('off')
                axs[3].imshow(overlay_vis)
                axs[3].set_title("Overlay")
                axs[3].axis('off')

                plt.tight_layout()
                buf = io.BytesIO()
                plt.savefig(buf, format='png')
                buf.seek(0)
                run[f"visuals/image_comparison_{i+1}"].append(neptune.types.File.from_content(buf.getvalue(), extension='png'))
                plt.close(fig)


def get_args_parser(path: typing.Union[str, bytes, os.PathLike]):
    help = '''path to .yml config file
    specyfying datasets/training params'''

    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", type=str,
                        default=path,
                        help=help)
    return parser