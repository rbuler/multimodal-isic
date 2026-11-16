import argparse
import copy
import fcntl
import os
import random
import sys
import time
import uuid
from pathlib import Path

import albumentations as A
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from albumentations.pytorch import ToTensorV2
from medmnist import ChestMNIST, OCTMNIST, OrganAMNIST, PathMNIST
from torch.utils.data import DataLoader, WeightedRandomSampler

if (Path.cwd() / "ConvMAE").is_dir():
	CONVMAE_DIR = Path.cwd() / "ConvMAE"
elif "__file__" in globals():
	CONVMAE_DIR = Path(__file__).resolve().parents[1] / "ConvMAE"
else:
	raise RuntimeError("Run this notebook from the multimodal-isic repository root.")
sys.path.insert(0, str(CONVMAE_DIR))
from models_convmae import convmae_convvit_base_patch16_dec512d8b


DATA_ROOT = Path("/users/project1/pt01191/MMODAL_ISIC/Data/MEDMNIST")
SCRIPT_DIR = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd() / "zMEDMNIST"
OUTPUT_ROOT = Path("models")
IMAGE_SIZE = 224

DATASETS = {
	"OrganAMNIST": OrganAMNIST,
	"PathMNIST": PathMNIST,
	"ChestMNIST": ChestMNIST,
	"OCTMNIST": OCTMNIST,
}


def parse_args():
	parser = argparse.ArgumentParser(description="Train ConvMAE on MEDMNIST.")
	parser.add_argument("--dataset", choices=[*DATASETS, "all"], default="all")
	parser.add_argument("--data-root", type=Path, default=DATA_ROOT)
	parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
	parser.add_argument("--results-path", type=Path, default=SCRIPT_DIR / "medmnist_convmae_results.pkl")
	parser.add_argument("--checkpoint", type=Path, default=Path("ConvMAE/checkpoint.pth"))
	parser.add_argument("--epochs", type=int, default=700)
	parser.add_argument("--batch-size", type=int, default=256)
	parser.add_argument("--val-batch-size", type=int, default=256)
	parser.add_argument("--num-workers", type=int, default=8)
	parser.add_argument("--encoder-lr", type=float, default=1e-5)
	parser.add_argument("--decoder-lr", type=float, default=1e-3)
	parser.add_argument("--weight-decay", type=float, default=0.05)
	parser.add_argument("--mask-ratio", type=float, default=0.75)
	parser.add_argument("--norm-pix-loss", action="store_true")
	parser.add_argument("--seed", type=int, default=42)
	parser.add_argument("--device", default="cuda")
	parser.add_argument("--no-oversampling", action="store_true")
	parser.add_argument("--num-visualizations", type=int, default=4)
	return parser.parse_args()


def seed_everything(seed):
	random.seed(seed)
	np.random.seed(seed)
	torch.manual_seed(seed)
	if torch.cuda.is_available():
		torch.cuda.manual_seed_all(seed)


def make_transforms():
	train_transform = A.Compose([
		A.RandomResizedCrop(size=(IMAGE_SIZE, IMAGE_SIZE), scale=(0.5, 1.0), ratio=(0.75, 1.33), p=1.0),
		A.HorizontalFlip(p=0.5),
		A.VerticalFlip(p=0.5),
		A.RandomRotate90(p=0.5),
		A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
		ToTensorV2(),
	])
	valid_transform = A.Compose([
		A.Resize(IMAGE_SIZE, IMAGE_SIZE),
		A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
		ToTensorV2(),
	])
	return train_transform, valid_transform


def make_sample_weights(labels):
	labels = np.asarray(labels)
	if labels.ndim == 1 or labels.shape[1] == 1:
		labels = labels.reshape(-1)
		counts = np.bincount(labels.astype(int))
		inverse_counts = np.zeros_like(counts, dtype=np.float64)
		nonzero = counts > 0
		inverse_counts[nonzero] = 1.0 / counts[nonzero]
		return inverse_counts[labels.astype(int)]

	# ChestMNIST is multi-label. Weight each image by the rarest positive label.
	positive_counts = labels.sum(axis=0)
	inverse_counts = np.zeros_like(positive_counts, dtype=np.float64)
	nonzero = positive_counts > 0
	inverse_counts[nonzero] = 1.0 / positive_counts[nonzero]
	weights = (labels * inverse_counts).sum(axis=1)
	weights[weights == 0] = inverse_counts[nonzero].mean()
	return weights


def build_loader(dataset, batch_size, num_workers, sampler=None, seed=0):
	generator = torch.Generator()
	generator.manual_seed(seed)
	return DataLoader(
		dataset,
		batch_size=batch_size,
		shuffle=sampler is None,
		sampler=sampler,
		num_workers=num_workers,
		pin_memory=torch.cuda.is_available(),
		persistent_workers=num_workers > 0,
		generator=generator,
	)


def load_model(checkpoint_path, device, norm_pix_loss):
	model = convmae_convvit_base_patch16_dec512d8b(norm_pix_loss=norm_pix_loss)
	checkpoint = torch.load(checkpoint_path, map_location=device)
	state_dict = checkpoint.get("model", checkpoint) if isinstance(checkpoint, dict) else checkpoint
	missing, unexpected = model.load_state_dict(state_dict, strict=False)
	if missing:
		print(f"Checkpoint missing {len(missing)} parameters")
	if unexpected:
		print(f"Checkpoint has {len(unexpected)} unexpected parameters")
	return model.to(device)


def run_epoch(model, loader, device, mask_ratio, optimizer=None, scaler=None):
	training = optimizer is not None
	model.train(training)
	total_loss = 0.0
	total_samples = 0

	for images, _ in loader:
		images = images.to(device, non_blocking=True)
		if training:
			optimizer.zero_grad(set_to_none=True)

		with torch.set_grad_enabled(training):
			with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
				loss, _, _ = model(images, mask_ratio=mask_ratio)
			if training:
				if scaler is not None and scaler.is_enabled():
					scaler.scale(loss).backward()
					scaler.step(optimizer)
					scaler.update()
				else:
					loss.backward()
					optimizer.step()

		if not torch.isfinite(loss):
			raise RuntimeError(f"Non-finite loss encountered: {loss.item()}")
		total_loss += loss.item() * images.size(0)
		total_samples += images.size(0)

	return total_loss / total_samples


def append_result(results_path, result):
	results_path.parent.mkdir(parents=True, exist_ok=True)
	lock_path = results_path.with_suffix(results_path.suffix + ".lock")
	temporary_path = results_path.with_suffix(results_path.suffix + f".{result['run_id']}.tmp")
	with lock_path.open("w") as lock_file:
		fcntl.flock(lock_file, fcntl.LOCK_EX)
		try:
			if results_path.exists():
				results_df = pd.read_pickle(results_path)
			else:
				results_df = pd.DataFrame()
			results_df = pd.concat([results_df, pd.DataFrame([result])], ignore_index=True)
			results_df.to_pickle(temporary_path)
			os.replace(temporary_path, results_path)
		finally:
			fcntl.flock(lock_file, fcntl.LOCK_UN)


def save_reconstructions(model, dataset, sample_indices, device, mask_ratio,
						 output_dir, epoch, norm_pix_loss):
	output_dir.mkdir(parents=True, exist_ok=True)
	model.eval()
	images = torch.stack([dataset[index][0] for index in sample_indices]).to(device)
	with torch.no_grad():
		_, pred, mask = model(images, mask_ratio=mask_ratio)

	if norm_pix_loss:
		target = model.patchify(images)
		mean = target.mean(dim=-1, keepdim=True)
		std = (target.var(dim=-1, keepdim=True) + 1.e-6).sqrt()
		pred = pred * std + mean

	original = images
	predicted = model.unpatchify(pred)
	mask_image = model.unpatchify(
		mask.unsqueeze(-1).expand(-1, -1, pred.shape[-1])
	).clamp(0, 1)
	masked = original * (1 - mask_image)
	reconstruction = original * (1 - mask_image) + predicted * mask_image

	def to_image(tensor):
		image = tensor.detach().cpu().permute(0, 2, 3, 1).numpy()
		image = image * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
		return np.clip(image, 0, 1)

	original_images = to_image(original)
	masked_images = to_image(masked)
	reconstruction_images = to_image(reconstruction)
	for image_number, (full_image, masked_image, reconstruction_image) in enumerate(
		zip(original_images, masked_images, reconstruction_images), start=1
	):
		image_dir = output_dir / f"image_{image_number:02d}"
		image_dir.mkdir(parents=True, exist_ok=True)
		plt.imsave(image_dir / "full.png", full_image)
		plt.imsave(image_dir / "masked.png", masked_image)
		plt.imsave(image_dir / "reconstruction.png", reconstruction_image)


def save_loss_plot(train_loss_history, val_loss_history, output_path, run_id, dataset_name):
	output_path.parent.mkdir(parents=True, exist_ok=True)
	epochs = np.arange(1, len(train_loss_history) + 1)
	fig, ax = plt.subplots(figsize=(8, 5))
	ax.plot(epochs, train_loss_history, label="Train loss")
	ax.plot(epochs, val_loss_history, label="Validation loss")
	ax.set_title(f"ConvMAE loss | {dataset_name} | run {run_id}")
	ax.set_xlabel("Epoch")
	ax.set_ylabel("Loss")
	ax.grid(True, alpha=0.3)
	ax.legend()
	fig.tight_layout()
	fig.savefig(output_path, dpi=150)
	plt.close(fig)


def train_dataset(dataset_name, args):
	run_id = uuid.uuid4().hex
	start_time = time.perf_counter()
	seed_everything(args.seed)
	device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")
	DatasetClass = DATASETS[dataset_name]
	train_transform, valid_transform = make_transforms()

	train_dataset = DatasetClass(
		split="train", root=str(args.data_root), size=IMAGE_SIZE,
		as_rgb=True, transform=train_transform, download=False,
	)
	val_dataset = DatasetClass(
		split="val", root=str(args.data_root), size=IMAGE_SIZE,
		as_rgb=True, transform=valid_transform, download=False,
	)
	visualization_dir = SCRIPT_DIR / "reconstructions" / dataset_name / run_id
	rng = np.random.default_rng(args.seed)
	num_visualizations = min(args.num_visualizations, len(val_dataset))
	sample_indices = rng.choice(len(val_dataset), size=num_visualizations, replace=False).tolist()
	sampler = None
	if not args.no_oversampling:
		weights = torch.as_tensor(make_sample_weights(train_dataset.labels), dtype=torch.double)
		sampler = WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)

	train_loader = build_loader(train_dataset, args.batch_size, args.num_workers, sampler, args.seed)
	val_loader = build_loader(val_dataset, args.val_batch_size, args.num_workers, seed=args.seed)
	model = load_model(args.checkpoint, device, args.norm_pix_loss)
	save_reconstructions(
		model=model,
		dataset=val_dataset,
		sample_indices=sample_indices,
		device=device,
		mask_ratio=args.mask_ratio,
		output_dir=visualization_dir / "epoch_0000",
		epoch=0,
		norm_pix_loss=args.norm_pix_loss,
	)

	encoder_params = [p for name, p in model.named_parameters() if "decoder" not in name]
	decoder_params = [p for name, p in model.named_parameters() if "decoder" in name]
	optimizer = torch.optim.AdamW([
		{"params": encoder_params, "lr": args.encoder_lr},
		{"params": decoder_params, "lr": args.decoder_lr},
	], betas=(0.9, 0.95), weight_decay=args.weight_decay)
	scaler = torch.cuda.amp.GradScaler(enabled=device.type == "cuda")

	best_val_loss = float("inf")
	best_train_loss = float("nan")
	best_epoch = -1
	train_loss_history = []
	val_loss_history = []
	best_state = None
	for epoch in range(args.epochs):
		train_loss = run_epoch(model, train_loader, device, args.mask_ratio, optimizer, scaler)
		with torch.no_grad():
			val_loss = run_epoch(model, val_loader, device, args.mask_ratio)
		train_loss_history.append(train_loss)
		val_loss_history.append(val_loss)
		if (epoch + 1) % 10 == 0 or epoch == args.epochs - 1:
			save_reconstructions(
				model=model,
				dataset=val_dataset,
				sample_indices=sample_indices,
				device=device,
				mask_ratio=args.mask_ratio,
				output_dir=visualization_dir / f"epoch_{epoch + 1:04d}",
				epoch=epoch + 1,
				norm_pix_loss=args.norm_pix_loss,
			)
		print(
			f"{dataset_name} | epoch {epoch + 1}/{args.epochs} | "
			f"train loss {train_loss:.5f} | val loss {val_loss:.5f}"
		)
		if val_loss < best_val_loss:
			best_val_loss = val_loss
			best_train_loss = train_loss
			best_epoch = epoch + 1
			best_state = copy.deepcopy(model.state_dict())

	loss_plot_path = visualization_dir / "loss_curve.png"
	save_loss_plot(
		train_loss_history=train_loss_history,
		val_loss_history=val_loss_history,
		output_path=loss_plot_path,
		run_id=run_id,
		dataset_name=dataset_name,
	)
	args.output_root.mkdir(parents=True, exist_ok=True)
	output_path = args.output_root / f"convmae_{dataset_name}_{run_id}.pth"
	torch.save(best_state, output_path)

	result = {
		"run_id": run_id,
		"dataset": dataset_name,
		"train_loss_history": train_loss_history,
		"val_loss_history": val_loss_history,
		"best_epoch": best_epoch,
		"best_train_loss": best_train_loss,
		"best_val_loss": best_val_loss,
		"epochs": args.epochs,
		"batch_size": args.batch_size,
		"val_batch_size": args.val_batch_size,
		"num_workers": args.num_workers,
		"encoder_lr": args.encoder_lr,
		"decoder_lr": args.decoder_lr,
		"weight_decay": args.weight_decay,
		"mask_ratio": args.mask_ratio,
		"norm_pix_loss": args.norm_pix_loss,
		"oversampling": not args.no_oversampling,
		"seed": args.seed,
		"device": str(device),
		"elapsed_seconds": time.perf_counter() - start_time,
		"model_path": str(output_path),
		"visualization_dir": str(visualization_dir),
		"loss_plot_path": str(loss_plot_path),
	}
	append_result(args.results_path, result)

	print(f"Saved results row to {args.results_path}")
	print(f"Saved best {dataset_name} model to {output_path}")


def main():
	args = parse_args()
	print(args)
	dataset_names = list(DATASETS) if args.dataset == "all" else [args.dataset]
	for dataset_name in dataset_names:
		train_dataset(dataset_name, args)


if __name__ == "__main__":
	main()
