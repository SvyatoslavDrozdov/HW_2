import os
import numpy as np
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import segmentation_models_pytorch as smp
import time

if torch.cuda.is_available():
    device = torch.device("cuda")
    print("GPU is used:", torch.cuda.get_device_name(0))
else:
    device = torch.device("cpu")
    print("CPU is used.")


def encoded_pixels_to_masks(fname: str, df: pd.DataFrame):
    fname_df = df[df['ImageId'] == fname]
    masks = np.zeros((256 * 1600, 4), dtype=int)

    for i_row, row in fname_df.iterrows():
        cls_id = row['ClassId']
        encoded_pixels = row['EncodedPixels']
        if encoded_pixels is not np.nan:
            pixel_list = list(map(int, encoded_pixels.split(' ')))
            for i in range(0, len(pixel_list), 2):
                start_pixel = pixel_list[i] - 1
                num_pixel = pixel_list[i + 1]
                masks[start_pixel:(start_pixel + num_pixel), cls_id - 1] = 1

    masks = masks.reshape(256, 1600, 4, order='F')
    return masks


def masks_to_encoded_pixels(mask_layer, class_number, class_thresholds):
    cutoff = class_thresholds[class_number]
    flat_mask = mask_layer.flatten(order='F')
    binarized_mask = (flat_mask > cutoff).astype(int)
    modernized_mask = np.r_[0, binarized_mask, 0]
    edge_points = np.flatnonzero(modernized_mask[1:] != modernized_mask[:-1]) + 1
    run_lengths = edge_points[1::2] - edge_points[::2]
    run_starts = edge_points[::2]
    pairs = [f"{start} {length}" for start, length in zip(run_starts, run_lengths)]
    encoded_rle = ' '.join(pairs)
    return encoded_rle


class SeverstalSteelDataset(Dataset):
    def __init__(self, fnames, df, img_dir, transform=None):
        self.df = df
        self.img_dir = img_dir
        self.fnames = fnames
        self.transform = transform

    def __len__(self):
        return len(self.fnames)

    def __getitem__(self, idx):
        img_id = self.fnames[idx]
        img_path = os.path.join(self.img_dir, img_id)
        img = np.array(Image.open(img_path).convert('RGB'))
        masks = encoded_pixels_to_masks(img_id, self.df)
        if self.transform:
            img = self.transform(image=img)['image']
        img = torch.tensor(img, dtype=torch.float32).permute(2, 0, 1)
        masks = torch.tensor(masks, dtype=torch.float32).permute(2, 0, 1)
        return img_id, img, masks


def collate_fn(batch_items):
    batched_fnames = [item[0] for item in batch_items]
    batched_imgs = torch.stack([item[1] for item in batch_items])
    batched_masks = torch.stack([item[2] for item in batch_items])

    return batched_fnames, batched_imgs, batched_masks


class SegModel(torch.nn.Module):
    def __init__(self):
        super(SegModel, self).__init__()
        self.model = smp.Unet(
            encoder_name='resnet34',
            encoder_weights='imagenet',
            classes=4,
            activation=None
        )

    def forward(self, x):
        return self.model(x)


def load_data(
        train_csv,
        test_csv,
        train_img_dir,
        test_img_dir,
        max_train_samples=None,
        max_val_samples=None,
        max_test_samples=None
):
    train_df = pd.read_csv(train_csv)
    train_fnames = pd.unique(train_df.ImageId)

    test_df = pd.read_csv(test_csv)
    test_fnames = pd.unique(test_df.ImageId)

    train_fnames, val_fnames = train_test_split(train_fnames, test_size=0.2, random_state=42)

    if max_train_samples is not None:
        train_fnames = train_fnames[:max_train_samples]
    if max_val_samples is not None:
        val_fnames = val_fnames[:max_val_samples]
    if max_test_samples is not None:
        test_fnames = test_fnames[:max_test_samples]

    train_dataset = SeverstalSteelDataset(train_fnames, train_df, train_img_dir)
    val_dataset = SeverstalSteelDataset(val_fnames, train_df, train_img_dir)
    test_dataset = SeverstalSteelDataset(test_fnames, test_df, test_img_dir)

    train_loader_ = DataLoader(train_dataset, batch_size=4, shuffle=True, collate_fn=collate_fn, num_workers=2)
    val_loader_ = DataLoader(val_dataset, batch_size=4, shuffle=False, collate_fn=collate_fn, num_workers=2)
    test_loader_ = DataLoader(test_dataset, batch_size=4, shuffle=False, collate_fn=collate_fn, num_workers=2)

    return train_loader_, val_loader_, test_loader_


def init_model():
    model = SegModel()
    criterion = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    return model, criterion, optimizer


thresholds = [0.5, 0.5, 0.7, 0.5]


def train(model, criterion, optimizer, loader, epochs, my_device):
    model.train()
    for epoch in range(epochs):
        time_start = time.time()
        running_loss = 0.0
        for batch_fnames, batch_imgs, batch_masks in loader:
            batch_imgs, batch_masks = batch_imgs.to(my_device), batch_masks.to(my_device)

            optimizer.zero_grad()
            outputs = model(batch_imgs)
            loss = criterion(outputs, batch_masks)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        print(f"Epoch {epoch + 1}/{epochs}, Loss: {running_loss / len(loader)}")
        print(f"Computation time: {time.time() - time_start} sec.")
    return model


def compute_dice_coefficient(predicted, ground_truth):
    predicted = predicted.flatten()
    ground_truth = ground_truth.flatten()

    intersection = (predicted * ground_truth).sum()
    total_sum = predicted.sum() + ground_truth.sum()
    if total_sum:
        dice_value = 2 * intersection / total_sum
    else:
        return 1
    return dice_value


def evaluate(model, data_loader, my_device):
    model.eval()
    dice_results = []

    with torch.no_grad():
        for file_names, images, masks in data_loader:
            images = images.to(my_device)
            masks = masks.to(my_device)

            predictions = model(images)
            probabilities = torch.sigmoid(predictions) > 0.5

            predictions = probabilities.cpu().numpy()
            masks = masks.cpu().numpy()

            for idx in range(len(file_names)):
                for class_idx in range(4):
                    dice_score_value = compute_dice_coefficient(predictions[idx, class_idx], masks[idx, class_idx])
                    dice_results.append(dice_score_value)

    average_dice = np.mean(dice_results)
    print(f"Overall Average Dice Score: {average_dice:.4f}")

    for class_idx in range(4):
        class_dice = np.mean([score for idx, score in enumerate(dice_results) if idx % 4 == class_idx])
        print(f"Class {class_idx + 1} Dice Score: {class_dice:.4f}")


def write_submission(model, test_loader_, my_device, thresholds_):
    model.eval()
    submission = []

    with torch.no_grad():
        for batch_fnames, batch_imgs, _ in test_loader_:
            batch_imgs = batch_imgs.to(my_device)
            outputs = model(batch_imgs)
            preds = torch.sigmoid(outputs)

            preds = preds.cpu().numpy()
            for i, fname in enumerate(batch_fnames):
                encoded_pixels_list = []
                for cls_id in range(4):
                    encoded_pixels = masks_to_encoded_pixels(preds[i, cls_id], cls_id, thresholds_)
                    encoded_pixels_list.append((fname, cls_id + 1, encoded_pixels))

                submission.extend(encoded_pixels_list)

    submission_df = pd.DataFrame(submission, columns=['ImageId', 'ClassId', 'EncodedPixels'])
    submission_df.to_csv('submission.csv', index=False)


train_loader, val_loader, test_loader = load_data(
    'train.csv',
    'sample_submission.csv',
    'train_images',
    'test_images',
    max_train_samples=15000,
    max_val_samples=15000,
    max_test_samples=15000
)
