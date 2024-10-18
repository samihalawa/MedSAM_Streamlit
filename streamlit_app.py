# 1. Import necessary libraries
import os
import numpy as np
import matplotlib.pyplot as plt
import cv2
import requests
from tqdm import tqdm
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

import albumentations as A
from albumentations.pytorch import ToTensorV2

import streamlit as st

# 2. Import MedSAM libraries
from segment_anything import sam_model_registry, SamPredictor
from segment_anything.utils.transforms import ResizeLongestSide

# 3. Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 4. Function to download MedSAM model
def download_medsam_model():
    """
    Download the MedSAM model checkpoint if not already present.
    """
    model_url = "https://github.com/bowang-lab/MedSAM/raw/main/weights/medsam_vit_b.pth"
    model_path = "medsam_vit_b.pth"
    if not os.path.exists(model_path):
        st.write("Downloading MedSAM model...")
        response = requests.get(model_url, stream=True)
        total_size = int(response.headers.get('content-length', 0))
        block_size = 1024
        with open(model_path, "wb") as f:
            for data in tqdm(response.iter_content(block_size), total=total_size//block_size, unit='KB'):
                f.write(data)
        st.write("MedSAM model downloaded.")
    else:
        st.write("MedSAM model already exists.")
    return model_path

# 5. Dataset class for Glaucoma images
class GlaucomaDataset(Dataset):
    """
    Custom Dataset class for Glaucoma images and masks.
    """
    def __init__(self, images, masks, transform=None):
        self.images = images
        self.masks = masks
        self.transform = transform
        self.resize = A.Resize(1024, 1024)
        self.to_tensor = ToTensorV2()

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        # Load image and mask
        image = np.array(Image.open(self.images[idx]).convert("RGB"))
        mask = np.array(Image.open(self.masks[idx]))
        
        # Apply resizing
        resized = self.resize(image=image, mask=mask)
        image = resized['image']
        mask = resized['mask']

        # Apply transformations
        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented['image']
            mask = augmented['mask']

        # Convert to tensor
        image = self.to_tensor(image=image)['image']
        mask = torch.tensor(mask, dtype=torch.float32).unsqueeze(0)

        return image, mask

# 6. Function to get data transformations
def get_transforms():
    """
    Get data augmentation transformations using Albumentations.
    """
    return A.Compose([
        A.HorizontalFlip(p=0.5),
        A.RandomBrightnessContrast(p=0.5),
        A.ShiftScaleRotate(p=0.5),
        A.RandomCrop(height=512, width=512, p=1.0),
    ])

# 7. Function to compute Dice coefficient
def compute_dice_coef(preds, targets, smooth=1e-7):
    """
    Compute the Dice coefficient metric.
    """
    preds = preds.contiguous().view(-1)
    targets = targets.contiguous().view(-1)
    intersection = (preds * targets).sum()
    dice = (2. * intersection + smooth) / (preds.sum() + targets.sum() + smooth)
    return dice.item()

# 8. Function to compute vertical cup-to-disc ratio (vCDR)
def compute_vcdr(pred_od, pred_oc):
    """
    Compute vertical cup-to-disc ratio from predicted masks.
    """
    pred_od = pred_od.squeeze().cpu().numpy()
    pred_oc = pred_oc.squeeze().cpu().numpy()
    od_diameter = np.max(np.sum(pred_od, axis=1))
    oc_diameter = np.max(np.sum(pred_oc, axis=1))
    vcdr = oc_diameter / (od_diameter + 1e-7)
    return vcdr

# 9. Function to compute vCDR error
def compute_vCDR_error(pred_od, pred_oc, gt_od, gt_oc):
    """
    Compute vCDR prediction error, along with predicted and ground truth vCDR.
    """
    pred_vCDR = compute_vcdr(pred_od, pred_oc)
    gt_vCDR = compute_vcdr(gt_od, gt_oc)
    vCDR_err = np.abs(gt_vCDR - pred_vCDR)
    return vCDR_err, pred_vCDR, gt_vCDR

# 10. Function to refine segmentation masks
def refine_segmentation(pred_mask):
    """
    Refine segmentation by keeping the largest connected component.
    """
    pred_mask_np = pred_mask.squeeze().cpu().numpy()
    num_labels, labels_im = cv2.connectedComponents(pred_mask_np.astype(np.uint8))
    if num_labels > 1:
        max_label = 1 + np.argmax([np.sum(labels_im == i) for i in range(1, num_labels)])
        refined_mask = (labels_im == max_label).astype(np.float32)
        refined_mask = torch.tensor(refined_mask).unsqueeze(0)
    else:
        refined_mask = pred_mask
    return refined_mask

# 11. MedSAM Model class
class MedSAMModel(nn.Module):
    """
    MedSAM Model class for fine-tuning.
    """
    def __init__(self, model_path):
        super(MedSAMModel, self).__init__()
        self.sam = sam_model_registry["vit_b"](checkpoint=model_path)
        self.sam.to(device)
        # Freeze image encoder
        for param in self.sam.image_encoder.parameters():
            param.requires_grad = False

    def forward(self, images):
        # Get image embeddings
        with torch.no_grad():
            image_embeddings = self.sam.image_encoder(images)

        # Generate prompt embeddings
        batch_size = images.shape[0]
        input_points = torch.tensor([[[images.shape[-1] // 2, images.shape[-2] // 2]]]*batch_size).to(device)
        input_labels = torch.ones((batch_size, 1), dtype=torch.int).to(device)

        sparse_embeddings, dense_embeddings = self.sam.prompt_encoder(
            points=(input_points, input_labels),
            boxes=None,
            masks=None,
        )

        # Run mask decoder
        low_res_masks, _ = self.sam.mask_decoder(
            image_embeddings=image_embeddings,
            image_pe=self.sam.prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=False,
        )

        # Upsample masks to original image size
        masks = F.interpolate(
            low_res_masks,
            size=(images.shape[-2], images.shape[-1]),
            mode='bilinear',
            align_corners=False
        )
        return masks

# 12. Training function
def train_model(model, dataloaders, criterion, optimizer, scheduler, num_epochs):
    """
    Training loop for the model.
    """
    best_model_wts = None
    best_dice = 0.0
    train_losses = []
    val_losses = []
    train_dices = []
    val_dices = []

    for epoch in range(num_epochs):
        st.write(f'Epoch {epoch+1}/{num_epochs}')
        st.write('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
                dataloader = dataloaders['train']
            else:
                model.eval()
                dataloader = dataloaders['val']

            running_loss = 0.0
            running_dice = 0.0

            # Iterate over data
            for inputs, masks in tqdm(dataloader):
                inputs = inputs.to(device)
                masks = masks.to(device)

                # Zero the parameter gradients
                optimizer.zero_grad()

                # Forward pass
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    outputs = outputs.squeeze(1)
                    preds = torch.sigmoid(outputs)
                    loss = criterion(outputs, masks)

                    # Backward and optimize
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # Statistics
                preds_binary = (preds > 0.5).float()
                running_loss += loss.item() * inputs.size(0)
                running_dice += compute_dice_coef(preds_binary, masks) * inputs.size(0)

            epoch_loss = running_loss / len(dataloader.dataset)
            epoch_dice = running_dice / len(dataloader.dataset)

            if phase == 'train':
                train_losses.append(epoch_loss)
                train_dices.append(epoch_dice)
                scheduler.step()
            else:
                val_losses.append(epoch_loss)
                val_dices.append(epoch_dice)

            st.write(f'{phase.capitalize()} Loss: {epoch_loss:.4f} Dice: {epoch_dice:.4f}')

            # Deep copy the model
            if phase == 'val' and epoch_dice > best_dice:
                best_dice = epoch_dice
                best_model_wts = model.state_dict()
                # Save the best model
                torch.save(model.state_dict(), 'best_model.pth')
                st.write("Best model saved.")

        # Plot loss and dice curves
        plot_metrics(train_losses, val_losses, train_dices, val_dices)

    st.write(f'Best Validation Dice: {best_dice:.4f}')

    # Load best model weights
    if best_model_wts:
        model.load_state_dict(best_model_wts)
    return model

# 13. Function to plot training metrics
def plot_metrics(train_losses, val_losses, train_dices, val_dices):
    """
    Plot training and validation loss and Dice coefficient curves.
    """
    fig, ax = plt.subplots(1, 2, figsize=(12, 5))

    # Plot losses
    ax[0].plot(train_losses, label='Train Loss')
    ax[0].plot(val_losses, label='Validation Loss')
    ax[0].set_title('Loss')
    ax[0].legend()

    # Plot Dice scores
    ax[1].plot(train_dices, label='Train Dice')
    ax[1].plot(val_dices, label='Validation Dice')
    ax[1].set_title('Dice Coefficient')
    ax[1].legend()

    st.pyplot(fig)

# 14. Inference function
def inference(model, image_path):
    """
    Run inference on a single image and return the result with overlay.
    """
    model.eval()
    image = np.array(Image.open(image_path).convert("RGB"))
    resize = A.Resize(1024, 1024)
    to_tensor = ToTensorV2()
    image_resized = resize(image=image)['image']
    image_tensor = to_tensor(image=image_resized)['image'].unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(image_tensor)
        output = output.squeeze(1)
        preds = torch.sigmoid(output)
        pred_mask = (preds > 0.5).float().cpu().numpy().squeeze()

    # Resize prediction back to original size
    pred_resized = cv2.resize(pred_mask, (image.shape[1], image.shape[0]))
    pred_resized = (pred_resized > 0.5).astype(np.uint8)

    # Create overlay
    overlay = image.copy()
    overlay[pred_resized == 1] = [255, 0, 0]  # Red color for mask
    overlay = cv2.addWeighted(image, 0.7, overlay, 0.3, 0)

    return overlay

# 15. Function to download sample dataset
def download_sample_dataset():
    """
    Download a sample Glaucoma dataset for demonstration purposes.
    """
    dataset_url = "https://github.com/username/sample-glaucoma-dataset/archive/refs/heads/main.zip"
    dataset_path = "glaucoma_dataset.zip"
    if not os.path.exists('glaucoma_dataset'):
        st.write("Downloading sample dataset...")
        response = requests.get(dataset_url, stream=True)
        total_size = int(response.headers.get('content-length', 0))
        block_size = 1024
        with open(dataset_path, "wb") as f:
            for data in tqdm(response.iter_content(block_size), total=total_size//block_size, unit='KB'):
                f.write(data)
        st.write("Extracting dataset...")
        with zipfile.ZipFile(dataset_path, 'r') as zip_ref:
            zip_ref.extractall('.')
        os.remove(dataset_path)
        st.write("Dataset ready.")
    else:
        st.write("Sample dataset already exists.")

# 16. Main function for Streamlit app
def main():
    """
    Main function to run the Streamlit app.
    """
    st.title("Glaucoma Segmentation with MedSAM")

    # Sidebar settings
    st.sidebar.title("Settings")
    data_option = st.sidebar.selectbox("Select Data Option", ("Use Sample Dataset", "Upload Your Own Data"))
    num_epochs = st.sidebar.slider("Number of Epochs", 1, 50, 5)
    batch_size = st.sidebar.slider("Batch Size", 1, 16, 2)
    learning_rate = st.sidebar.select_slider("Learning Rate", options=[1e-4, 1e-3, 1e-2], value=1e-3)
    val_split = st.sidebar.slider("Validation Split", 0.1, 0.5, 0.2)

    # Data handling
    if data_option == "Use Sample Dataset":
        if st.sidebar.button("Download Sample Dataset"):
            download_sample_dataset()
            st.success("Sample dataset downloaded successfully.")
        data_dir = "glaucoma_dataset"
        images_dir = os.path.join(data_dir, "Images")
        masks_dir = os.path.join(data_dir, "Masks")
        image_files = [os.path.join(images_dir, fname) for fname in os.listdir(images_dir) if fname.endswith(('.jpg', '.png', '.jpeg'))]
        mask_files = [os.path.join(masks_dir, fname) for fname in os.listdir(masks_dir) if fname.endswith(('.jpg', '.png', '.jpeg'))]
        image_files.sort()
        mask_files.sort()
    else:
        # Upload images and masks
        uploaded_images = st.sidebar.file_uploader("Upload Images", type=['png', 'jpg', 'jpeg'], accept_multiple_files=True)
        uploaded_masks = st.sidebar.file_uploader("Upload Masks", type=['png', 'jpg', 'jpeg'], accept_multiple_files=True)

    # Load data
    if st.button("Load Data"):
        if data_option == "Upload Your Own Data":
            if uploaded_images and uploaded_masks:
                os.makedirs('uploaded_images', exist_ok=True)
                os.makedirs('uploaded_masks', exist_ok=True)
                image_files = []
                mask_files = []
                for img_file, mask_file in zip(uploaded_images, uploaded_masks):
                    image_path = os.path.join('uploaded_images', img_file.name)
                    mask_path = os.path.join('uploaded_masks', mask_file.name)
                    with open(image_path, 'wb') as f:
                        f.write(img_file.getbuffer())
                    with open(mask_path, 'wb') as f:
                        f.write(mask_file.getbuffer())
                    image_files.append(image_path)
                    mask_files.append(mask_path)
            else:
                st.error("Please upload both images and masks.")
                return

        # Split data
        train_imgs, val_imgs, train_masks, val_masks = train_test_split(
            image_files, mask_files, test_size=val_split, random_state=42
        )

        # Create datasets
        train_dataset = GlaucomaDataset(train_imgs, train_masks, transform=get_transforms())
        val_dataset = GlaucomaDataset(val_imgs, val_masks)

        # Create dataloaders
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
        dataloaders = {'train': train_loader, 'val': val_loader}
        st.success("Data loaded successfully.")

    # Load MedSAM model
    model_path = download_medsam_model()

    # Initialize model
    model = MedSAMModel(model_path).to(device)

    # Define loss function and optimizer
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.sam.mask_decoder.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min')

    # Train model
    if st.button("Train Model"):
        if 'dataloaders' in locals():
            with st.spinner("Training model..."):
                model = train_model(model, dataloaders, criterion, optimizer, scheduler, num_epochs)
            st.success("Model trained successfully.")
        else:
            st.error("Please load the data first.")

    # Inference
    st.header("Inference")
    uploaded_test_image = st.file_uploader("Upload an Image for Inference", type=['png', 'jpg', 'jpeg'])
    if uploaded_test_image and st.button("Run Inference"):
        with st.spinner("Running inference..."):
            # Save uploaded image
            os.makedirs('test_images', exist_ok=True)
            test_image_path = os.path.join('test_images', uploaded_test_image.name)
            with open(test_image_path, 'wb') as f:
                f.write(uploaded_test_image.getbuffer())
            result = inference(model, test_image_path)
            st.image(result, caption="Segmentation Result")

    # Download model
    if os.path.exists('best_model.pth'):
        with open('best_model.pth', 'rb') as f:
            st.download_button('Download Model', f, file_name='best_model.pth')

if __name__ == "__main__":
    main()
