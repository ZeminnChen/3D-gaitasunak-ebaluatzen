import os, json, torch, random, glob, natsort
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as T
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from transformers import CLIPModel, CLIPProcessor
from torchmetrics.classification import BinaryJaccardIndex


# ======================================
# CLIP-en OINARRITUTAKO ESPERIMENTUA
# ======================================
seed = 0
random.seed(seed)
np.random.seed(seed)             # numpy
torch.manual_seed(seed)          # cpu
torch.cuda.manual_seed(seed)     # gpu
torch.cuda.manual_seed_all(seed) # gpu guztientzat


# ======================================
# 1. CLIP PROCESSOR + PADDING
# ======================================

## Padding
class ResizeWithPadding:
    def __init__(self, size, border_size, top_factor, bottom_factor):
        self.size = size
        self.border_size = border_size
        self.top_factor = top_factor
        self.bottom_factor = bottom_factor

    def get_background_color(self, img):
        img_np = np.array(img)
        b = self.border_size

        top = img_np[:b, :, :]
        bottom = img_np[-b:, :, :]
        left = img_np[:, :b, :]
        right = img_np[:, -b:, :]

        border_pixels = np.concatenate([top.reshape(-1,3), bottom.reshape(-1,3), left.reshape(-1,3), right.reshape(-1,3)], axis=0)

        return border_pixels.mean(axis=0)

    def __call__(self, img):
        w, h = img.size

        scale = self.size / max(w, h)
        new_w, new_h = int(w * scale), int(h * scale)
        img_resized = img.resize((new_w, new_h), Image.BICUBIC)

        base_color = self.get_background_color(img_resized)

        canvas = np.zeros((self.size, self.size, 3), dtype=np.uint8)

        y0 = (self.size - new_h)//2
        y1 = y0 + new_h

        # Create top and bottom padding
        for y in range(self.size):
            if y < y0:  # top padding
                factor = self.top_factor
            elif y >= y1:  # bottom padding
                factor = self.bottom_factor
            else:
                factor = 1.0

            color = np.clip(base_color * factor, 0, 255).astype(np.uint8)
            canvas[y, :, :] = color

        x0 = (self.size - new_w)//2
        canvas[y0:y1, x0:x0+new_w] = np.array(img_resized)

        return Image.fromarray(canvas)
    
## ClevrDataset
class CLEVR3DDataset(Dataset):
    def __init__(self, image_paths, voxel_paths, processor, transform=None):
        self.image_paths = image_paths
        self.voxel_paths = voxel_paths
        self.processor = processor
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert("RGB")

        # 1. Padding
        if self.transform is not None:
            image = self.transform(image)

        # 2. CLIP preprocessing: normalize + tensor
        inputs = self.processor.image_processor(images=image, return_tensors="pt", do_resize=False, do_center_crop=False)
        image = inputs.pixel_values.squeeze(0)

        # 3. Voxels
        voxel_array = np.load(self.voxel_paths[idx])
        voxels = torch.from_numpy(voxel_array).float()

        return image, voxels


# ======================================
# 2. DECODER
# ======================================

# ======================================
# 2.1 Reassemble 
# ======================================
class ReadProj(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        self.embed_dim = embed_dim
        self.proj = nn.Linear(2 * embed_dim, embed_dim)
        self.act = nn.GELU()

    def forward(self, x):
        cls_token = x[:, 0:1, :]           
        tokens = x[:, 1:, :]               

        cls_repeated = cls_token.expand(-1, tokens.size(1), -1) 
        concat = torch.cat([tokens, cls_repeated], dim=-1)                   

        return  self.act(self.proj(concat))

class Reassemble(nn.Module):
    def __init__(self, in_dim=1024, out_dim=256):
        super().__init__()
        self.read_proj = ReadProj(in_dim)
        self.proj = nn.Conv2d(in_dim, out_dim, kernel_size=1)

    def forward(self, x):
        x = self.read_proj(x)  
        B, N, D = x.shape
        H = W = int(N ** 0.5)
        x = x.view(B, H, W, D).permute(0, 3, 1, 2)  
        x = self.proj(x)  
        return x  # (B, 256, 7, 7)

# ======================================
# 2.2 DPT Fusion
# ======================================
class DPTFusion(nn.Module):
    def __init__(self, dim=256):
        super().__init__()
        self.conv4 = nn.Conv2d(dim, dim, 3, padding=1)
        self.conv3 = nn.Conv2d(dim, dim, 3, padding=1)
        self.conv2 = nn.Conv2d(dim, dim, 3, padding=1)
        self.conv1 = nn.Conv2d(dim, dim, 3, padding=1)
        self.out_conv = nn.Conv2d(dim, dim, 3, padding=1)
        self.relu = nn.ReLU()

    def forward(self, feats):   
        f1, f2, f3, f4 = feats
        x = self.conv4(f4)
        x = self.relu(x + self.conv3(f3))
        x = self.relu(x + self.conv2(f2))
        x = self.relu(x + self.conv1(f1))
        return self.out_conv(x)


# ======================================
# 2.3 Cross Attention 2D → 3D 
# ======================================
class LiftTo3D(nn.Module):
    def __init__(self, in_channels=256, target_voxels=4):
        super().__init__()
        self.in_channels = in_channels
        self.target_voxels = target_voxels
        
        # 1. Kanalak proiektatu: 256 → 256 * target_voxels
        self.depth_projector = nn.Sequential(
            nn.Conv2d(in_channels, in_channels * target_voxels, kernel_size=1),
            nn.BatchNorm2d(in_channels * target_voxels),
            nn.GELU())
        
        self.adaptive_pool = nn.AdaptiveAvgPool3d((target_voxels, target_voxels, target_voxels))
        
        self.refine_3d = nn.Sequential(
            nn.Conv3d(in_channels, in_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(in_channels),
            nn.GELU()
        )
    
    def forward(self, x):
        B, C, H, W = x.shape
       
        # 1. Learned Depth Projection. [B, 256, H, W] -> [B, 1024, H, W]
        x = self.depth_projector(x)
       
        # 2. Reshape to 3D. [B, 256, 4, H, W]
        x = x.view(B, self.in_channels, self.target_voxels, H, W)
        x = self.adaptive_pool(x)  
       
       # 3. Refine in 3D space
        x = self.refine_3d(x)
       
        return x
        
        
# ======================================
# 2.4 DPT3DModel
# ======================================
class DPT3DModel(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.clip = clip_model
        self.reassemble = nn.ModuleList([Reassemble(1024, 256) for _ in range(4)])
        self.fusion = DPTFusion(256)
        self.lift3d = LiftTo3D(256)
        self.decoder3d = nn.Sequential(
            nn.ConvTranspose3d(256, 128, 4, 2, 1),
            nn.BatchNorm3d(128),
            nn.ReLU(),
            nn.ConvTranspose3d(128, 64, 4, 2, 1),
            nn.BatchNorm3d(64),
            nn.ReLU(),
            nn.ConvTranspose3d(64, 32, 4, 2, 1),
            nn.BatchNorm3d(32),
            nn.ReLU(),
            nn.ConvTranspose3d(32, 1, 4, 2, 1),
        )

    def forward(self, images):
        with torch.no_grad():
            outputs = self.clip.vision_model(pixel_values=images, output_hidden_states=True)
        hidden_states = outputs.hidden_states

        layers = [hidden_states[i] for i in [5, 12, 18, 24]]

        # 1. Reassemble
        feats = [r(l) for r, l in zip(self.reassemble, layers)]

        # 2. Multiscale fusion
        x = self.fusion(feats)  # (B, 256, H, W)

        # 3. Lift 2D → 3D
        x = self.lift3d(x)      # (B, 256, 4, 4, 4)

        # Decoder 3D → voxels
        x = self.decoder3d(x)   # (B, 1, 64, 64, 64)
        return x.squeeze(1)


# ======================================
# 4. Train + Test
# ======================================
def get_dataloader(indices, mode, batch_size, processor):
    base_path = "/home/zchen002/TFG"
    
    if indices is not None: # train + val
        img_dir = os.path.join(base_path, f"CLEVR_images/images/train/")
        vox_dir = os.path.join(base_path, f"voxels/train/")
        image_paths = [os.path.join(img_dir, f"CLEVR_train_{str(idx).zfill(6)}.png") for idx in indices]
        voxel_paths = [os.path.join(vox_dir, f"scene_{str(idx).zfill(6)}.npy") for idx in indices]
    else: # test
        img_dir = os.path.join(base_path, f"CLEVR_images/images/test/")
        vox_dir = os.path.join(base_path, f"voxels/test/")
        image_paths = natsort.natsorted(glob.glob(os.path.join(img_dir, "*.png")))
        voxel_paths = natsort.natsorted(glob.glob(os.path.join(vox_dir, "*.npy")))

    transform = ResizeWithPadding(size=224, border_size=10, top_factor=0.9, bottom_factor=1.2)
    dataset = CLEVR3DDataset(image_paths, voxel_paths, processor, transform)
    
    return DataLoader(dataset, batch_size=batch_size, shuffle=(mode == "train"))


## Train
def train(dataloader, decoder, criterion, optimizer, metric, device):
    decoder.train()
    metric.reset()
    
    total_loss = 0
    
    for images, voxels in dataloader:
        images, voxels = images.to(device), voxels.to(device)

        # 1. Forward DPT
        pred = decoder(images)
        loss = criterion(pred, voxels)

        # 2. Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # 3. Metrics
        total_loss += loss.item()
        
        preds_bin = (torch.sigmoid(pred) > 0.5).int()
        metric.update(preds_bin, voxels.int())

    return total_loss / len(dataloader), metric.compute().item()


## Test
def test(dataloader, decoder, criterion, metric, device, save_predictions, save_dir):
    decoder.eval()
    
    all_preds = []
    metric.reset()
    total_loss = 0
    
    with torch.no_grad():
        for images, voxels in dataloader:
            images, voxels = images.to(device), voxels.to(device)
            
            # 1. Forward DPT
            pred = decoder(images)
            total_loss += criterion(pred, voxels).item()
            
            # 2. IoU
            preds_bin = (torch.sigmoid(pred) > 0.5).int()
            metric.update(preds_bin, voxels.int())

            if save_predictions:
                all_preds.append(pred.cpu())
    
    if save_predictions:
        all_preds = torch.cat(all_preds, dim=0)
        selected_preds = torch.cat([all_preds[:10], all_preds[-10:]], dim=0)
        np.save(os.path.join(save_dir, f"test_clip_32_average_pooling2.npy"), selected_preds.numpy())
        print(f"Test loss: {total_loss / len(dataloader):.4f}, mIoU: {metric.compute().item():.4f}")
    return total_loss / len(dataloader), metric.compute().item()


def train_and_validate(train_loader, val_loader, decoder, optimizer, criterion, metric, epochs, device, model_path):
    
    best_dev_miou = 0.0
    
    for epoch in range(epochs):
        print(f"\nEpoch {epoch+1}\n-------------------------------")
        
        train_loss, train_miou = train(train_loader, decoder, criterion, optimizer, metric, device)
        print(f"Train loss: {train_loss:.4f}, mIoU: {train_miou:.4f}")
        
        dev_loss, dev_miou = test(val_loader, decoder, criterion, metric, device, False, None)
        print(f"Dev loss: {dev_loss:.4f}, mIoU: {dev_miou:.4f}")
        
        # Save the best model based on dev mIoU
        if dev_miou > best_dev_miou:
            best_dev_miou = dev_miou
            torch.save(decoder.state_dict(), model_path)
            print(f"New best model saved at epoch {epoch+1} with dev mIoU: {best_dev_miou:.4f}")


# ======================================
# Main 
# ======================================
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. Encoder 
    os.environ["HF_HOME"] = "/home/zchen002/TFG/cache/ablation_study"
    os.environ["TRANSFORMERS_CACHE"] = "/home/zchen002/TFG/cache/ablation_study"
    cache_path = "/home/zchen002/TFG/cache/ablation_study"
    os.makedirs(cache_path, exist_ok=True)

    clip = CLIPModel.from_pretrained("openai/clip-vit-large-patch14", cache_dir=cache_path).to(device)
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14", cache_dir=cache_path, use_fast=True)
    
    for param in clip.parameters():
        param.requires_grad = False
    clip.eval()
    decoder = DPT3DModel(clip).to(device)
    
    # 3. Configuration
    BATCH_SIZE = 16 # 16
    EPOCHS = 10 # 10
    LR = 1e-3 # 1e-4
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(decoder.parameters(), LR)
    metric = BinaryJaccardIndex().to(device)

    # 4. Dataloaders
    split_path = "/home/zchen002/TFG/kodea/data_split.json"
    with open(split_path, 'r') as f:
        split_data = json.load(f)
    train_indices = split_data['train']
    val_indices = split_data['val']
    
    train_loader = get_dataloader(train_indices, "train", BATCH_SIZE, processor)
    val_loader = get_dataloader(val_indices, "validation", BATCH_SIZE, processor)
    test_loader = get_dataloader(None, "test", BATCH_SIZE, processor)
    
    # 5. Train
    model_path = f"/home/zchen002/TFG/trained_models/ablation_study/clip_large_avgpooling.pth"
    print(f"Starting training for CLIP, epochs: {EPOCHS}, batch size: {BATCH_SIZE}, learning rate: {LR}")
    train_and_validate(train_loader, val_loader, decoder, optimizer, criterion, metric, EPOCHS, device, model_path)
    
    # 6. Test
    decoder.load_state_dict(torch.load(model_path, map_location=device))
    predictions_path = f"/home/zchen002/TFG/iragarpenak/ablation_study/"
    os.makedirs(predictions_path, exist_ok=True)
    test(test_loader, decoder, criterion, metric, device, True, predictions_path)

