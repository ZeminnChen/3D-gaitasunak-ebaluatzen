import os, json, torch, random, glob, natsort
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as T
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torchmetrics.classification import BinaryJaccardIndex  
from transformers import CLIPModel, CLIPProcessor


# ======================================
# Haziak konfiguratu
# ======================================
seed = 0
random.seed(seed)
np.random.seed(seed)             # numpy
torch.manual_seed(seed)          # cpu
torch.cuda.manual_seed(seed)     # gpu
torch.cuda.manual_seed_all(seed) # gpu guztientzat
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


# ======================================
# 1. Dataset processor
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
    def __init__(self, image_paths, voxel_paths, model_type, transform=None):
        self.image_paths = image_paths
        self.voxel_paths = voxel_paths
        self.model_type = model_type
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        with Image.open(self.image_paths[idx]) as img:
            image = img.convert("RGB")

        # 1. Padding + self.processor
        if self.transform is not None:
            image = self.transform(image)

        voxels = torch.from_numpy(np.load(self.voxel_paths[idx])).float()
        return image, voxels
# ======================================
# ======================================



# ======================================
# 3. Vision Encoder
# ======================================
    
class VisionEncoder(nn.Module):
    def __init__(self, model_type, size, device, cache_path):
        super().__init__()

        self.model_type = model_type
        self.size = size
        self.device = device
        self.cache_path = cache_path

        self.layer_config = {"base": [3, 6, 9, 12], "large": [5, 12, 18, 24],}
        self.registry = self._build_registry()

        self.model, self.processor, self.embed_dim, self.patch_size = self._load_model()
        self.model.to(device)
        self.model.eval()

        for p in self.model.parameters():
            p.requires_grad = False


    def _build_registry(self):
        return {
            "clip": {
                "base": "openai/clip-vit-base-patch32", #32
                "large": "openai/clip-vit-large-patch14",
                "model_cls": CLIPModel,
                "processor_cls": CLIPProcessor,
                "embed": lambda m: m.config.vision_config.hidden_size,
                "patch": lambda m: m.config.vision_config.patch_size,
            }
        }


    def _load_model(self):
        config = self.registry[self.model_type]
        model_name = config[self.size]
        
        model = config["model_cls"].from_pretrained(model_name, cache_dir=self.cache_path)
        processor = config["processor_cls"].from_pretrained(model_name, cache_dir=self.cache_path)

        embed_dim = config["embed"](model)
        patch_size = config["patch"](model)

        print(f"Loaded {self.model_type.upper()} {self.size} with embed_dim={embed_dim}, patch_size={patch_size}, extracted layers={self.layer_config[self.size]}")
        return model, processor, embed_dim, patch_size


    def forward(self, images):
        with torch.no_grad():
            
            target_layers = self.layer_config[self.size]

            if self.model_type == "clip":
                outputs = self.model.vision_model(pixel_values=images, output_hidden_states=True)
                return [outputs.hidden_states[i] for i in target_layers]

            raise ValueError(f"Unknown model type: {self.model_type}")
# ======================================
# ======================================



# ======================================
# 3. Decoder
# ======================================

# ======================================
# 3.1 Reassemble 
# ====================================== 
class Reassemble(nn.Module):
    def __init__(self, in_dim, out_dim=256, model_type=None):
        super().__init__()
        self.model_type = model_type
        
        self.proj_in = nn.Sequential(nn.Linear(2 * in_dim, in_dim), nn.GELU())
        self.proj_out = nn.Conv2d(in_dim, out_dim, kernel_size=1)

    def forward(self, x):
        
        cls_token = x[:, 0:1, :]
        tokens = x[:, 1:, :] 
        
        cls_repeated = cls_token.expand(-1, tokens.size(1), -1) 
        concat = torch.cat([tokens, cls_repeated], dim=-1)    
        x = self.proj_in(concat)  
        
        B, N, D = x.shape 
        H = W = int(N ** 0.5)
        x = x.view(B, H, W, D).permute(0, 3, 1, 2)  

        return self.proj_out(x)  


# ======================================
# 3.2 DPT Fusion
# ======================================
class DPTFusion(nn.Module):
    def __init__(self, dim=256):
        super().__init__()
        self.conv4 = nn.Conv2d(dim, dim, 3, padding=1)
        self.conv3 = nn.Conv2d(dim, dim, 3, padding=1)
        self.conv2 = nn.Conv2d(dim, dim, 3, padding=1)
        self.conv1 = nn.Conv2d(dim, dim, 3, padding=1)
        self.out_conv = nn.Conv2d(dim, dim, 3, padding=1)
        self.GELU = nn.GELU()

    def forward(self, feats):
        f1, f2, f3, f4 = feats
        x = self.conv4(f4)
        x = self.GELU(x + self.conv3(f3))
        x = self.GELU(x + self.conv2(f2))
        x = self.GELU(x + self.conv1(f1))
        return self.out_conv(x)

# ======================================
# 3.3 Lift to 3D
# ======================================
class LiftTo3D(nn.Module):
    def __init__(self, in_channels=256, target_voxels=4):
        super().__init__()
        self.in_channels = in_channels
        self.target_voxels = target_voxels
        
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
        B, _, H, W = x.shape
       
        # 1. Learned Depth Projection. [B, 256, H, W] -> [B, 1024, H, W]
        x = self.depth_projector(x)
       
        # 2. Reshape to 3D. [B, 256, 4, H, W]
        x = x.view(B, self.in_channels, self.target_voxels, H, W)
        x = self.adaptive_pool(x)  
       
       # 3. Refine in 3D space
        x = self.refine_3d(x)
       
        return x


# ======================================
# 3.4 DPT3DModel
# ======================================
class DPT3DModel(nn.Module):
    def __init__(self, embed_dim, model_type):  
        super().__init__()
        self.reassemble = nn.ModuleList([Reassemble(embed_dim, 256, model_type) for _ in range(4)])
        self.fusion = DPTFusion(256)
        self.lift3d = LiftTo3D(256)
        # 4 → 8 → 16 → 32 → 64
        self.decoder3d = nn.Sequential(
            nn.Sequential(nn.ConvTranspose3d(256, 128, 4, 2, 1), nn.BatchNorm3d(128), nn.GELU()),
            nn.Sequential(nn.ConvTranspose3d(128, 64, 4, 2, 1), nn.BatchNorm3d(64), nn.GELU()),
            nn.Sequential(nn.ConvTranspose3d(64, 32, 4, 2, 1),  nn.BatchNorm3d(32), nn.GELU()),
            nn.ConvTranspose3d(32, 1, 4, 2, 1)
        )
        
    def forward(self, layers):  # receives the 4 feature layers directly
        
        # 1. Reassemble
        feats = [reassemble(layer) for reassemble, layer in zip(self.reassemble, layers)]

        # 2. Multifeature fusion
        x = self.fusion(feats)  

        # 3. Lift 2D → 3D
        x = self.lift3d(x)      # (B, 256, 4, 4, 4)

        # 4. Decoder 3D → voxels
        x = self.decoder3d(x)   # (B, 1, 64, 64, 64)
        
        return x.squeeze(1)
# ======================================
# ======================================



# ======================================
# 4. Train + Test
# ======================================
def get_dataloader(indices, input_size, mode, batch_size, processor, model_type):
    base_path = "/home/zchen002/TFG"
    
    padding = ResizeWithPadding(size=input_size, border_size=10, top_factor=0.9, bottom_factor=1.2)
    
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

    if model_type == "clip":
        full_transform = lambda img: processor.image_processor(padding(img), return_tensors="pt", do_resize=False, do_center_crop=False).pixel_values.squeeze(0)
    else:
        raise ValueError(f"Unknown transform for model type: {model_type}")
    
    dataset = CLEVR3DDataset(image_paths, voxel_paths, model_type, full_transform)
    return DataLoader(dataset, batch_size=batch_size, shuffle=(mode == "train"))


## Train
def train(dataloader, encoder, decoder, criterion, optimizer, metric, device):
    decoder.train()
    metric.reset()
    
    total_loss = 0
    
    for images, voxels in dataloader:
        images, voxels = images.to(device), voxels.to(device)

        layers = encoder(images)        
        pred = decoder(layers)          
        loss = criterion(pred, voxels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        preds_bin = (torch.sigmoid(pred) > 0.5).int()
        metric.update(preds_bin, voxels.int())
        
    train_loss = total_loss / len(dataloader)
    
    return train_loss, metric.compute().item()


## Test
def build_num_objects_map(scenes_json_path):
    with open(scenes_json_path, 'r') as f:
        data = json.load(f)
    return {scene["image_index"]: len(scene["objects"]) for scene in data["scenes"]}

def test(dataloader, encoder, decoder, model_type, size, patch_size, device, criterion, metric, save_predictions, save_dir, num_objects_map=None):
    decoder.eval()
    metric.reset()

    all_preds = []
    per_obj_metrics = {}
    test_loss = 0.0

    with torch.no_grad():
        for batch_idx, (images, voxels) in enumerate(dataloader):
            images, voxels = images.to(device), voxels.to(device)

            layers = encoder(images)    
            pred = decoder(layers)      

            loss = criterion(pred, voxels)
            test_loss += loss.item()

            preds_bin = (torch.sigmoid(pred) > 0.5).int()
            metric.update(preds_bin, voxels.int())

            if num_objects_map is not None:
                for i in range(images.shape[0]):
                    global_idx = batch_idx * dataloader.batch_size + i
                    voxel_fname = os.path.basename(dataloader.dataset.voxel_paths[global_idx])
                    image_index = int(voxel_fname.split("_")[1].split(".")[0])
                    n_obj = num_objects_map.get(image_index, -1)

                    if n_obj not in per_obj_metrics:
                        per_obj_metrics[n_obj] = BinaryJaccardIndex().to(device)

                    per_obj_metrics[n_obj].update(
                        preds_bin[i].unsqueeze(0),
                        voxels[i].int().unsqueeze(0)
                    )

            if save_predictions:
                all_preds.append(pred.cpu())

    if save_predictions:
        all_preds = torch.cat(all_preds, dim=0)
        np.save(os.path.join(save_dir, f"test_{model_type}_{size}_{patch_size}.npy"), all_preds[:25].numpy())

    test_miou = metric.compute().item()
    total_loss = test_loss / len(dataloader)
    print(f"\nTest loss: {total_loss:.4f},   Test mIoU global: {test_miou:.4f}")

    if per_obj_metrics:
        print("\nmIoU per number of objects:")
        for n_obj in sorted(per_obj_metrics.keys()):
            print(f"  {n_obj} objects: {per_obj_metrics[n_obj].compute().item():.4f}")

    return total_loss, test_miou


def train_and_validate(train_loader, val_loader, encoder, decoder, model_type, size, optimizer, criterion, metric, epochs, device, model_path):
    best_dev_miou = 0.0
    
    for epoch in range(epochs):
        print(f"\nEpoch {epoch+1}\n-------------------------------")
        
        train_loss, train_miou = train(train_loader, encoder, decoder, criterion, optimizer, metric, device)
        print(f"Train loss: {train_loss:.4f}, mIoU: {train_miou:.4f}")
        
        dev_loss, dev_miou = test(val_loader, encoder, decoder, model_type, size, encoder.patch_size, device, criterion, metric, False, None)
        print(f"Dev loss: {dev_loss:.4f},  mIoU: {dev_miou:.4f}")
        
        if dev_miou > best_dev_miou:
            best_dev_miou = dev_miou
            torch.save(decoder.state_dict(), model_path)
            print(f"New best model saved at epoch {epoch+1} with dev mIoU: {best_dev_miou:.4f}")
# ======================================
# ======================================
            

# ======================================
# Main 
# ======================================
if __name__ == "__main__":
    model_type = "clip"  # "clip", "siglip2", "dinov2"
    size = "base"
    input_size = 224
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. Encoder 
    os.environ["HF_HOME"] = f"/home/zchen002/TFG/cache/{model_type}"
    os.environ["TRANSFORMERS_CACHE"] = f"/home/zchen002/TFG/cache/{model_type}"
    cache_path = f"/home/zchen002/TFG/cache/{model_type}"
    os.makedirs(cache_path, exist_ok=True)
    
    encoder = VisionEncoder(model_type, size, device, cache_path)
    processor = encoder.processor

    # 2. Decoder 
    decoder = DPT3DModel(encoder.embed_dim, model_type).to(device)
    
    # 3. Configuration
    BATCH_SIZE = 16 # 16
    EPOCHS = 10 # 10
    LR = 1e-3 # 1e-4
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(decoder.parameters(), LR)
    metric = BinaryJaccardIndex().to(device)
    
    # 4. Dataloaders
    split_path = "/home/zchen002/TFG/voxels/data_split.json"
    with open(split_path, 'r') as f:
        split_data = json.load(f)
    train_indices = split_data['train']
    val_indices = split_data['val']
    
    train_loader = get_dataloader(train_indices, input_size, "train", BATCH_SIZE, processor, model_type)
    val_loader = get_dataloader(val_indices, input_size, "validation", BATCH_SIZE, processor, model_type)
    test_loader = get_dataloader(None, input_size, "test", BATCH_SIZE, processor, model_type)
    
    # 5. Train and Validate
    model_path = "/home/zchen002/TFG/proba.pth"
    print(f"Starting training for {model_type.upper()}, epochs: {EPOCHS}, batch size: {BATCH_SIZE}, learning rate: {LR}")
    train_and_validate(train_loader, val_loader, encoder, decoder, model_type, size, optimizer, criterion, metric, EPOCHS, device, model_path)
    
    # 6. Test
    decoder.load_state_dict(torch.load(model_path, map_location=device))
    predictions_path = f"/home/zchen002/TFG/proba_iragarpenak/"
    os.makedirs(predictions_path, exist_ok=True)
    #num_objects_map = build_num_objects_map("/home/zchen002/TFG/CLEVR_images/scenes/CLEVR_val_scenes.json")
    test(test_loader, encoder, decoder, model_type, size, encoder.patch_size, device, criterion, metric, True, predictions_path, num_objects_map=None)