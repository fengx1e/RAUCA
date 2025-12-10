import argparse
import os
import shutil
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import yaml
import neural_renderer
from Image_Segmentation.network import U_Net

from models.experimental import attempt_load
from utils.datasets_RAUCA import create_dataloader, LoadImagesAndLabels
from utils.datasets import letterbox
from utils.general import non_max_suppression, scale_coords, set_logging
from utils.torch_utils import select_device

def cal_texture(texture_param, texture_origin, texture_mask, texture_content=None, content=False):
    if content:
        textures = 0.5 * (torch.nn.Tanh()(texture_content) + 1)
    else:
        textures = 0.5 * (torch.nn.Tanh()(texture_param) + 1)
    return texture_origin * (1 - texture_mask) + texture_mask * textures

def load_texture_and_assets(texture_path, args, device):
    """Load a specific texture file and return the rendered texture tensor."""
    if not os.path.exists(texture_path):
        raise FileNotFoundError(f"Texture not found: {texture_path}")
        
    texture_param = torch.from_numpy(np.load(texture_path)).to(device)
    if texture_param.ndim == 5:
        texture_param = texture_param.unsqueeze(0)
        
    vertices, faces, texture_origin = neural_renderer.load_obj(
        filename_obj=args.obj_file, texture_size=args.texture_size, load_texture=True
    )
    vertices = vertices.to(device)
    faces = faces.to(device)
    texture_origin = texture_origin.to(device)

    texture_mask = np.zeros((faces.shape[0], args.texture_size, args.texture_size, args.texture_size, 3), "int8")
    with open(args.faces, "r") as f:
        for face_id in f:
            if face_id.strip():
                texture_mask[int(face_id) - 1, ...] = 1
    texture_mask = torch.from_numpy(texture_mask).to(device).unsqueeze(0)

    textures = cal_texture(texture_param, texture_origin, texture_mask)
    return textures, faces, vertices

@torch.no_grad()
def get_max_conf_for_class(im0, model, device, stride, imgsz, target_cls_idx=2):
    img = letterbox(im0, imgsz, stride=stride)[0]
    img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3xHxW
    img = np.ascontiguousarray(img)
    img = torch.from_numpy(img).to(device)
    img = img.float() / 255.0
    if img.ndimension() == 3:
        img = img.unsqueeze(0)

    pred = model(img, augment=False)[0]
    pred = non_max_suppression(pred, conf_thres=0.1, iou_thres=0.45, classes=None, agnostic=False)

    max_conf = 0.0
    det = pred[0]
    if len(det):
        for *xyxy, conf, cls in det:
            if int(cls) == target_cls_idx:
                if conf > max_conf:
                    max_conf = float(conf)
    
    return max_conf

def main(args):
    set_logging()
    device = select_device(args.device, batch_size=1)

    print(f"Loading Texture 1: {args.texture1}")
    textures1, faces, vertices = load_texture_and_assets(args.texture1, args, device)
    
    print(f"Loading Texture 2: {args.texture2}")
    textures2, _, _ = load_texture_and_assets(args.texture2, args, device)

    # Load dataset definition
    with open(args.data) as f:
        data_dict = yaml.safe_load(f)
    
    image_dir = os.path.join(args.datapath, "train_new") 
    if not os.path.exists(image_dir):
        image_dir = os.path.join(args.datapath, "train")
    
    print(f"Scanning images from: {image_dir}")

    # Build Dataset directly (not DataLoader)
    dummy_opt = argparse.Namespace(single_cls=False, cache_images=False, workers=args.workers, world_size=1)
    
    # We use create_dataloader to get the dataset instance cleanly
    _, dataset = create_dataloader(
        image_dir,
        args.img_size,
        batch_size=1, # Not relevant for direct access
        stride=32,
        faces=faces,
        texture_size=args.texture_size,
        vertices=vertices,
        opt=dummy_opt,
        hyp=None,
        augment=False,
        cache=False,
        pad=0.0,
        rect=False,
        rank=-1,
        world_size=1,
        workers=args.workers,
        prefix="filter: ",
        mask_dir=os.path.join(args.datapath, "masks"),
        ret_mask=True,
        phase="training", 
    )

    # Load NSR model
    model_nsr = U_Net()
    saved_state_dict = torch.load(args.nsr_weights, map_location=device)
    new_state_dict = {}
    for k, v in saved_state_dict.items():
        name = k[7:] if k.startswith("module.") else k
        new_state_dict[name] = v
    model_nsr.load_state_dict(new_state_dict)
    model_nsr.to(device)
    model_nsr.eval()

    # Load Detector
    model = attempt_load(args.det_weights, map_location=device)
    # Patch old checkpoints
    for m in model.modules():
        if isinstance(m, nn.Upsample) and not hasattr(m, "recompute_scale_factor"):
            m.recompute_scale_factor = None
            
    stride = int(model.stride.max())
    stride = int(model.stride.max())
    model.eval()
    print(f"DEBUG: Model loaded successfully: {type(model)}")

    # Output Setup
    out_root = Path(args.output_datapath)
    dirs_to_make = ["masks", "test", "test_new", "test_label_new"]
    for d in dirs_to_make:
        (out_root / d).mkdir(parents=True, exist_ok=True)
    
    import re
    import random
    
    # -----------------------------
    # BUCKETED SAMPLING LOGIC
    # -----------------------------
    
    # Regex to find integer in filename "data123.jpg" -> 123
    id_pattern = re.compile(r"data(\d+)")
    
    print("Organizing images into buckets of 1000 IDs...")
    buckets = {} # Key: bucket_index (0 for 1-1000), Value: list of indices
    
    # Filter dataset.img_files to only those matching pattern and organize
    valid_indices = []
    
    for i, filepath in enumerate(dataset.img_files):
        filename = Path(filepath).name
        match = id_pattern.search(filename)
        if match:
            file_id = int(match.group(1))
            # Bucket 0: 1-1000, Bucket 1: 1001-2000
            if file_id == 0: continue # Skip 0 if exists, usually 1-based
            bucket_idx = (file_id - 1) // 1000
            
            if bucket_idx not in buckets:
                buckets[bucket_idx] = []
            buckets[bucket_idx].append(i)
        else:
            # warning or skip
            pass

    print(f"Created {len(buckets)} buckets.")
    
    total_saved = 0
    sorted_bucket_keys = sorted(buckets.keys())
    
    for b_key in sorted_bucket_keys:
        indices = buckets[b_key]
        random.shuffle(indices) # Randomize check order
        
        print(f"Processing Bucket {b_key} (IDs {b_key*1000 + 1}-{b_key*1000 + 1000}). Candidates: {len(indices)}")
        
        bucket_saved_count = 0
        
        for idx in indices:
            if bucket_saved_count >= 10:
                print(f"  -> Reached 10 valid samples for Bucket {b_key}. Moving to next.")
                break
                
            # Process Item
            try:
                # 1. Pipeline with Texture 1
                dataset.set_textures(textures1)
                item1 = dataset[idx] 
                tex1_2d = item1[1].to(device).unsqueeze(0) 
                mask = item1[2].to(device).unsqueeze(0)
                img_cut = item1[3].to(device).unsqueeze(0)
                imgs = item1[0].to(device).unsqueeze(0) 
                
                path = item1[5]
                filename = Path(path).name
                
                im0_orig = cv2.imread(path)
                if im0_orig is None:
                    continue
        
                # Check Original Confidence
                conf_orig = get_max_conf_for_class(im0_orig.copy(), model, device, stride, args.img_size, target_cls_idx=2)
                if conf_orig <= 0.7:
                    continue
        
                # 2. Pipeline with Texture 2
                dataset.set_textures(textures2)
                item2 = dataset[idx]
                tex2_2d = item2[1].to(device).unsqueeze(0)
                
                # Render and Check
                with torch.no_grad():
                    img_cut_norm = img_cut / 255.0
                    out_tensor = torch.sigmoid(model_nsr(img_cut_norm))
                    tensor1 = out_tensor[:, 0:3, :, :]
                    tensor2 = out_tensor[:, 3:6, :, :]
                    
                    masks_rep = mask.unsqueeze(1).repeat(1, 3, 1, 1).float()
                    
                    def render_check(tex_tensor_2d):
                        tensor3 = torch.clamp(tex_tensor_2d * tensor1 + tensor2, max=1)
                        imgs_nsr = (1 - masks_rep) * imgs + (255 * tensor3) * masks_rep
                        camo_np = imgs_nsr[0].permute(1, 2, 0).cpu().numpy()
                        im_camo = cv2.cvtColor(np.clip(camo_np, 0, 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
                        return im_camo
        
                    # Check Texture 1
                    im_camo1 = render_check(tex1_2d)
                    conf_camo1 = get_max_conf_for_class(im_camo1, model, device, stride, args.img_size, target_cls_idx=2)
                    if conf_camo1 > 0.5:
                        continue
        
                    # Check Texture 2
                    im_camo2 = render_check(tex2_2d)
                    conf_camo2 = get_max_conf_for_class(im_camo2, model, device, stride, args.img_size, target_cls_idx=2)
                    if conf_camo2 > 0.5:
                        continue
                        
                # Success - Save
                print(f"  Matched {filename}: Orig={conf_orig:.2f}, Tex1={conf_camo1:.2f}, Tex2={conf_camo2:.2f}")
                bucket_saved_count += 1
                total_saved += 1
                
                # 1. Copy Image (.png) -> test_new
                shutil.copy(path, out_root / "test_new" / filename)
                
                # 2. Copy NPZ (.npz) -> test
                # Assumes source npz is in datapath/train/data123.npz
                npz_name = Path(filename).with_suffix('.npz').name
                source_npz = Path(args.datapath) / "train" / npz_name
                if source_npz.exists():
                    shutil.copy(source_npz, out_root / "test" / npz_name)
                else:
                     print(f"Warning: NPZ not found for {filename}")

                # 3. Copy Mask (.png) -> masks
                mask_name_png = Path(filename).with_suffix('.png').name
                mask_path_png = Path(args.datapath) / "masks" / mask_name_png
                
                if mask_path_png.exists():
                    shutil.copy(mask_path_png, out_root / "masks" / mask_name_png)
                else:
                    print(f"Warning: Mask not found for {filename}")
                    
                # 4. Copy Label (.txt) -> test_label_new
                label_filename = Path(filename).with_suffix('.txt').name
                label_path = Path(args.datapath) / "train_label_new" / label_filename
                if label_path.exists():
                    shutil.copy(label_path, out_root / "test_label_new" / label_filename)
            except Exception as e:
                import traceback
                traceback.print_exc()
                print(f"Error processing {idx}: {e}")
                continue

    print(f"Done. Saved {total_saved} valid samples across all buckets.")
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--texture1", type=str, required=True)
    parser.add_argument("--texture2", type=str, required=True)
    parser.add_argument("--datapath", type=str, required=True)
    parser.add_argument("--output_datapath", type=str, default="data/test_dataset")
    
    parser.add_argument("--data", type=str, default="data/carla.yaml")
    parser.add_argument("--obj_file", type=str, default="car_assets/audi_et_te.obj")
    parser.add_argument("--faces", type=str, default="car_assets/exterior_face.txt")
    parser.add_argument("--texture_size", type=int, default=6)
    parser.add_argument("--img_size", type=int, default=640)
    parser.add_argument("--det_weights", type=str, default="weights/yolov3_9_5.pt")
    parser.add_argument("--nsr_weights", type=str, default="NRP_weights/NRP_weight.pth")
    parser.add_argument("--device", type=str, default="0")
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--max_samples", type=int, default=0)

    args = parser.parse_args()
    main(args)
