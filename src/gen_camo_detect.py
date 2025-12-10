import argparse
import os
from pathlib import Path
from tqdm import tqdm

import cv2
import numpy as np
import torch
import torch.nn as nn
import yaml
import neural_renderer
from Image_Segmentation.network import U_Net

from models.experimental import attempt_load
from utils.datasets_RAUCA import create_dataloader
from utils.datasets import letterbox
from utils.general import non_max_suppression, scale_coords, set_logging
from utils.plots import plot_one_box
from utils.torch_utils import select_device


def cal_texture(texture_param, texture_origin, texture_mask, texture_content=None, content=False):
    """Utility copied from training script to compose adversarial texture."""
    if content:
        textures = 0.5 * (torch.nn.Tanh()(texture_content) + 1)
    else:
        textures = 0.5 * (torch.nn.Tanh()(texture_param) + 1)
    return texture_origin * (1 - texture_mask) + texture_mask * textures


def load_texture_and_assets(args, device):
    texture_param = torch.from_numpy(np.load(args.textures)).to(device)
    if texture_param.ndim == 5:
        # add batch dim if user saved without leading 1
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




def run_detection(im0, model, device, stride, imgsz, conf_thres, iou_thres, names, colors):
    """Runs YOLOv3 inference on a single BGR image and returns annotated BGR image and max confidence."""
    img = letterbox(im0, imgsz, stride=stride)[0]
    img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3xHxW
    img = np.ascontiguousarray(img)
    img = torch.from_numpy(img).to(device)
    img = img.float() / 255.0
    if img.ndimension() == 3:
        img = img.unsqueeze(0)

    pred = model(img, augment=False)[0]
    pred = non_max_suppression(pred, conf_thres, iou_thres, classes=None, agnostic=False)

    max_conf = 0.0
    det = pred[0]
    if len(det):
        det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()
        for *xyxy, conf, cls in reversed(det):
            # Track max confidence for 'car' (index 2)
            if int(cls) == 2:
                if float(conf) > max_conf:
                    max_conf = float(conf)
            label = f"{names[int(cls)]} {conf:.2f}"
            plot_one_box(xyxy, im0, label=label, color=colors[int(cls)], line_thickness=2)
    return im0, max_conf


def main(args):
    set_logging()
    device = select_device(args.device, batch_size=1)

    # Load textures and assets
    textures, faces, vertices = load_texture_and_assets(args, device)

    # Load dataset definition
    with open(args.data) as f:
        data_dict = yaml.safe_load(f)
    train_path = data_dict["train"]
    if not os.path.isabs(train_path):
        train_path = os.path.join(data_dict.get("path", ""), train_path)

    # Build dataloader (batch size 1 is enough for export)
    dummy_opt = argparse.Namespace(single_cls=False, cache_images=False, workers=args.workers, world_size=1)
    dataloader, dataset = create_dataloader(
        train_path,
        args.img_size,
        batch_size=1,
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
        prefix="gen: ",
        mask_dir=os.path.join(args.datapath, "masks"),
        ret_mask=True,
        phase="test" if "test_new" in train_path else "training",
    )
    dataset.set_textures(textures)

    # Load NSR model for weather/lighting rendering
    model_nsr = U_Net()
    saved_state_dict = torch.load(args.nsr_weights, map_location=device)
    new_state_dict = {}
    for k, v in saved_state_dict.items():
        name = k[7:] if k.startswith("module.") else k
        new_state_dict[name] = v
    model_nsr.load_state_dict(new_state_dict)
    model_nsr.to(device)
    model_nsr.eval()

    # Load detector
    model = attempt_load(args.det_weights, map_location=device)
    # Patch old checkpoints that lack recompute_scale_factor attr
    for m in model.modules():
        if isinstance(m, nn.Upsample) and not hasattr(m, "recompute_scale_factor"):
            m.recompute_scale_factor = None
    stride = int(model.stride.max())
    names = model.module.names if hasattr(model, "module") else model.names
    colors = [[int(x) for x in np.random.randint(0, 255, 3)] for _ in names]
    model.eval()

    save_dir = Path(args.output)
    save_dir.mkdir(parents=True, exist_ok=True)

    processed = 0
    total_conf_orig = 0.0
    total_conf_camo = 0.0
    detected_orig = 0
    detected_camo = 0
    
    # Store per-sample metrics
    metrics_list = []

    print(f"Start processing {len(dataloader)} batches...")

    for imgs, texture_img, masks, imgs_cut, _, paths, _ in dataloader:
        # Save original image path copy
        orig_path = paths[0]
        base = Path(orig_path).stem
        im0_orig = cv2.imread(orig_path)
        if im0_orig is None:
            continue

        with torch.no_grad():
            imgs = imgs.to(device).float()
            texture_img = texture_img.to(device).float()
            masks = masks.to(device).float()
            imgs_cut = imgs_cut.to(device).float() / 255.0

            out_tensor = torch.sigmoid(model_nsr(imgs_cut))
            tensor1 = out_tensor[:, 0:3, :, :]
            tensor2 = out_tensor[:, 3:6, :, :]
            tensor3 = torch.clamp(texture_img * tensor1 + tensor2, max=1)

            masks_rep = masks.unsqueeze(1).repeat(1, 3, 1, 1)
            imgs_nsr = (1 - masks_rep) * imgs + (255 * tensor3) * masks_rep

        camo = imgs_nsr[0].permute(1, 2, 0).cpu().numpy()
        camo = np.clip(camo, 0, 255).astype(np.uint8)
        im0_camo = cv2.cvtColor(camo, cv2.COLOR_RGB2BGR)

        # Run detection
        det_orig, conf_orig = run_detection(im0_orig.copy(), model, device, stride, args.img_size, args.conf_thres,
                                 args.iou_thres, names, colors)
        det_camo, conf_camo = run_detection(im0_camo.copy(), model, device, stride, args.img_size, args.conf_thres,
                                 args.iou_thres, names, colors)

        if not args.no_save:
            cv2.imwrite(str(save_dir / f"{base}_orig_det.jpg"), det_orig)
            cv2.imwrite(str(save_dir / f"{base}_camo_det.jpg"), det_camo)
        
        # Determine detection (assuming threshold 0.5 for 'detected')
        # args.conf_thres is NMS threshold, but for ASR usually we just say did the detector output a box?
        # conf_orig > 0 means a box was found (because run_detection updates max_conf only if box found)
        # However, run_detection uses args.conf_thres for NMS. So if any box survived NMS, it's a detection.
        
        is_det_orig = conf_orig > 0
        is_det_camo = conf_camo > 0
        
        metrics_list.append({
            "id": base,
            "conf_orig": conf_orig,
            "conf_camo": conf_camo,
            "detected_orig": is_det_orig,
            "detected_camo": is_det_camo
        })

        total_conf_orig += conf_orig
        total_conf_camo += conf_camo
        if is_det_orig: detected_orig += 1
        if is_det_camo: detected_camo += 1

        processed += 1
        if args.max_samples and processed >= args.max_samples:
            break

    # Calculate aggregate metrics
    asr = 0.0
    if detected_orig > 0:
        # ASR = (Detected Original - Detected Camo) / Detected Original ???
        # Or standard ASR: Proportion of samples where attack succeeded (i.e. detector failed on camo but worked on orig, or just failed on camo)
        # Usually ASR = 1 - (Detected Camo / Total)
        # Let's count "Success" as: Originally detected, but Camo NOT detected.
        
        success_count = 0
        valid_targets = 0
        for m in metrics_list:
            if m["detected_orig"]:
                valid_targets += 1
                if not m["detected_camo"]:
                    success_count += 1
        
        if valid_targets > 0:
            asr = success_count / valid_targets
        else:
            asr = 0.0 # No valid targets found to attack
    
    avg_conf_orig = total_conf_orig / processed if processed else 0
    avg_conf_camo = total_conf_camo / processed if processed else 0
    
    metrics = {
        "total_samples": processed,
        "valid_targets": detected_orig,
        "detected_camo_count": detected_camo,
        "asr": asr,
        "avg_conf_orig": avg_conf_orig,
        "avg_conf_camo": avg_conf_camo,
        "details": metrics_list
    }
    
    import json
    with open(save_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    if not args.no_save:
        print(f"Saved {processed * 2} detection images to {save_dir}")
    print(f"Metrics saved to {save_dir}/metrics.json (ASR={asr:.2%})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate camouflaged samples and YOLO detections.")
    parser.add_argument("--textures", type=str, required=True, help="Path to trained texture .npy")
    parser.add_argument("--data", type=str, default="data/carla.yaml", help="data yaml path")
    parser.add_argument("--datapath", type=str, required=True, help="dataset root containing masks/")
    parser.add_argument("--obj_file", type=str, default="car_assets/audi_et_te.obj", help="3D car model obj")
    parser.add_argument("--faces", type=str, default="car_assets/exterior_face.txt", help="face id file")
    parser.add_argument("--texture_size", type=int, default=6, help="texture grid size")
    parser.add_argument("--img_size", type=int, default=640, help="detection image size")
    parser.add_argument("--det_weights", type=str, default="weights/yolov3_9_5.pt", help="detector weights")
    parser.add_argument("--nsr_weights", type=str, default="NRP_weights/NRP_weight.pth", help="NSR U-Net weights")
    parser.add_argument("--device", type=str, default="", help="cuda device or cpu")
    parser.add_argument("--output", type=str, default="compare_out", help="output folder")
    parser.add_argument("--workers", type=int, default=2, help="dataloader workers")
    parser.add_argument("--conf_thres", type=float, default=0.25, help="confidence threshold")
    parser.add_argument("--iou_thres", type=float, default=0.45, help="NMS IoU threshold")
    parser.add_argument("--max_samples", type=int, default=0, help="limit number of samples (0 means all)")
    parser.add_argument("--no_save", action="store_true", help="Do not save detection images (metrics only)")
    args = parser.parse_args()
    main(args)
