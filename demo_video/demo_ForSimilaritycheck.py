# Copyright (c) 2021-2022, NVIDIA Corporation & Affiliates. All rights reserved.
#
# This work is made available under the Nvidia Source Code License-NC.
# To view a copy of this license, visit
# https://github.com/NVlabs/MinVIS/blob/main/LICENSE


# Copyright (c) Facebook, Inc. and its affiliates.
# Modified by Bowen Cheng from: https://github.com/facebookresearch/detectron2/blob/master/demo/demo.py
import torch
from sklearn.metrics.pairwise import cosine_similarity
import argparse
import glob
import multiprocessing as mp
import os


# fmt: off
import sys
sys.path.insert(1, os.path.join(sys.path[0], '..'))
# fmt: on


import tempfile
import time
import warnings


import numpy as np
import tqdm


from torch.cuda.amp import autocast


from detectron2.config import get_cfg
from detectron2.data.detection_utils import read_image
from detectron2.projects.deeplab import add_deeplab_config
from detectron2.utils.logger import setup_logger


from mask2former import add_maskformer2_config
from mask2former_video import add_maskformer2_video_config
from minvis import add_minvis_config
from predictor import VideoPredictor, VisualizationDemo


import shutil


def setup_cfg(args):
    # load config from file and command-line arguments
    cfg = get_cfg()
    add_deeplab_config(cfg)
    add_maskformer2_config(cfg)
    add_maskformer2_video_config(cfg)
    add_minvis_config(cfg)
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    return cfg


def get_parser():
    parser = argparse.ArgumentParser(description="maskformer2 demo for builtin configs")
    parser.add_argument(
        "--config-file",
        default="configs/youtubevis_2019/video_maskformer2_R50_bs32_8ep_frame.yaml",
        metavar="FILE",
        help="path to config file",
    )
    parser.add_argument(
        "--input",
        help="directory of input video frames",
        required=True,
    )
    parser.add_argument(
        "--output",
        help="directory to save output frames",
        required=True,
    )
    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=0.5,
        help="Minimum score for instance predictions to be shown",
    )
    parser.add_argument(
        "--opts",
        help="Modify config options using the command-line 'KEY VALUE' pairs",
        default=[],
        nargs=argparse.REMAINDER,
    )
    return parser


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    args = get_parser().parse_args()
    setup_logger(name="fvcore")
    logger = setup_logger()
    logger.info("Arguments: " + str(args))

    cfg = setup_cfg(args)
    demo = VisualizationDemo(cfg)
    demo1 = VideoPredictor(cfg)

    assert args.input and args.output

    video_root = args.input
    output_root = args.output

    os.makedirs(output_root, exist_ok=True)
    
    frames_path = video_root
    frames_path = glob.glob(os.path.expanduser(os.path.join(frames_path, '*.jpg')))
    frames_path.sort()

    vid_frames = []
    for path in frames_path:
        img = read_image(path, format="BGR")
        vid_frames.append(img)

    start_time = time.time()
    with autocast():
        predictions = demo1(vid_frames)
    logger.info(
        "detected {} instances per frame in {:.2f}s".format(
            len(predictions["pred_scores"]), time.time() - start_time
        )
    )

    # --- EMBEDDING ANALYSIS (without t-SNE visualization) ---
    if "pred_embds" in predictions:
        embeds = predictions["pred_embds"]
        
        # Access the mask_embed module from the model
        model = demo1.model
        mask_embed_module = model.sem_seg_head.predictor.mask_embed
        device = next(mask_embed_module.parameters()).device  # Get the device of the model
        
        # Convert to tensor and move to correct device
        if not isinstance(embeds, torch.Tensor):
            embeds_tensor = torch.from_numpy(embeds).to(device)
        else:
            embeds_tensor = embeds.to(device)
        
        print(f"Original embeds shape: {embeds_tensor.shape}")
        print(f"Model device: {device}")
        
        # Apply mask_embed to the embeddings
        with torch.no_grad():
            # embeds_tensor is in shape [num_predictions, embed_dim]
            # mask_embed expects [batch, num_queries, embed_dim]
            # So we just add a batch dimension
            embeds_batched = embeds_tensor.unsqueeze(0)  # [1, num_predictions, embed_dim]
            
            # Apply mask_embed
            mask_embeds_tensor = mask_embed_module(embeds_batched)  # [1, num_predictions, mask_dim]
            
            # Remove batch dimension
            mask_embeds_tensor = mask_embeds_tensor.squeeze(0)  # [num_predictions, mask_dim]
        
        print(f"Mask embeds shape: {mask_embeds_tensor.shape}")
        
        # Convert to numpy for analysis
        if hasattr(embeds, 'numpy'):
            embeds = embeds.numpy()
        elif hasattr(embeds, 'cpu'): 
            embeds = embeds.cpu().numpy()
        else:
            embeds = embeds_tensor.cpu().numpy()
        
        # Convert mask embeddings to numpy
        mask_embeds = mask_embeds_tensor.cpu().numpy()
        
        labels = np.array(predictions["pred_labels"])
        scores = np.array(predictions["pred_scores"])
        frame_ids = np.array(predictions["frame_ids"])

        # Apply confidence threshold
        conf_threshold = 0.3
        valid_mask = scores >= conf_threshold
        
        embeds = embeds[valid_mask]
        labels = labels[valid_mask]
        scores = scores[valid_mask]
        frame_ids = frame_ids[valid_mask]
        mask_embeds = mask_embeds[valid_mask]

        print(f"\n=== Embedding Analysis Stats ===")
        print(f"Total predictions above {conf_threshold}: {len(labels)}")
        print(f"Unique classes: {np.unique(labels)}")
        print(f"Unique frames: {np.unique(frame_ids)}")
        print(f"Score range: [{scores.min():.3f}, {scores.max():.3f}]")
        print(f"Query embed shape: {embeds.shape}")
        print(f"Mask embed shape: {mask_embeds.shape}")
        
        if len(embeds) >= 2:
            unique_classes = np.unique(labels)
            unique_frames = np.unique(frame_ids)
            
            # Print per-class, per-frame breakdown
            print(f"\n=== Per-Class, Per-Frame Breakdown ===")
            for class_id in unique_classes:
                for frame_id in unique_frames:
                    count = np.sum((labels == class_id) & (frame_ids == frame_id))
                    if count > 0:
                        avg_score = scores[(labels == class_id) & (frame_ids == frame_id)].mean()
                        print(f"  Class {class_id}, Frame {frame_id}: {count} samples, avg score: {avg_score:.3f}")
            
            # Analyze similarity between query embeddings
            print(f"\n=== Query Embedding Cosine Similarities ===")
            pred_masks = np.array(predictions['pred_masks'])
            pred_masks = pred_masks[valid_mask]
            
            # Compare all pairs
            for i in range(len(embeds)):
                for j in range(i+1, len(embeds)):
                    emb1 = embeds[i]
                    emb2 = embeds[j]
                    
                    cos_sim = cosine_similarity([emb1], [emb2])[0, 0]
                    
                    # Determine relationship
                    if labels[i] == labels[j] and frame_ids[i] == frame_ids[j]:
                        relationship = "Same class, same frame"
                    elif labels[i] == labels[j]:
                        relationship = "Same class, different frames"
                    else:
                        relationship = "Different classes"

                    mask_emb1 = mask_embeds[i]
                    mask_emb2 = mask_embeds[j]
                    
                    mask_cos_sim = cosine_similarity([mask_emb1], [mask_emb2])[0, 0]

                    # Calculate mask difference and IoU
                    mask1 = pred_masks[i]
                    mask2 = pred_masks[j]
                    mask_diff = np.sum(mask1 != mask2)
                    intersection = np.logical_and(mask1, mask2).sum()
                    union = np.logical_or(mask1, mask2).sum()
                    iou = intersection / union if union > 0 else float('nan')
                    print(f"  C{labels[i]} F{frame_ids[i]} ↔ C{labels[j]} F{frame_ids[j]}: "
                          f"query_cosine={cos_sim:.4f}, {relationship}, "
                          f"mask_embed_cosine={mask_cos_sim:.4f}, "
                          f"mask_diff={mask_diff}, IoU={iou:.4f}")
        else:
            print("⚠ Not enough samples for similarity analysis (need at least 2)")