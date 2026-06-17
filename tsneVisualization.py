import argparse
import json
import os
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import pycocotools.mask as mask_util
from PIL import Image
from sklearn.manifold import TSNE
from scipy.optimize import linear_sum_assignment

from detectron2.modeling import build_model
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.structures import ImageList
from train_net_video import setup

def build_minvis_model(cfg_file, weights_path, device="cuda"):
    args = argparse.Namespace(
        config_file=cfg_file,
        opts=["MODEL.WEIGHTS", weights_path], 
        num_gpus=1, num_machines=1, machine_rank=0, dist_url="auto",
        eval_only=True, resume=False,
    )
    cfg = setup(args)
    model = build_model(cfg)
    DetectionCheckpointer(model).load(cfg.MODEL.WEIGHTS)
    model.to(device).eval()
    return cfg, model

def get_img(img_path, device):
    img = Image.open(img_path).convert("RGB")
    img = np.asarray(img).astype(np.float32) / 255.0
    img = torch.from_numpy(img).permute(2, 0, 1).to(device)
    return img

@torch.no_grad()
def forward_one_image(model, img_path, prev_queries=None, prev_scores=None):
    device = next(model.parameters()).device
    x = get_img(img_path, device)
    x = (x - model.pixel_mean) / model.pixel_std
    images = ImageList.from_tensors([x], model.size_divisibility)
    
    features = model.backbone(images.tensor)
    
    # 1. Pass the temporal states into the head
    out = model.sem_seg_head(features, mask=None, prev_queries=prev_queries, prev_scores=prev_scores)
    
    # Extract visualization data
    pred_logits = out["pred_logits"][0, 0]             
    pred_embds = out["pred_embds"][0, :, 0, :].permute(1, 0) 
    pred_masks = out["pred_masks"][0, :, 0, :, :] 
    
    original_size = (x.shape[1], x.shape[2])
    pred_masks = F.interpolate(
        pred_masks.unsqueeze(0), size=original_size, mode="bilinear", align_corners=False
    )[0]
    
    # 2. Extract and format the new states for the NEXT frame
    next_queries = out['final_queries'].detach() # Shape: [Q, 1, C]
    
    # Calculate max scores to serve as the gating mechanism for the next frame
    scores = F.softmax(out['pred_logits'], dim=-1)[..., :-1] # Shape: [1, 1, Q, K]
    max_scores, _ = scores.max(dim=-1)                       # Shape: [1, 1, Q]
    next_scores = max_scores.squeeze(1).detach()             # Shape: [1, Q]
    
    return pred_logits, pred_masks, pred_embds, next_queries, next_scores

def get_video_data(ann_file, img_dir, video_name):
    """Parses the YTVIS JSON to find all frames and annotations for a specific video."""
    with open(ann_file, 'r') as f:
        data = json.load(f)
        
    video_data = None
    for v in data['videos']:
        if video_name in v['file_names'][0]:
            video_data = v
            break
            
    if video_data is None:
        raise ValueError(f"Video '{video_name}' not found in {ann_file}")
        
    vid_id = video_data['id']
    image_paths = [os.path.join(img_dir, fn) for fn in video_data['file_names']]
    
    anns = [a for a in data['annotations'] if a['video_id'] == vid_id]
    
    return image_paths, anns

def get_gt_for_frame(anns, frame_idx, device):
    """Extracts GT classes, IDs, and decoded masks for a single frame."""
    gt_classes = []
    gt_ids = []
    gt_masks = []
    
    for ann in anns:
        seg = ann['segmentations'][frame_idx]
        if seg is not None:
            gt_classes.append(ann['category_id'])
            gt_ids.append(ann['id'])
            
            # Decode COCO RLE into a binary mask
            if isinstance(seg, dict):
                if isinstance(seg.get('counts'), list):
                    # FIX: Convert Uncompressed RLE (list) to Compressed RLE
                    h, w = seg['size']
                    rle = mask_util.frPyObjects([seg], h, w)[0]
                    mask = mask_util.decode(rle)
                else:
                    # Already Compressed RLE (string/bytes)
                    mask = mask_util.decode(seg)
            else:
                # Fallback for polygons
                h, w = ann.get('height', 720), ann.get('width', 1280)
                from pycocotools import mask as pycocomask
                rles = pycocomask.frPyObjects([seg], h, w)
                mask = pycocomask.decode(rles)[:, :, 0]
                
            gt_masks.append(mask)
            
    # Handle frames with zero GT objects
    if len(gt_classes) == 0:
        return torch.tensor([], dtype=torch.int64, device=device), \
               torch.tensor([], dtype=torch.int64, device=device), \
               torch.tensor([], dtype=torch.float32, device=device)
               
    gt_classes = torch.tensor(gt_classes, dtype=torch.int64, device=device)
    gt_ids = torch.tensor(gt_ids, dtype=torch.int64, device=device)
    gt_masks = torch.tensor(np.stack(gt_masks, axis=0), dtype=torch.float32, device=device) # [N, H, W]
    
    return gt_classes, gt_ids, gt_masks

def match_queries_to_gt(pred_logits, pred_masks, gt_classes, gt_masks):
    """
    Computes bipartite matching between queries and GT instances for a single frame
    using Class, Mask BCE, and Dice costs.
    """
    Q = pred_logits.shape[0]
    N = gt_classes.shape[0]
    
    if N == 0:
        return np.array([]), np.array([])
        
    # 1. Class Cost
    pred_probs = pred_logits.softmax(-1)
    cost_class = -pred_probs[:, gt_classes] # [Q, N]
    
    # 2. Flatten masks for cost computation
    pred_masks_flat = pred_masks.flatten(1) # [Q, H*W]
    gt_masks_flat = gt_masks.flatten(1).float() # [N, H*W]
    
    cost_mask = torch.zeros((Q, N), device=pred_logits.device)
    cost_dice = torch.zeros((Q, N), device=pred_logits.device)
    
    pred_masks_sig = pred_masks_flat.sigmoid()
    
    for i in range(N):
        gt_i = gt_masks_flat[i:i+1] # [1, HW]
        
        # Binary Cross Entropy Cost
        bce = F.binary_cross_entropy_with_logits(
            pred_masks_flat, gt_i.expand(Q, -1), reduction='none'
        ).mean(1)
        cost_mask[:, i] = bce
        
        # Dice Cost
        numerator = 2 * (pred_masks_sig * gt_i).sum(1)
        denominator = pred_masks_sig.sum(1) + gt_i.sum(1)
        dice = 1 - (numerator + 1) / (denominator + 1)
        cost_dice[:, i] = dice
        
    # Combine costs (standard Mask2Former weights: class=2.0, mask=5.0, dice=5.0)
    C = 2.0 * cost_class + 5.0 * cost_mask + 5.0 * cost_dice
    
    # Hungarian matching
    query_indices, gt_indices = linear_sum_assignment(C.cpu().numpy())
    return query_indices, gt_indices

def tsne_gt_anchored_queries(all_embds, all_gt_assignments, out_dir, perplexity=30, seed=42, dpi=300):
    os.makedirs(out_dir, exist_ok=True)
    
    T = len(all_embds)
    Q = all_embds[0].shape[0]
    X_full = torch.cat(all_embds, dim=0).cpu().numpy() # [T*Q, C]
    assignments_full = np.array(all_gt_assignments)
    
    # 1. CRITICAL: Filter out the background noise
    # We only want to plot queries that successfully matched to a Ground Truth object
    bg_mask = assignments_full == -1
    fg_mask = ~bg_mask
    
    X_filtered = X_full[fg_mask]
    assignments_filtered = assignments_full[fg_mask]
    
    if X_filtered.shape[0] == 0:
        print("No foreground queries found! Check your Hungarian Matcher.")
        return

    # 2. CRITICAL: L2 Normalize the features explicitly
    from sklearn.preprocessing import normalize
    X_norm = normalize(X_filtered, norm='l2', axis=1)

    print(f"Running t-SNE on {X_norm.shape[0]} matched foreground embeddings...")
    
    # 3. Optimized t-SNE parameters for Cosine embeddings
    # Perplexity must be scaled down since we removed the background queries
    dynamic_perplexity = min(perplexity, max(5, X_norm.shape[0] // 10))
    
    tsne = TSNE(
        n_components=2, 
        metric="cosine",            # Matches the MinVIS temporal bipartite matching metric
        perplexity=dynamic_perplexity, 
        learning_rate="auto", 
        init="random", 
        n_iter=1000,                # Reduced from 4000 to prevent overfitting to noise
        random_state=seed
    )
    
    Y = tsne.fit_transform(X_norm) # [N_matched, 2]
    
    # Create DataFrame only for the matched queries
    df = pd.DataFrame({
        'x': Y[:, 0],
        'y': Y[:, 1],
        'assigned_gt_id': assignments_filtered
    })
    
    plt.figure(figsize=(10, 8))
    
    unique_gt_ids = df['assigned_gt_id'].unique()
    cmap = plt.get_cmap('tab10')
    
    # Calculate a tiny jitter scale based on the t-SNE coordinate spread
    jitter_scale = (Y[:, 0].max() - Y[:, 0].min()) * 0.012
    
    for i, gt_id in enumerate(unique_gt_ids):
        obj_data = df[df['assigned_gt_id'] == gt_id]
        color = cmap(i % 10)
        
        # Apply Jitter
        x_plot = obj_data['x'] + np.random.normal(0, jitter_scale, len(obj_data))
        y_plot = obj_data['y'] + np.random.normal(0, jitter_scale, len(obj_data))
        
        # Plot the scattered points
        plt.scatter(x_plot, y_plot, 
                    color=color, s=90, alpha=0.75, edgecolors='white', linewidth=0.8,
                    label=f'Matches GT ID {gt_id}', zorder=3)

    plt.title("Queries Colored by Ground Truth ID Matching (t-SNE - Cosine Metric)")
    plt.axis("off")
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
    plt.tight_layout()
    
    out_path = os.path.join(out_dir, "gt_matched_queries_tsne_minvis.png")
    plt.savefig(out_path, dpi=dpi, bbox_inches="tight", pad_inches=0.05)
    plt.close()
    print(f"Visualization saved to {out_path}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--weights", required=True)
    ap.add_argument("--ann_file", required=True, help="Path to YTVIS instances.json")
    ap.add_argument("--img_dir", required=True, help="Path to YTVIS JPEGImages folder")
    ap.add_argument("--video", required=True, help="Folder name of the video (e.g., '00f88c4f0a')")
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--perplexity", type=float, default=30)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    cfg, model = build_minvis_model(args.config, args.weights, device=device)

    print(f"Loading Ground Truth for video: {args.video}")
    image_paths, video_anns = get_video_data(args.ann_file, args.img_dir, args.video)

    all_embds = []
    all_gt_assignments = []
    
    # Initialize temporal memory as None for the very first frame
    current_queries = None
    current_scores = None

    print(f"Processing {len(image_paths)} frames with Query Propagation ENABLED...")
    for t, img_path in enumerate(image_paths):
        # Pass the memory in, and catch the updated memory coming out
        pred_logits, pred_masks, pred_embds, current_queries, current_scores = forward_one_image(
            model, img_path, prev_queries=None, prev_scores=None
        )
        all_embds.append(pred_embds)
        
        gt_classes, gt_ids, gt_masks = get_gt_for_frame(video_anns, t, device)

        Q = pred_logits.shape[0]
        frame_assignments = np.full(Q, -1, dtype=int)
        
        q_indices, g_indices = match_queries_to_gt(pred_logits, pred_masks, gt_classes, gt_masks)
        
        for q_idx, g_idx in zip(q_indices, g_indices):
            frame_assignments[q_idx] = gt_ids[g_idx].item()
            
        all_gt_assignments.extend(frame_assignments.tolist())

    tsne_gt_anchored_queries(all_embds, all_gt_assignments, args.outdir, 
                             perplexity=args.perplexity, seed=args.seed)

if __name__ == "__main__":
    main()