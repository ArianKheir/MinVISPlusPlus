import argparse
import torch
from detectron2.config import get_cfg
from detectron2.modeling import build_model
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.structures import ImageList
from PIL import Image
import numpy as np
import os
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns
from train_net_video import setup

def build_minvis_model(cfg_file, weights_path, device="cuda"):
    args = argparse.Namespace(
        config_file=cfg_file,
        opts=["MODEL.WEIGHTS", weights_path],  # pass weights via opts
        # the following are unused by setup(), but harmless to include
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
def forward_one_image(model, img_path):
    device = next(model.parameters()).device
    x = get_img(img_path, device)
    x = (x - model.pixel_mean) / model.pixel_std
    images = ImageList.from_tensors([x], model.size_divisibility)
    features = model.backbone(images.tensor)
    out = model.sem_seg_head(features)
    pred_logits = out["pred_logits"]
    pred_embds = out["pred_embds"]
    return pred_logits, pred_embds

def tsne_per_class(model ,pred_logits, pred_embds, out_dir,conf_thresh=0.5, perplexity=30, seed=42, dpi=300):
    os.makedirs(out_dir, exist_ok=True)
    logits = pred_logits[0] #no need for batch logits.shape is [t, q, c]
    embds = pred_embds[0].permute(1, 2, 0) #no need for batch embds.shape is [t, q, c]
    num_obj_classes = model.sem_seg_head.num_classes
    T, Q, C = logits.shape
    # X_list, labels = [], []
    # for t in range(T):
    #     E = embds[t]                          # [Q, C_embed]
    #     X_list.append(E.cpu().numpy())
    #     labels += [f"image_{t}"] * E.shape[0]
    # X = np.concatenate(X_list, axis=0)        # [T*Q, C_embed]

    # # t-SNE projection to 2D
    # tsne = TSNE(
    #     n_components=2,
    #     perplexity=min(perplexity, max(5, X.shape[0] - 1)),
    #     learning_rate="auto",
    #     init="pca",
    #     max_iter=1000,
    #     random_state=seed,
    # )
    # Y2 = tsne.fit_transform(X)                # [T*Q, 2]

    # # Plot, colored by frame id
    # plt.figure(figsize=(5, 5))
    # sns.scatterplot(x=Y2[:, 0], y=Y2[:, 1], hue=labels,
    #                 palette="tab10", s=14, linewidth=0, alpha=0.95)
    # plt.title("All queries (t-SNE)")
    # plt.axis("off")
    # plt.tight_layout()
    # plt.savefig(os.path.join(out_dir, "all_queries_tsne.png"),
    #             dpi=dpi, bbox_inches="tight", pad_inches=0.05)
    # plt.close()    
    # 1) Build the full set: all queries from both frames
    X_list, frame_labels = [], []
    for t in range(T):
        E = embds[t]                                 # [Q, C_embed]
        X_list.append(E.cpu().numpy())
        frame_labels += [f"image_{t}"] * E.shape[0]
    X = np.concatenate(X_list, axis=0)               # [T*Q, C_embed]
    classes_to_show = None
    # 2) Pick one best query per class per frame
    selected_classes = list(range(num_obj_classes)) if classes_to_show is None else list(classes_to_show)
    highlight_rows, highlight_frames, highlight_classes = [], [], []
    for t in range(T):
        cls_logits = logits[t, :, :num_obj_classes]  # [Q, num_obj_classes]
        best_q_per_cls = cls_logits.argmax(dim=0)    # [num_obj_classes]
        for c in selected_classes:
            q_idx = int(best_q_per_cls[c].item())
            row = t * Q + q_idx                      # row index in X / Y2
            highlight_rows.append(row)
            highlight_frames.append(f"image_{t}")
            highlight_classes.append(f"cls_{c}")

    # 3) t‑SNE on all queries
    tsne = TSNE(
        n_components=2,
        perplexity=min(perplexity, max(5, X.shape[0] - 1)),
        learning_rate="auto",
        init="pca",
        max_iter=1000,
        random_state=seed,
    )  # scikit-learn TSNE API
    Y2 = tsne.fit_transform(X)                       # [T*Q, 2]

    # 4) Plot: all queries (colored by frame), plus highlighted best-per-class points
    plt.figure(figsize=(6, 5))
    # base layer: all queries
    sns.scatterplot(x=Y2[:, 0], y=Y2[:, 1], hue=frame_labels,
                    palette="tab10", s=12, linewidth=0, alpha=0.35, legend=True)
    # overlay: highlighted top-per-class queries
    Y2_h = Y2[highlight_rows]
    sns.scatterplot(x=Y2_h[:, 0], y=Y2_h[:, 1],
                    hue=highlight_classes, style=highlight_frames,
                    palette="husl", s=70, linewidth=1.0, edgecolor="black", legend=True)
    plt.title("All queries + top per class per frame (t‑SNE)")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "all_queries_with_top_per_class_tsne.png"),
                dpi=dpi, bbox_inches="tight", pad_inches=0.05)  # savefig
    plt.close()
def tsne_per_class1(model, pred_logits, pred_embds, out_dir,
                                  p_thresh=0.03, classes_to_show=None,
                                  perplexity=30, seed=42, dpi=300):
    import os, numpy as np, matplotlib.pyplot as plt, seaborn as sns
    from sklearn.manifold import TSNE
    import torch

    os.makedirs(out_dir, exist_ok=True)

    logits = pred_logits[0]                           # [T, Q, C]
    embds  = pred_embds[0].permute(1, 2, 0)          # [T, Q, C_embed]
    T, Q, C = logits.shape
    num_obj_classes = model.sem_seg_head.num_classes  # exclude no-object

    X_list, objbg_labels = [], []
    for t in range(T):
        E = embds[t]                                  # [Q, C_embed]
        X_list.append(E.cpu().numpy())

        probs = torch.softmax(logits[t], dim=-1)      # [Q, C]
        top_probs, top_cls = probs.max(dim=-1)        # [Q], [Q]
        # object if: top class is among real classes and its prob >= threshold
        is_object = (top_cls < num_obj_classes) & (top_probs >= p_thresh)
        tags = ["obj" if bool(v) else "bg" for v in is_object]
        objbg_labels += tags

    X = np.concatenate(X_list, axis=0)                # [T*Q, C_embed]

    # Highlights (optional): only keep classes that pass threshold somewhere
    highlight_rows, highlight_frames, highlight_classes = [], [], []
    selected_classes = list(range(num_obj_classes)) if classes_to_show is None else list(classes_to_show)
    for t in range(T):
        probs = torch.softmax(logits[t, :, :num_obj_classes], dim=-1)  # [Q, K]
        for c in selected_classes:
            q_idx = int(probs[:, c].argmax().item())
            if float(probs[q_idx, c].item()) >= p_thresh:
                row = t * Q + q_idx
                highlight_rows.append(row)
                highlight_frames.append(f"image_{t}")
                highlight_classes.append(f"cls_{c}")

    tsne = TSNE(
        n_components=2,
        perplexity=min(perplexity, max(5, X.shape[0] - 1)),
        learning_rate="auto",
        init="pca",
        max_iter=1000,
        random_state=seed,
    )
    Y2 = tsne.fit_transform(X)

    plt.figure(figsize=(6, 5))
    # Base: hue by obj/bg after thresholding
    sns.scatterplot(x=Y2[:, 0], y=Y2[:, 1], hue=objbg_labels,
                    palette={"obj": "#1f77b4", "bg": "#ff7f0e"},
                    s=14, linewidth=0, alpha=0.35, legend=True)

    # Overlay: top-per-class per frame that also pass threshold
    if highlight_rows:
        Y2_h = Y2[highlight_rows]
        sns.scatterplot(x=Y2_h[:, 0], y=Y2_h[:, 1],
                        hue=highlight_classes, style=highlight_frames,
                        palette="husl", s=70, linewidth=1.0, edgecolor="black", legend=True)

    plt.title(f"All queries (obj vs bg, p≥{p_thresh}) + top per class")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"pca_all_queries_obj_vs_bg_p{int(p_thresh*100)}.png"),
                dpi=dpi, bbox_inches="tight", pad_inches=0.05)
    plt.close()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--weights", required=True)
    ap.add_argument("--img1", required=True)
    ap.add_argument("--img2", required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--conf", type=float, default=0.05)
    ap.add_argument("--perplexity", type=float, default=30)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    cfg, model = build_minvis_model(args.config, args.weights, device=device)

    logits1, embds1 = forward_one_image(model, args.img1)
    logits2, embds2 = forward_one_image(model, args.img2)
    logits = torch.cat([logits1, logits2], dim=1)
    embds  = torch.cat([embds1, embds2],  dim=2)
    tsne_per_class1(model, logits, embds, args.outdir, perplexity=args.perplexity, seed=args.seed)
if __name__ == "__main__":
    main()