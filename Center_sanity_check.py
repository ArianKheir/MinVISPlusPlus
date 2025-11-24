import torch
import matplotlib.pyplot as plt
from detectron2.utils.visualizer import Visualizer

# 1. Setup the Visualizer
# Convert image from BGR (OpenCV/Detectron2 default) to RGB
img_rgb = image[:, :, ::-1] 
v = Visualizer(img_rgb, scale=1.0)

# 2. Draw GT Masks (using overlay_instances for ground truth)
# We must move instances to CPU first
instances_cpu = instances.to("cpu")

vis_output = v.overlay_instances(
    masks=instances_cpu.gt_masks if instances_cpu.has("gt_masks") else None,
    boxes=instances_cpu.gt_boxes if instances_cpu.has("gt_boxes") else None, # Optional: draw boxes too
    alpha=0.5
)
vis_image = vis_output.get_image()

# 3. Process Centers
h, w = image.shape[:2]
# Un-normalize centers: normalized_centers * [w, h]
# Note: We use the original instances (gpu or cpu) to get the tensor, then convert to numpy
gt_centers = instances.gt_centers * torch.tensor([w, h], device=instances.gt_centers.device)
gt_centers_np = gt_centers.cpu().numpy()

# 4. Plot and Save
plt.figure(figsize=(14, 10))
plt.imshow(vis_image)

# Overlay centers as red dots
plt.scatter(gt_centers_np[:, 0], gt_centers_np[:, 1], c='red', s=50, label='GT Centers')
plt.legend()
plt.title("Augmented Image with GT Masks and Centers")
plt.axis("off") # Optional: remove axis ticks for a cleaner image

# Save to project root
plt.savefig("debug_vis_masks_centers(1).png", bbox_inches='tight', dpi=150)
plt.close()