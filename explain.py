"""Explainability (XAI) with Grad-CAM family methods."""
import numpy as np
import torch
from PIL import Image
from pytorch_grad_cam import GradCAM, GradCAMPlusPlus, LayerCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

from common import DEVICE, IMG_SIZE, eval_tf

METHODS = {"Grad-CAM": GradCAM, "Grad-CAM++": GradCAMPlusPlus, "LayerCAM": LayerCAM}


def predict_and_explain(model, pil_img, method="Grad-CAM", target_class=None):
    """Returns (probs, predicted idx, heatmap overlay RGB uint8, raw cam in [0,1])."""
    pil_img = pil_img.convert("RGB")
    x = eval_tf(pil_img).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        probs = torch.softmax(model(x), dim=1)[0].cpu().numpy()
    pred = int(probs.argmax())
    cls = pred if target_class is None else target_class

    rgb = np.array(pil_img.resize((IMG_SIZE, IMG_SIZE))).astype(np.float32) / 255.0
    target_layers = [model.layer4[-1]]
    with METHODS[method](model=model, target_layers=target_layers) as cam:
        cam_map = cam(input_tensor=x, targets=[ClassifierOutputTarget(cls)])[0]
    overlay = show_cam_on_image(rgb, cam_map, use_rgb=True)
    return probs, pred, overlay, cam_map