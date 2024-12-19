"""
Adapted from Grounded SAM2 demo file
"""

import os
import cv2
import json
import torch
import numpy as np
import supervision as sv
import pycocotools.mask as mask_util
from pathlib import Path
from PIL import Image
import gsam2.grounding_dino.groundingdino.datasets.transforms as T
from torchvision.ops import box_convert
from gsam2.sam2.build_sam import build_sam2
from gsam2.sam2.sam2_image_predictor import SAM2ImagePredictor
from gsam2.grounding_dino.groundingdino.util.inference import load_model, predict

"""
Hyper parameters
"""
SAM2_CHECKPOINT = "assets/weights/sam2.1_hiera_large.pt"
SAM2_MODEL_CONFIG = "configs/sam2.1/sam2.1_hiera_l.yaml"
GROUNDING_DINO_CONFIG = "gsam2/grounding_dino/groundingdino/config/GroundingDINO_SwinT_OGC.py"
GROUNDING_DINO_CHECKPOINT = "assets/weights/groundingdino_swint_ogc.pth"
BOX_THRESHOLD = 0.35
TEXT_THRESHOLD = 0.25
OUTPUT_DIR = Path("outputs/")
# create output directory
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# environment settings
# use bfloat16

class GSAM2:

    def __init__(self, device = None):
        
        if device is not None:
            self.device = device
        else:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # build SAM2 image predictor
        self.sam2_model = build_sam2(SAM2_MODEL_CONFIG, SAM2_CHECKPOINT, device=self.device)
        self.sam2_predictor = SAM2ImagePredictor(self.sam2_model)

        # build grounding dino model
        self.grounding_model = load_model(
            model_config_path=GROUNDING_DINO_CONFIG, 
            model_checkpoint_path=GROUNDING_DINO_CHECKPOINT,
            device=self.device
        )
    
    def load_video_frame(self, video_path, frame = 0):
        
        transform = T.Compose(
            [
                T.RandomResize([800], max_size=1333),
                T.ToTensor(),
                T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise IOError(f"Cannot open video: {video_path}")
        
        # Set the video to the desired frame index
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame)
    
        ret, image = cap.read()
        if not ret:
            raise ValueError(f"Cannot read frame {frame}")
    
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_source = Image.fromarray(image)
        image_transformed, _ = transform(image_source, None)

        return image, image_transformed
    
    def get_masks(self, object_names, video_path, frame = 0):

        image_source, image = self.load_video_frame(video_path, frame)

        self.sam2_predictor.set_image(image_source)

        boxes, confidences, labels = predict(
            model=self.grounding_model,
            image=image,
            caption=object_names,
            box_threshold=BOX_THRESHOLD,
            text_threshold=TEXT_THRESHOLD,
        )

        h, w, _ = image_source.shape
        boxes = boxes * torch.Tensor([w, h, w, h])
        input_boxes = box_convert(boxes=boxes, in_fmt="cxcywh", out_fmt="xyxy").numpy()

        # FIXME: figure how does this influence the G-DINO model
        torch.autocast(device_type="cuda", dtype=torch.bfloat16).__enter__()

        if torch.cuda.get_device_properties(0).major >= 8:
            # turn on tfloat32 for Ampere GPUs (https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices)
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True

        masks, scores, logits = self.sam2_predictor.predict(
            point_coords=None,
            point_labels=None,
            box=input_boxes,
            multimask_output=False,
        )

        return masks, scores, logits, confidences, labels, input_boxes
    
    def visualize(self, video_path, masks, confidences, labels, input_boxes, frame = 0):

        """
        Post-process the output of the model to get the masks, scores, and logits for visualization
        """
        # convert the shape to (n, H, W)
        if masks.ndim == 4:
            masks = masks.squeeze(1)

        confidences = confidences.numpy().tolist()
        class_names = labels

        class_ids = np.array(list(range(len(class_names))))

        labels = [
            f"{class_name} {confidence:.2f}"
            for class_name, confidence
            in zip(class_names, confidences)
        ]

        """
        Visualize image with supervision useful API
        """
        # img = cv2.imread(img_path)
        # cv2.imwrite("img.jpg", img)

        # np.save("duck_mask.npy", masks[0])

        img, _ = self.load_video_frame(video_path, frame)

        detections = sv.Detections(
            xyxy=input_boxes,  # (n, 4)
            mask=masks.astype(bool),  # (n, h, w)
            class_id=class_ids
        )

        box_annotator = sv.BoxAnnotator()
        annotated_frame = box_annotator.annotate(scene=img.copy(), detections=detections)

        label_annotator = sv.LabelAnnotator()
        annotated_frame = label_annotator.annotate(scene=annotated_frame, detections=detections, labels=labels)
        annotated_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
        cv2.imwrite(os.path.join(OUTPUT_DIR, "groundingdino_annotated_image.jpg"), annotated_frame)

        mask_annotator = sv.MaskAnnotator()
        annotated_frame = mask_annotator.annotate(scene=annotated_frame, detections=detections)
        cv2.imwrite(os.path.join(OUTPUT_DIR, "grounded_sam2_annotated_image_with_mask.jpg"), annotated_frame)