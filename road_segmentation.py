import torch
import numpy as np
import cv2
from PIL import Image
from transformers import SegformerImageProcessor, SegformerForSemanticSegmentation

# Load pretrained SegFormer trained on CityScapes
processor = SegformerImageProcessor.from_pretrained("nvidia/segformer-b0-finetuned-cityscapes-512-1024")
model = SegformerForSemanticSegmentation.from_pretrained("nvidia/segformer-b0-finetuned-cityscapes-512-1024")
model.eval()

# Cityscapes color map for key classes
LABEL_COLORS = {
    0:  (128, 64, 128),   # road — purple
    1:  (244, 35, 232),   # sidewalk — pink
    2:  (70, 70, 70),     # building — dark gray
    8:  (107, 142, 35),   # vegetation — green
    13: (0, 0, 142),      # car — dark blue
    11: (220, 20, 60),    # person — red
    17: (119, 11, 32),    # motorcycle
    18: (0, 0, 230),      # bicycle
}

def colorize(seg_map):
    h, w = seg_map.shape
    color_mask = np.zeros((h, w, 3), dtype=np.uint8)
    for label, color in LABEL_COLORS.items():
        color_mask[seg_map == label] = color
    return color_mask

def segment_frame(frame):
    image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    inputs = processor(images=image, return_tensors="pt")

    with torch.no_grad():
        outputs = model(**inputs)

    logits = outputs.logits  # (1, num_labels, H/4, W/4)
    upsampled = torch.nn.functional.interpolate(
        logits, size=frame.shape[:2], mode="bilinear", align_corners=False
    )
    seg_map = upsampled.argmax(dim=1).squeeze().cpu().numpy()
    return seg_map

def run(source="test_video.mp4"):
    cap = cv2.VideoCapture(source)
    frame_count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1

        # process every 2nd frame for speed
        if frame_count % 2 != 0:
            continue

        seg_map = segment_frame(frame)
        color_mask = colorize(seg_map)

        # blend original frame with segmentation mask
        overlay = cv2.addWeighted(frame, 0.6, color_mask, 0.4, 0)

        # legend
        cv2.putText(overlay, "Road", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (128, 64, 128), 2)
        cv2.putText(overlay, "Person", (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (220, 20, 60), 2)
        cv2.putText(overlay, "Car", (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 142), 2)
        cv2.putText(overlay, "Vegetation", (10, 105), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (107, 142, 35), 2)

        cv2.imshow("Road Segmentation", overlay)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    run(source="test_video.mp4")