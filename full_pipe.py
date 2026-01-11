from pathlib import Path
from PIL import Image, ImageOps
import torch
import os
import torchvision.transforms as T
from ultralytics import YOLO
from ctc_decode import ctc_greedy_decode, CHECKPOINT
from crnn_model import CRNN, NUM_CLASSES
from crnn_model import BLANK_IDX, idx2char

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

DET_WEIGHTS = "runs_det3/svhn_p3/weights/best.pt"
CRNN_WEIGHTS = CHECKPOINT

CONF_TH = 0.2   # from your F1–confidence tuning
IOU_TH  = 0.6
H_TARGET = 48
DOWNSAMPLE_RATIO = 4  # same as in collate


# ---------- rectification ----------

def rectify_crop(pil_img):
    """
    Rectification stub:
    - currently: just add a bit of extra border and center it.
    - if you implement a real rectifier (e.g., STN), you plug it here.
    """
    # add small uniform border (5% of height) to avoid digits touching edge
    h = pil_img.height
    pad = int(0.05 * h)
    if pad > 0:
        pil_img = ImageOps.expand(pil_img, border=pad, fill=pil_img.getpixel((0,0)))
    # For SVHN, we skip rotation/perspective; CRNN is already robust enough.
    return pil_img


def resize_keep_aspect(img, h_target=H_TARGET):
    W, H = img.size
    scale = h_target / H
    new_w = max(8, int(round(W * scale)))
    return img.resize((new_w, h_target), Image.BILINEAR)


def pad_box(x0, y0, x1, y1, W, H, pad_frac=0.15):
    w = x1 - x0
    h = y1 - y0
    px, py = w * pad_frac, h * pad_frac
    x0 = max(0, x0 - px)
    y0 = max(0, y0 - py)
    x1 = min(W - 1, x1 + px)
    y1 = min(H - 1, y1 + py)
    return int(x0), int(y0), int(x1), int(y1)


# load detector
det_model = YOLO(DET_WEIGHTS)

# load CRNN
crnn_model = CRNN(img_h=H_TARGET, num_classes=NUM_CLASSES).to(DEVICE)
with torch.no_grad():
    dummy_input = torch.randn(1, 3, 48, 100).to(DEVICE)
    _ = crnn_model(dummy_input)
state = torch.load(CRNN_WEIGHTS, map_location=DEVICE)
crnn_model.load_state_dict(state)
crnn_model.eval()

# same normalization as in training
img_transform = T.Compose([
    T.ToTensor(),
    T.Normalize([0.5, 0.5, 0.5],
                [0.5, 0.5, 0.5]),
])

def full_inference(image_path):
    """
    Returns a list of dicts:
      {'box': (x0,y0,x1,y1), 'conf': float, 'text': str}
    """
    im = Image.open(image_path).convert("RGB")
    W, H = im.size

    # 1) run detector
    det_output = det_model.predict(
        source=image_path,
        imgsz=640,
        conf=CONF_TH,
        iou=IOU_TH,
        device=DEVICE,
        verbose=False
    )[0]

    outputs = []

    if det_output.boxes is None or len(det_output.boxes) == 0:
        return outputs  # no signs found

    # 2) prepare crops for CRNN in a small batch
    crops = []
    boxes_out = []
    for box in det_output.boxes:
        cls_id = int(box.cls[0].item())
        conf = float(box.conf[0].item())
        if cls_id != 0:
            continue  # skip non-house_number (if you ever add more classes)

        x0, y0, x1, y1 = box.xyxy[0].tolist()
        x0, y0, x1, y1 = pad_box(x0, y0, x1, y1, W, H, pad_frac=0.15)
        patch = im.crop((x0, y0, x1, y1))

        # rectification step (currently light)
        patch = rectify_crop(patch)

        # resize to CRNN height
        patch = resize_keep_aspect(patch, H_TARGET)

        # remember original box & conf
        boxes_out.append(((x0, y0, x1, y1), conf))
        crops.append(patch)

    if not crops:
        return outputs

    # 3) make batch tensor for CRNN
    imgs_t = [img_transform(c) for c in crops]        # each: C x H x W_i
    widths = [t.shape[2] for t in imgs_t]
    max_w = max(widths)

    padded = []
    for t in imgs_t:
        _, _, w = t.shape
        if w < max_w:
            pad = (0, max_w - w, 0, 0)  # right pad
            t = torch.nn.functional.pad(t, pad, value=0.0)
        padded.append(t)
    batch = torch.stack(padded, dim=0).to(DEVICE)     # B x C x H x max_w

    # 4) compute input_lengths for CTC
    input_lengths = []
    for w in widths:
        T_i = w // DOWNSAMPLE_RATIO
        if T_i <= 0:
            T_i = 1
        input_lengths.append(T_i)
    input_lengths = torch.tensor(input_lengths, dtype=torch.long, device=DEVICE)

    # 5) run CRNN + decode
    with torch.no_grad():
        log_probs = crnn_model(batch)  # T x B x C
        texts = ctc_greedy_decode(log_probs, input_lengths)

    # 6) pack results
    for (box, conf), text in zip(boxes_out, texts):
        outputs.append({
            "box": box,
            "conf": conf,
            "text": text
        })

    return outputs

def calc_accuracy(pred, targets):
    # assert len(outputs) == len(targets)
    correct = 0
    for out, target in zip(pred, targets):
        if out == target:
            correct += 1
    return correct / len(pred)

if __name__ == "__main__":
    # img_path = "test_img.jpg"
    # outputs = full_inference(img_path)
    preds = []
    true = []
    type = 'train'
    img_dir = 'sign_level_boxes/svhn_sign_yolo/{}/images/'.format(type)
    label_dir = 'sign_level_boxes/svhn_sign_yolo/{}/seq_labels/'.format(type)
    # counts = 0
    for img in os.listdir(img_dir):
        # print(img)
        outputs = full_inference(img_dir+img)
        if not outputs:
            outputs = [{'text': '0'}]
        preds.append(outputs[0]['text'])
        # except IndexError:
        #     print(outputs)
        #     break
        # counts += 1
        # if counts >= 10:
        #     break
    # counts = 0
    for label in os.listdir(label_dir):
        # print(label)
        with open(label_dir+label, 'r') as f:
            lines = f.readlines()
            for line in lines:
                true.append(line.strip())

        # counts += 1
        # if counts >= 10:
        #     break
    acc = calc_accuracy(preds, true)
    print(acc)
    # print(preds)
    # print(true)