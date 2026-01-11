import math
import time
from collections import defaultdict
from ultralytics import YOLO
import torch
import torchvision.transforms as T
from PIL import Image, ImageOps
from full_pipe import rectify_crop, resize_keep_aspect
from crnn_model import CRNN, NUM_CLASSES, BLANK_IDX, idx2char
from ctc_decode import CHECKPOINT
import cv2, os

DET_WEIGHTS = "runs_det3/svhn_p3/weights/best.pt"
CRNN_WEIGHTS = CHECKPOINT
CONF_TH = 0.339   # from your F1–confidence tuning
IOU_TH  = 0.6
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
H_TARGET = 48
DOWNSAMPLE_RATIO = 4
test_dir = "test_video"
fourcc = cv2.VideoWriter_fourcc(*"mp4v")   # or "XVID"
max_keep_tracks = 1000


img_transform = T.Compose([
    T.ToTensor(),
    T.Normalize([0.5, 0.5, 0.5],
                [0.5, 0.5, 0.5]),
])

def pad_box(x0, y0, x1, y1, W, H, pad_frac=0.15):
    w = x1 - x0
    h = y1 - y0
    px, py = w * pad_frac, h * pad_frac
    x0 = max(0, x0 - px)
    y0 = max(0, y0 - py)
    x1 = min(W - 1, x1 + px)
    y1 = min(H - 1, y1 + py)
    return int(x0), int(y0), int(x1), int(y1)

def compute_iou(box_a, box_b):
    """
    box: (x0, y0, x1, y1)
    """
    x0a, y0a, x1a, y1a = box_a[:4]
    x0b, y0b, x1b, y1b = box_b[:4]

    inter_x0 = max(x0a, x0b)
    inter_y0 = max(y0a, y0b)
    inter_x1 = min(x1a, x1b)
    inter_y1 = min(y1a, y1b)

    if inter_x1 <= inter_x0 or inter_y1 <= inter_y0:
        return 0.0

    inter_area = (inter_x1 - inter_x0) * (inter_y1 - inter_y0)
    area_a = (x1a - x0a) * (y1a - y0a)
    area_b = (x1b - x0b) * (y1b - y0b)
    union = area_a + area_b - inter_area

    if union <= 0:
        return 0.0
    return inter_area / union

def fuse_track(history):
    """
    history: list of (text, det_score, rec_score) for this track.
      - det_score: YOLO confidence for that detection (0..1)
      - rec_score: CRNN score ~ mean log-prob along greedy path (log-space)

    We do a weighted majority vote over strings:
      weight = exp(rec_score)  (higher CRNN confidence = higher weight)

    For the winning string, we also compute:
      - fused_det_score: weighted avg YOLO confidence
      - fused_rec_conf:  exp(weighted avg rec_score)  (back to approx prob)
    """
    # text -> accumulators
    #   "w"   : sum of weights
    #   "det" : sum of w * det_score
    #   "rec" : sum of w * rec_score (still in log-space)
    accum = defaultdict(lambda: {"w": 0.0, "det": 0.0, "rec": 0.0})

    for text, det_score, rec_score in history:
        # rec_score is mean log-prob; exp(rec_score) ~ confidence in [0,1]
        w = math.exp(rec_score)
        d = accum[text]
        d["w"]   += w
        d["det"] += w * det_score
        d["rec"] += w * rec_score

    # pick the text with largest total weight
    best_text = None
    best_stats = None
    best_w = -1.0
    for text, stats in accum.items():
        if stats["w"] > best_w:
            best_w = stats["w"]
            best_text = text
            best_stats = stats

    if best_text is None:
        return "", 0.0, 0.0

    w_tot = best_stats["w"]
    fused_det_score = best_stats["det"] / w_tot           # weighted avg YOLO conf
    fused_rec_log   = best_stats["rec"] / w_tot           # weighted avg log-prob
    fused_rec_conf  = math.exp(fused_rec_log)             # back to [0,1]-ish prob

    return best_text, fused_det_score, fused_rec_conf

# track_id -> {
#   "last_box": (x0,y0,x1,y1),
#   "history":  [(text, score), ...],
#   "last_seen": frame_idx
# }

def assign_tracks(boxes, tracks, next_track_id, iou_thresh=0.5):
    """
    boxes: list of (x0,y0,x1,y1) for current frame.
    tracks: dict from tid -> track_info
    Returns: list of (box_idx, track_id).
    """
    # global next_track_id
    assignments = []
    used_tracks = set()

    for i, box in enumerate(boxes):
        best_iou = 0.0
        best_tid = None
        for tid, info in tracks.items():
            if tid in used_tracks:
                continue
            iou = compute_iou(box, info["last_box"])
            if iou > best_iou:
                best_iou = iou
                best_tid = tid

        if best_tid is not None and best_iou >= iou_thresh:
            assignments.append((i, best_tid))
            used_tracks.add(best_tid)
        else:
            # new track
            tid = next_track_id
            next_track_id += 1
            assignments.append((i, tid))

    return assignments, next_track_id

def recognize_patch_with_score(patch_pil, model, transform):
    """
    patch_pil: rectified & cropped PIL RGB image
    Returns: (text, score) where score ~ mean log-prob along greedy path.
    """
    patch_pil = resize_keep_aspect(patch_pil, H_TARGET)
    x = transform(patch_pil).unsqueeze(0).to(DEVICE)  # 1 x C x H x W
    W = x.shape[3]
    T_len = max(1, W // DOWNSAMPLE_RATIO)
    input_lengths = torch.tensor([T_len], dtype=torch.long, device=DEVICE)

    with torch.no_grad():
        log_probs = model(x)  # T x 1 x C
        # greedy path
        T_max, B, C = log_probs.shape
        preds = log_probs.argmax(dim=-1)[:, 0]  # T

        T_valid = min(T_len, T_max)
        lp_single = log_probs[:T_valid, 0, :]     # T_valid x C
        pred_indices = preds[:T_valid]           # T_valid
        path_log_probs = lp_single.gather(
            1, pred_indices.unsqueeze(1)
        ).squeeze(1)                             # T_valid

        score = float(path_log_probs.mean().item())

        # decode to string
        text = []
        prev = BLANK_IDX
        for t in range(T_valid):
            p = preds[t].item()
            if p == BLANK_IDX or p == prev:
                prev = p
                continue
            text.append(idx2char[p])
            prev = p
        text = "".join(text)

    return text, score

def process_video(video_path, det_model, rec_model, transform, output=None):
    tracks = {}  # tid -> {"last_box":..., "history":[(text, det_score, rec_score),...], "last_seen":frame_idx}
    next_track_id = 0

    filename = os.path.basename(video_path)
    output_file = os.path.join('test_output', filename)

    cap = cv2.VideoCapture(video_path)
    # --- FIX 1: DYNAMIC RESOLUTION & FPS ---
    # Fetch actual input video properties to prevent file corruption
    input_fps = cap.get(cv2.CAP_PROP_FPS)
    if input_fps == 0 or math.isnan(input_fps): input_fps = 30.0

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    out = cv2.VideoWriter(output_file, fourcc, 30, (width, height))

    frame_idx = 0
    written_frame =0
    frame_time = 0.0
    while True:
        ret, frame_bgr = cap.read()
        if not ret:
            print('Video has ended or failed, try a different video format!')
            break
        frame_idx += 1

        # convert to RGB for drawing + PIL for cropping
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        pil_frame = Image.fromarray(frame_rgb)
        W, H = pil_frame.size
        det_start_time = time.time()
        # ---------------- YOLO DETECTION ----------------
        det_res = det_model.predict(
            frame_rgb,
            imgsz=640,
            conf=CONF_TH,
            iou=IOU_TH,
            device=DEVICE,
            verbose=False
        )[0]
        det_end_time = time.time()
        det_time = det_end_time - det_start_time
        boxes = []
        if det_res.boxes is not None and len(det_res.boxes) > 0:
            for box in det_res.boxes:
                if int(box.cls[0].item()) != 0:
                    continue
                x0, y0, x1, y1 = box.xyxy[0].tolist()
                x0, y0, x1, y1 = pad_box(x0, y0, x1, y1, W, H, pad_frac=0.15)
                yolo_score = float(box.conf[0].item())
                boxes.append((x0, y0, x1, y1, yolo_score))

        total_rec_time = 0.0
        # ---------------- TRACKING + RECOG (only if we have boxes) ----------------
        if boxes:
            assignments, next_track_id = assign_tracks(boxes, tracks, next_track_id, iou_thresh=0.5)

            for (box_idx, tid) in assignments:
                x0, y0, x1, y1, det_score = boxes[box_idx]
                patch = pil_frame.crop((x0, y0, x1, y1))
                patch = rectify_crop(patch)
                rec_start_time = time.time()
                text, rec_score = recognize_patch_with_score(patch, rec_model, transform)
                rec_end_time = time.time()
                rec_time = rec_end_time - rec_start_time
                total_rec_time += rec_time
                if tid not in tracks:
                    tracks[tid] = {
                        "last_box": (x0, y0, x1, y1),
                        "history": [],
                        "last_seen": frame_idx,
                    }
                else:
                    tracks[tid]["last_box"] = (x0, y0, x1, y1)
                    tracks[tid]["last_seen"] = frame_idx

                # draw per-detection rectangle
                x0_d, y0_d, x1_d, y1_d = tracks[tid]["last_box"]
                cv2.rectangle(frame_rgb, (x0_d, y0_d), (x1_d, y1_d),
                              color=(0, 255, 0), thickness=5)

                tracks[tid]["history"].append((text, det_score, rec_score))

        # ---------------- TEMPORAL FUSION + OVERLAY ----------------
        cv2.namedWindow("Video", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Video", 1920, 1080)

        pred_house_num = None
        pred_det_score = None
        pred_rec_score = None

        tids_to_delete = []
        for tid, info in tracks.items():
            # fuse only if we have enough observations
            if len(info["history"]) >= 8:
                fused_text, fused_det, fused_rec = fuse_track(info["history"])
                pred_house_num = fused_text
                pred_det_score = round(fused_det, 3)
                pred_rec_score = round(fused_rec, 3)

            # clean up old tracks
            if frame_idx - info["last_seen"] > max_keep_tracks:
                tids_to_delete.append(tid)

        for tid in tids_to_delete:
            del tracks[tid]

        # Draw overlay only if we have at least one track
        if tracks:
            # use the first track's box as anchor for text
            first_tid = next(iter(tracks.keys()))
            x0_box, y0_box, x1_box, y1_box = tracks[first_tid]["last_box"]

            # if we have a fused prediction, show it
            if pred_house_num is not None:
                cv2.putText(
                    frame_rgb,
                    f'Prediction: {pred_house_num}',
                    (x0_box, max(0, y0_box - 10)),
                    fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=2,
                    color=(0, 0, 255),
                    thickness=4
                )
                cv2.putText(
                    frame_rgb,
                    f'Det Score: {pred_det_score}   Rec Score: {pred_rec_score}',
                    (x0_box, min(y1_box + 60, frame_rgb.shape[0] - 10)),
                    fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=2.0,
                    color=(0, 255, 255),
                    thickness=4
                )
        cv2.putText(frame_rgb, str(frame_idx), (50, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 4, (0, 0, 255), 3)

        # show (for debugging) and ALWAYS write this frame
        cv2.imshow("Video", frame_rgb)
        cv2.waitKey(1)

        frame_out = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
        # if frame_out.shape[1] != width or frame_out.shape[0] != height:
        #     frame_out = cv2.resize(frame_out, (width, height))
        # print(frame_out.shape)
        frame_time += (det_time + total_rec_time)
        out.write(frame_out)
        written_frame +=1

    print(f'Processed {written_frame} frames out of {frame_idx}')
    cap.release()
    out.release()
    cv2.destroyAllWindows()

    print(f'Average FPS: {written_frame / frame_time}')

    # ---------------- FINAL FUSED RESULTS PER TRACK ----------------
    final_results = {}
    for tid, info in tracks.items():
        if not info["history"]:
            continue
        fused_text, fused_det, fused_rec = fuse_track(info["history"])
        final_results[tid] = {
            "text": fused_text,
            "yolo_conf": fused_det,
            "crnn_conf": fused_rec,
        }

    return final_results



if __name__ == "__main__":
    # load CRNN
    crnn_model = CRNN(img_h=H_TARGET, num_classes=NUM_CLASSES).to(DEVICE)
    with torch.no_grad():
        dummy_input = torch.randn(1, 3, 48, 100).to(DEVICE)
        _ = crnn_model(dummy_input)
    num_params = sum(p.numel() for p in crnn_model.parameters() if p.requires_grad)
    print("Total trainable parameters:", num_params)
    # crnn_model.load_state_dict(torch.load(CRNN_WEIGHTS, map_location=DEVICE))
    # crnn_model.eval()
    # # load YOLO
    # detector = YOLO(DET_WEIGHTS)
    # for file in os.listdir(test_dir):
    #     # vid = os.path.join(test_dir, file)
    #     vid = "test_video/IMG_0388_4k30_cfr.mp4"
    # # vid = "output_4k30_cfr.mp4"
    #     results = process_video(vid, detector, crnn_model, img_transform)
    #     break

        # for tid, text in results.items():
        #     print(f"Track {tid}: final house number = {text}")





