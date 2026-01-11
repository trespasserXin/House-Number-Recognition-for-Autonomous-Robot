from pathlib import Path
from PIL import Image
import numpy as np
from tqdm import tqdm
import h5py
import pickle

SRC = Path("svhn")              # same as before
OUT = Path("crnn_data")              # new folder
H_TARGET = 48                        # CRNN input height

# --- Reuse robust attr reader you already debugged ---
def _get_attr(f, bb, key):
    node = bb[key]
    data = node[()]
    import numpy as np, h5py
    if isinstance(data, np.ndarray) and data.dtype.kind == 'O':
        vals = []
        for i in range(node.shape[0]):
            ref = node[i][0]
            if isinstance(ref, h5py.Reference):
                ds = f[ref]
                v = ds[()]
                vals.append(int(np.array(v).squeeze()))
            else:
                vals.append(int(np.array(ref).squeeze()))
        return vals if len(vals) > 1 else vals[0]
    arr = np.array(data)
    if arr.ndim == 0:
        return int(arr)
    flat = arr.astype(np.int64).ravel().tolist()
    return flat if len(flat) > 1 else flat[0]

def read_digit_struct(mat_path):
    out = []
    with h5py.File(mat_path, "r") as f:
        DS = f["digitStruct"]
        names = DS["name"]
        bboxs = DS["bbox"]

        def _get_name(i):
            ref = names[i][0]
            return ''.join(chr(c[0]) for c in f[ref][:])

        for i in range(len(names)):
            name = _get_name(i)
            bb = f[bboxs[i][0]]
            left   = _get_attr(f, bb, 'left')
            top    = _get_attr(f, bb, 'top')
            width  = _get_attr(f, bb, 'width')
            height = _get_attr(f, bb, 'height')
            label  = _get_attr(f, bb, 'label')

            # normalize to lists
            if not isinstance(left, list):   left   = [left]
            if not isinstance(top, list):    top    = [top]
            if not isinstance(width, list):  width  = [width]
            if not isinstance(height, list): height = [height]
            if not isinstance(label, list):  label  = [label]

            boxes = []
            for l, t, w, h, lab in zip(left, top, width, height, label):
                boxes.append({
                    "left":   float(l),
                    "top":    float(t),
                    "width":  float(w),
                    "height": float(h),
                    "label":  int(lab),
                })
            out.append({"name": name, "boxes": boxes})
    return out

def union_box(boxes, W, H, pad_frac=0.15):
    x0 = min(b["l"] for b in boxes)
    y0 = min(b["t"] for b in boxes)
    x1 = max(b["l"]+b["w"] for b in boxes)
    y1 = max(b["t"]+b["h"] for b in boxes)
    pw = (x1-x0)*pad_frac
    ph = (y1-y0)*pad_frac
    x0 = max(0, x0-pw); y0 = max(0, y0-ph)
    x1 = min(W-1, x1+pw); y1 = min(H-1, y1+ph)
    return int(round(x0)), int(round(y0)), int(round(x1)), int(round(y1))

def build_transcript(boxes):
    # Sort digits left-to-right
    boxes_sorted = sorted(boxes, key=lambda b: b["l"])
    chars = []
    for b in boxes_sorted:
        # print(b)
        label = b["label"]
        if label == 10: label = 0   # SVHN encodes '0' as label 10
        chars.append(str(label))
    return "".join(chars)

def resize_keep_aspect(img, h_target):
    W, H = img.size
    scale = h_target / H
    new_w = max(8, int(round(W * scale)))
    img = img.resize((new_w, h_target), Image.BILINEAR)
    return img

def convert_split(split):
    src_dir = SRC/split/split
    # mat_path = src_dir/"digitStruct.mat"
    # records = read_digit_struct(mat_path)
    with open(f"svhn_crnn/svhn_boxes_crnn_{split}.pkl", "rb") as f:
        records = pickle.load(f)
        out_img_dir = OUT/split/"images"
        out_img_dir.mkdir(parents=True, exist_ok=True)
        labels_path = OUT/split/"labels.txt"
        f_labels = open(labels_path, "w", encoding="utf-8")

        for name, box_info in tqdm(records.items(), desc=f"CRNN {split}"):

            # img_path = src_dir/rec["name"]
            img_path = src_dir/name
            # print(img_path)
            if not img_path.exists():
                continue
            with Image.open(img_path).convert("RGB") as im:
                W, H = im.size
                boxes = box_info

                # skip weird/noisy cases
                if len(boxes) == 0:
                    continue
                # crop using union + padding
                x0, y0, x1, y1 = union_box(boxes, W, H, pad_frac=0.15)
                if (x1-x0) < 8 or (y1-y0) < 8:
                    continue
                crop = im.crop((x0, y0, x1, y1))
                crop = resize_keep_aspect(crop, H_TARGET)
                transcript = build_transcript(boxes)
                if not transcript:
                    continue

                fname = img_path.stem + ".png"
                out_path = out_img_dir/fname
                crop.save(out_path)
                f_labels.write(f"images/{fname}\t{transcript}\n")

        f_labels.close()

if __name__ == "__main__":
    for split in ["test", "train"]:  # later you’ll carve val out of train
        convert_split(split)
    print("Done ->", OUT)
