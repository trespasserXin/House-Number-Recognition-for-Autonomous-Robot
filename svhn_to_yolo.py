import h5py, numpy as np
from pathlib import Path
from PIL import Image
from tqdm import tqdm
import shutil
import pickle

SRC = Path("svhn")                       # raw
OUT = Path("sign_level_boxes/svhn_sign_yolo")         # yolo

def _get_name(f, names, i):
    ref = names[i][0]
    return ''.join(chr(c[0]) for c in f[ref][:])

def _get_attr(f, obj, key):
    """
        Read 'left'/'top'/'width'/'height' (and 'label' if you reuse this later)
        robustly from SVHN's digitStruct. Works whether the field is:
          - an array of object references, or
          - a numeric dataset (scalar or vector).
        Returns: int or list[int]
        """
    node = obj[key]  # h5py.Dataset
    data = node[()]  # load actual content (could be scalar, array, or object array)

    vals = []
    # node is typically shape (N, 1); iterate rows

    # Case B: plain numeric dataset (scalar or array)
    if data.dtype.kind != 'O':
        arr = np.array(data)
        if arr.ndim == 0:
            return int(arr)  # scalar
        # flatten e.g., shape (N,1) → (N,)
        flat = arr.astype(np.int64).ravel().tolist()
        return flat if len(flat) > 1 else flat[0]

    # Case A: object references (dtype=object). Each element is an h5py.Reference.
    for i in range(node.shape[0]):
        ref = node[i][0]
        if isinstance(ref, h5py.Reference):
            ds = f[ref]  # referenced dataset
            v = ds[()]  # often shape (1,1)
            # coerce to python int
            vals.append(int(np.array(v).squeeze()))
        else:
            # very rare: already numeric
            vals.append(int(np.array(ref).squeeze()))
    return vals if len(vals) > 1 else vals[0]

def read_digit_struct(mat_path):
    out = []
    with h5py.File(mat_path, "r") as f:
        DS = f["digitStruct"]
        names = DS["name"]
        bboxs = DS["bbox"]
        for i in range(len(names)):
            name = _get_name(f, names, i)
            bb   = f[bboxs[i][0]]
            left  = _get_attr(f, bb, "left")
            top   = _get_attr(f, bb, "top")
            width = _get_attr(f, bb, "width")
            height= _get_attr(f, bb, "height")
            # force lists
            if not isinstance(left, list):   left   = [left]
            if not isinstance(top, list):    top    = [top]
            if not isinstance(width, list):  width  = [width]
            if not isinstance(height, list): height = [height]
            boxes=[]
            for l,t,w,h in zip(left,top,width,height):
                boxes.append(dict(l=float(l), t=float(t), w=float(w), h=float(h)))
            out.append(dict(name=name, boxes=boxes))
    return out

def union_pad(boxes, W, H, pad_frac=0.15):
    x0 = min(b["l"] for b in boxes)
    y0 = min(b["t"] for b in boxes)
    x1 = max(b["l"]+b["w"] for b in boxes)
    y1 = max(b["t"]+b["h"] for b in boxes)
    pad_w = (x1 - x0) * pad_frac
    pad_h = (y1 - y0) * pad_frac
    x0 = max(0, x0 - pad_w); y0 = max(0, y0 - pad_h)
    x1 = min(W - 1, x1 + pad_w); y1 = min(H - 1, y1 + pad_h)
    return int(round(x0)), int(round(y0)), int(round(x1)), int(round(y1))

def write_yolo(lbl_path, x0,y0,x1,y1,W,H):
    xc = (x0+x1)/2.0/W; yc = (y0+y1)/2.0/H
    ww = (x1-x0)/W;      hh = (y1-y0)/H
    if ww <= 0 or hh <= 0:
        return False
    with open(lbl_path, "w") as f:
        f.write(f"0 {xc:.6f} {yc:.6f} {ww:.6f} {hh:.6f}\n")
    return True

def convert_split(split):
    src_dir = SRC/split/split
    out_img = OUT/split/"images"
    out_lbl = OUT/split/"seq_labels"
    out_img.mkdir(parents=True, exist_ok=True)
    out_lbl.mkdir(parents=True, exist_ok=True)

    # label_info = read_digit_struct(src_dir / "digitStruct.mat")
    with open('svhn_seq_label_{}.pkl'.format(split), 'rb') as f:
        boxes = pickle.load(f)
        # for rec in tqdm(label_info, desc=f"Convert {split}"):
        count = 0
        for name, box_info in tqdm(boxes.items(), desc=f"Convert {split}"):
            # img_path = src_dir/rec["name"]
            img_path = src_dir/name
            if not img_path.exists():
                continue
            # with Image.open(img_path) as im:
            #     W,H = im.size
            #     # x0,y0,x1,y1 = union_pad(rec["boxes"], W,H, pad_frac=0.15)
            #     x0,y0,x1,y1 = union_pad(box_info, W,H, pad_frac=0.15)
            #     # skip tiny boxes
            #     if (x1-x0) < 8 or (y1-y0) < 8:
            #         continue
            #     # copy image
            out_path = out_img/img_path.name
            if not out_path.exists():
                shutil.copy2(img_path, out_path)
            #     # write label
            # ok = write_yolo(out_lbl/(img_path.stem + ".txt"), x0,y0,x1,y1,W,H)
            # print(name, box_info)
            label = ''
            for box in box_info:
                if box['label'] == 10:
                    box['label'] = 0
                label += str(box['label'])
            # print(label)
            # seq = box_info[0]['boxes']
            label_path = out_lbl/(name.split('.')[0] + ".txt")
            # print(label_path)
            with open(label_path, "w") as f:
                f.write(label)
            # if not ok:
            #     (out_lbl/(img_path.stem + ".txt")).write_text("")  # empty -> treated as no object

if __name__ == "__main__":
    for sp in ["test"]:
        convert_split(sp)
    print("Done:", OUT)
