from pathlib import Path
import random, shutil
from tqdm import tqdm

ROOT = Path("sign_level_backup")
SPLIT_RATIO = 0.20
SEED = 42

def collect(split):
    imgs = sorted((ROOT/split/"images").glob("*.*"))
    valids = []
    for img in imgs:
        if (ROOT/split/"labels"/(img.stem + ".txt")).exists():
            valids.append(img)
    return valids

def move_subset(src_split, dst_split, files):
    (ROOT/dst_split/"images").mkdir(parents=True, exist_ok=True)
    (ROOT/dst_split/"labels").mkdir(parents=True, exist_ok=True)
    for file in tqdm(files, desc="Move {}->{}".format(src_split, dst_split)):
        label = ROOT / src_split / "labels" / (file.stem + ".txt")
        shutil.move(str(file), str(ROOT / dst_split / "images" / file.name))
        shutil.move(str(label), str(ROOT / dst_split / "labels" / label.name))

if __name__ == "__main__":
    random.seed(SEED)

    # merge extra -> train
    extra_imgs = collect("extra")
    for img in extra_imgs:
        dst = ROOT /"train" /"images" / img.name
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(img), str(dst))
        label = ROOT / "extra" / "labels" / (img.stem + ".txt")
        shutil.move(str(label), str(ROOT / "train" / "labels" / label.name))

    # now split val from train
    train_imgs = collect("train")
    n_val = int(len(train_imgs) * SPLIT_RATIO)
    val_pick = set(random.sample(train_imgs, n_val))
    move_subset("train","val", list(val_pick))
    print("Done. train/ val/ test ready.")
