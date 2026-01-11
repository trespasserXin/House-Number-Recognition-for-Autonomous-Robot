import os
import shutil
from sklearn.model_selection import train_test_split

# ------------ PATHS ------------
ROOT = "crnn_data"                       # change if needed
train_imgs = os.path.join(ROOT, "train/images")
train_labels = os.path.join(ROOT, "train/labels.txt")

extra_imgs = os.path.join(ROOT, "extra/images")
extra_labels = os.path.join(ROOT, "extra/labels.txt")

# ------------ STEP 1: MERGE EXTRA → TRAIN ------------

# Read existing label lines
with open(train_labels, "r", encoding="utf-8") as f:
    train_lines = f.read().strip().split("\n")
    # print(train_lines[:5])

# with open(extra_labels, "r", encoding="utf-8") as f:
#     extra_lines = f.read().strip().split("\n")

# Move extra images to train/images
# for line in extra_lines:
#     img_line, _ = line.split("\t")
#     img_name = os.path.basename(img_line)
#     src = os.path.join(extra_imgs, img_name)
#     dst = os.path.join(train_imgs, img_name)
#
#     if os.path.exists(src):
#         shutil.move(src, dst)
#     else:
#         print(f"Warning: missing image {src}")
#         print(img_line)
#
# # Combine labels
# merged_lines = train_lines + extra_lines

# ------------ STEP 2: TRAIN/VAL SPLIT ------------

train_split, val_split = train_test_split(
    train_lines,
    test_size=0.1,          # 10% validation
    shuffle=True,
    random_state=42
)
print(train_split[:5])
# ------------ STEP 3: WRITE NEW LABEL FILES ------------

# Save updated train labels
with open(os.path.join(ROOT, "train/labels.txt"), "w", encoding="utf-8") as f:
    f.write("\n".join(train_split))
    # f.write("\n".join(["images/23187.png\t150", "images/3330.png\t643", "images/14626.png\t78"]))

# Create val folder
val_path = os.path.join(ROOT, "val")
val_img_path = os.path.join(val_path, "images")
os.makedirs(val_img_path, exist_ok=True)

# Write val labels
with open(os.path.join(val_path, "labels.txt"), "w", encoding="utf-8") as f:
    for line in val_split:
        img_line, label = line.split("\t")
        img_name = os.path.basename(img_line)

        src = os.path.join(train_imgs, img_name)
        dst = os.path.join(val_img_path, img_name)

        if os.path.exists(src):
            shutil.move(src, dst)
        else:
            print(f"Warning: missing image {src}")

        f.write(f"images/{img_name}\t{label}\n")

print("✔ Done! Extra merged → Train, Train split → Train + Val.")
