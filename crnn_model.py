# charset.py
from pathlib import Path
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms as T
import torch.nn as nn
import torch.nn.functional as F

CHARS = "0123456789"
BLANK_IDX = len(CHARS)  # 10
DOWNSAMPLE_RATIO = 4  # must match model's width downsampling
NUM_CLASSES = len(CHARS) + 1  # digits + blank


char2idx = {c: i for i, c in enumerate(CHARS)}
idx2char = {i: c for c, i in char2idx.items()}

def text_to_indices(text: str):
    return [char2idx[c] for c in text]  # only digits

def indices_to_text(idxs):
    return "".join(idx2char[i] for i in idxs)

def crnn_collate_fn(batch):
    """
    batch: list of (img_tensor, text)
      img_tensor: C x H x W_i (variable W)
      text: e.g. "174"
    """
    imgs, texts = zip(*batch)
    # Shapes & widths
    C, H = imgs[0].shape[:2]
    widths = [img.shape[2] for img in imgs]
    max_w = max(widths)

    # Pad images to max width
    padded = []
    for img in imgs:
        _, _, w = img.shape
        if w < max_w:
            pad_amount = max_w - w
            # pad right: (left, right, top, bottom)
            pad = (0, pad_amount, 0, 0)
            img = torch.nn.functional.pad(img, pad, value=0.0)
        padded.append(img)
    images = torch.stack(padded, dim=0)  # B x C x H x max_w

    # Targets & lengths
    targets = []
    target_lengths = []
    input_lengths = []

    for w, text in zip(widths, texts):
        idxs = text_to_indices(text)
        targets.extend(idxs)
        target_lengths.append(len(idxs))

        T_i = w // DOWNSAMPLE_RATIO  # after CNN width downsampling
        if T_i <= 0:
            T_i = 1
        input_lengths.append(T_i)

    targets = torch.tensor(targets, dtype=torch.long)          # sum(target_lengths)
    target_lengths = torch.tensor(target_lengths, dtype=torch.long)
    input_lengths = torch.tensor(input_lengths, dtype=torch.long)

    return images, targets, input_lengths, target_lengths

class CRNNDataset(Dataset):
    def __init__(self, root, split="train", height=48, augment=None):
        root = Path(root)
        self.img_dir = root / split / "images"
        labels_file = root / split / "labels.txt"
        self.height = height
        self.augment = augment
        self.samples = []
        with open(labels_file, "r", encoding="utf-8") as f:
            for line in f:
                path, text = line.strip().split("\t")
                self.samples.append((path, text))

        # same resize as during data prep (just ensure height=48 here)
        self.to_tensor = T.Compose([
            T.ToTensor(),                       # [0,1]
            T.Normalize([0.5,0.5,0.5],[0.5,0.5,0.5]),
        ])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        rel_path, text = self.samples[idx]
        img_path = self.img_dir / Path(rel_path).name
        img = Image.open(img_path).convert("RGB")

        # Assume already height=48 from preprocessing;
        # If not, you can enforce it here:
        #   W, H = img.size
        #   scale = self.height / H
        #   img = img.resize((int(W*scale), self.height), Image.BILINEAR)
        if self.augment is not None:
            img = self.augment(img)
            x = img
        else:
            x = self.to_tensor(img)     # C x H x W

        return x, text


class CRNN(nn.Module):
    def __init__(self, img_h=48, num_classes=NUM_CLASSES, hidden_size=256, num_lstm_layers=2,
                 fc_dropout=0.3, lstm_dropout=0.3):
        super().__init__()
        self.img_h = img_h

        # CNN backbone (similar to CRNN paper, simplified)
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 64, 3, 1, 1),   # 48 x W
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.MaxPool2d(2, 2),          # 24 x W/2

            nn.Conv2d(64, 128, 3, 1, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.MaxPool2d(2, 2),          # 12 x W/4

            nn.Conv2d(128, 256, 3, 1, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(True),

            nn.Conv2d(256, 256, 3, 1, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.MaxPool2d((2,1), (2,1)),  # 6 x W/4

            nn.Conv2d(256, 512, 3, 1, 1),
            nn.BatchNorm2d(512),
            nn.ReLU(True),

            nn.Conv2d(512, 512, 3, 1, 1),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            # no more pooling on width
        )
        self.downsample_ratio = 4  # must match collate

        # We'll flatten H dimension: feature_dim = 512 * H_out
        self.lstm_hidden = hidden_size
        # We don't know H_out at compile; derive at runtime via dummy forward
        self._feature_dim = None
        self.fc_dropout = nn.Dropout(fc_dropout)
        self.lstm_dropout = lstm_dropout
        self.lstm = None  # define lazily
        self.fc = None
        self.num_classes = num_classes

    def _build_lstm_if_needed(self, x):
        # x: B x C x H' x W' after CNN
        B, C, H, W = x.shape
        feat_dim = C * H
        if self._feature_dim is None:
            self._feature_dim = feat_dim
            self.lstm = nn.LSTM(
                input_size=feat_dim,
                hidden_size=self.lstm_hidden,
                num_layers=2,
                bidirectional=True,
                batch_first=False,  # we want T x B x F
                dropout=self.lstm_dropout
            )
            self.fc = nn.Linear(self.lstm_hidden * 2, self.num_classes)

            # move to same device as cnn
            self.lstm.to(x.device)
            self.fc.to(x.device)
            self.fc_dropout.to(x.device)

    def forward(self, images):
        """
        images: B x 3 x H x W
        returns: log_probs: T x B x C
        """
        x = self.cnn(images)  # B x C x H' x W'
        self._build_lstm_if_needed(x)

        B, C, H, W = x.shape
        # collapse H and C -> feature dim for each width position
        x = x.permute(0, 3, 1, 2).contiguous()   # B x W' x C x H'
        x = x.view(B, W, C * H)                  # B x W' x F
        x = x.permute(1, 0, 2)                   # T(=W') x B x F

        x, _ = self.lstm(x)                      # T x B x (2*hidden)
        x = self.fc_dropout(x)
        logits = self.fc(x)                      # T x B x num_classes
        log_probs = F.log_softmax(logits, dim=-1)
        return log_probs