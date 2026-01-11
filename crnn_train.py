# train_crnn.py
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.transforms as T
from crnn_model import CRNNDataset
from crnn_model import crnn_collate_fn, DOWNSAMPLE_RATIO
from crnn_model import CRNN, NUM_CLASSES
from crnn_model import BLANK_IDX, CHARS

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 64
EPOCHS_PRE = 8      # synthetic pretrain (can be more)
EPOCHS_FT = 40      # fine-tune on SVHN
LR_PRE = 1e-3
LR_FT = 2e-4

def train_epoch(model, loader, criterion, optimizer, scheduler):
    model.train()
    total_loss = 0.0
    for images, targets, input_lengths, target_lengths in loader:
        images = images.to(DEVICE)
        targets = targets.to(DEVICE)
        input_lengths = input_lengths.to(DEVICE)
        target_lengths = target_lengths.to(DEVICE)

        optimizer.zero_grad()
        log_probs = model(images)   # T x B x 11
        loss = criterion(log_probs, targets, input_lengths, target_lengths)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()
        scheduler.step()
        total_loss += loss.item() * images.size(0)

    return total_loss / len(loader.dataset)

@torch.no_grad()
def eval_epoch(model, loader, criterion):
    model.eval()
    total_loss = 0.0
    for images, targets, input_lengths, target_lengths in loader:
        images = images.to(DEVICE)
        targets = targets.to(DEVICE)
        input_lengths = input_lengths.to(DEVICE)
        target_lengths = target_lengths.to(DEVICE)

        log_probs = model(images)
        loss = criterion(log_probs, targets, input_lengths, target_lengths)
        total_loss += loss.item() * images.size(0)

    return total_loss / len(loader.dataset)

def main(pre_train = False):
    # 1) Build datasets
    # Pretrain dataset (synthetic) - OPTIONAL
    synth_root = "digit_synth"  # if you made this earlier
    svhn_root = "crnn_data"
    # train_aug = T.Compose([
    #     T.RandomAffine(
    #         degrees=7,  # small rotation
    #         translate=(0.05, 0.05),
    #         scale=(0.9, 1.1),
    #         shear=5
    #     ),
    #     T.ColorJitter(
    #         brightness=0.2,
    #         contrast=0.2
    #     ),
    #     T.GaussianBlur(kernel_size=(3,3), sigma=(0.1, 1.0))
    # ])

    train_aug = T.Compose([
        T.RandomApply([T.Grayscale(num_output_channels=3)], p=0.3),  # Force structure learning
        T.RandomAffine(
            degrees=10,  # Increased from 7
            translate=(0.1, 0.1),  # Increased from 0.05
            scale=(0.8, 1.2),  # Wider range
            shear=10  # Increased from 5
        ),
        T.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),  # More aggressive
        T.RandomApply([T.GaussianBlur(kernel_size=(3, 3), sigma=(0.1, 2.0))], p=0.3),
        T.ToTensor(),
        T.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        T.RandomErasing(p=0.3, scale=(0.02, 0.15))  # Crucial for occlusion robustness
    ])

    if pre_train:
        synth_train = CRNNDataset(synth_root, "train")
        synth_val = CRNNDataset(synth_root, "val")
        train_ds = synth_train
        val_ds   = synth_val
        LR = LR_PRE
        EPOCHS = EPOCHS_PRE
    else:
        train_ds = CRNNDataset(svhn_root, "train", augment=train_aug)
        val_ds = CRNNDataset(svhn_root, "val", augment=None)
        LR = LR_FT
        EPOCHS = EPOCHS_FT

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE,
                              shuffle=True, num_workers=4,
                              collate_fn=crnn_collate_fn)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE,
                            shuffle=False, num_workers=4,
                            collate_fn=crnn_collate_fn)

    model = CRNN(img_h=48, num_classes=NUM_CLASSES,fc_dropout=0.3, lstm_dropout=0.4).to(DEVICE)
    if not pre_train:
        with torch.no_grad():
            dummy_input = torch.randn(1, 3, 48, 100).to(DEVICE)
            _ = model(dummy_input)

        model.load_state_dict(torch.load("crnn_checkpoints/crnn_pretrain.pth"))

    criterion = nn.CTCLoss(blank=BLANK_IDX, zero_infinity=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    # Add Scheduler
    steps_per_epoch = len(train_loader)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=LR * 10,
        steps_per_epoch=len(train_loader),
        epochs=EPOCHS
    )
    best_val_loss = float("inf")
    # best_path = "crnn_checkpoints/crnn_best_svhn.pth"
    best_path = "crnn_checkpoints/crnn_ft_v5.pth"
    train_loss_list, val_loss_list = [], []
    for epoch in range(1, EPOCHS + 1):
        train_loss = train_epoch(model, train_loader, criterion, optimizer, scheduler)
        val_loss = eval_epoch(model, val_loader, criterion)
        print(f"[Epoch {epoch}] train {train_loss:.4f}  val {val_loss:.4f}")
        train_loss_list.append(train_loss)
        val_loss_list.append(val_loss)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), best_path)
            print(f"  -> new best saved to {best_path}")

    print("Done. Best val loss:", best_val_loss)
    print(train_loss_list, val_loss_list)

if __name__ == "__main__":
    main(pre_train=False)
