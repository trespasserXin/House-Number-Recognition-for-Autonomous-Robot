from torch.utils.data import DataLoader
import torch
import math
from crnn_model import BLANK_IDX, idx2char
from crnn_model import CRNN, CRNNDataset
from crnn_model import DOWNSAMPLE_RATIO

DATA_ROOT = "crnn_data"
CHECKPOINT = "crnn_checkpoints/crnn_ft_v5.pth"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
IMG_HEIGHT = 48
NUM_CLASSES = len(idx2char) + 1

def crnn_collate_fn_eval(batch):
    """
    For evaluation: pad images to same width, keep raw texts.
    Returns:
      images: B x C x H x W_max
      texts:  list[str] length B
      input_lengths: tensor[B] of valid time steps after CNN
    """
    imgs, texts = zip(*batch)
    C, H = imgs[0].shape[:2]
    widths = [img.shape[2] for img in imgs]
    max_w = max(widths)

    padded = []
    for img in imgs:
        _, _, w = img.shape
        if w < max_w:
            pad_amount = max_w - w
            # pad right: (left, right, top, bottom)
            pad = (0, pad_amount, 0, 0)
            img = torch.nn.functional.pad(img, pad, value=0.0)
        padded.append(img)

    images = torch.stack(padded, dim=0)  # B x C x H x W_max

    input_lengths = []
    for w in widths:
        T = w // DOWNSAMPLE_RATIO
        if T <= 0:
            T = 1
        input_lengths.append(T)
    input_lengths = torch.tensor(input_lengths, dtype=torch.long)

    return images, list(texts), input_lengths

def ctc_greedy_decode(log_probs, input_lengths):
    """
    log_probs: T x B x C (output of CRNN, log_softmax)
    input_lengths: tensor[B] number of valid time steps for each sample
    Returns: list of decoded strings length B
    """
    T, B, C = log_probs.shape
    preds = log_probs.argmax(dim=-1)  # T x B
    texts = []

    for b in range(B):
        prev = BLANK_IDX
        seq = []
        T_b = int(input_lengths[b].item())
        T_b = min(T_b, T)  # safety

        for t in range(T_b):
            p = preds[t, b].item()
            if p == BLANK_IDX or p == prev:
                prev = p
                continue
            seq.append(idx2char[p])
            prev = p

        texts.append("".join(seq))
    return texts


def log_sum_exp(a, b):
    if a == -math.inf:
        return b
    if b == -math.inf:
        return a
    if a > b:
        return a + math.log1p(math.exp(b - a))
    else:
        return b + math.log1p(math.exp(a - b))

def score_prefix(prefix, lp_b, lp_nb, alpha=0.5):
    """Length-normalized score for a prefix."""
    s = log_sum_exp(lp_b, lp_nb)
    L = max(1, len(prefix))
    return s / (L ** alpha)

def ctc_beam_search_single(log_probs_t_c, beam_size=5, alpha=0.5):
    """
    Beam search for a single sequence.
    log_probs_t_c: T x C (log-softmax)
    """
    T, C = log_probs_t_c.shape
    blank = BLANK_IDX

    # prefix -> (log_p_blank, log_p_nonblank)
    beam = {(): (0.0, -math.inf)}  # empty prefix at t=-1

    for t in range(T):
        next_beam = {}
        logp_t = log_probs_t_c[t]  # C

        for prefix, (lp_b, lp_nb) in beam.items():
            # 1) stay/go blank
            p_blank_t = logp_t[blank].item()
            lp_b_new = log_sum_exp(lp_b + p_blank_t, lp_nb + p_blank_t)
            prev_b, prev_nb = next_beam.get(prefix, (-math.inf, -math.inf))
            next_beam[prefix] = (log_sum_exp(prev_b, lp_b_new), prev_nb)

            # 2) extend with non-blank labels
            for c in range(C):
                if c == blank:
                    continue
                p_c_t = logp_t[c].item()

                if len(prefix) > 0 and c == prefix[-1]:
                    # same symbol as last: stay in same prefix, from blank only
                    new_prefix = prefix
                    lp_nb_new = lp_b + p_c_t
                    prev_b2, prev_nb2 = next_beam.get(new_prefix, (-math.inf, -math.inf))
                    next_beam[new_prefix] = (prev_b2, log_sum_exp(prev_nb2, lp_nb_new))
                else:
                    # different symbol: extend prefix
                    new_prefix = prefix + (c,)
                    lp_nb_new = log_sum_exp(lp_b + p_c_t, lp_nb + p_c_t)
                    prev_b2, prev_nb2 = next_beam.get(new_prefix, (-math.inf, -math.inf))
                    next_beam[new_prefix] = (prev_b2, log_sum_exp(prev_nb2, lp_nb_new))

        # prune
        sorted_beam = sorted(
            next_beam.items(),
            key=lambda kv: score_prefix(kv[0], kv[1][0], kv[1][1], alpha=alpha),
            reverse=True,
        )
        beam = dict(sorted_beam[:beam_size])

    # pick best final prefix
    best_prefix = None
    best_score = -math.inf
    for prefix, (lp_b, lp_nb) in beam.items():
        s = score_prefix(prefix, lp_b, lp_nb, alpha=alpha)
        if s > best_score:
            best_score = s
            best_prefix = prefix

    if best_prefix is None:
        return ""

    return "".join(idx2char[i] for i in best_prefix)

def ctc_beam_search_decode(log_probs, input_lengths, beam_size=5, alpha=0.5):
    T, B, C = log_probs.shape
    texts = []
    for b in range(B):
        T_b = int(input_lengths[b].item())
        T_b = min(T_b, T)
        lp = log_probs[:T_b, b, :]  # T_b x C
        text = ctc_beam_search_single(lp, beam_size=beam_size, alpha=alpha)
        texts.append(text)
    return texts

def levenshtein(a, b):
    """Simple Levenshtein distance (edit distance) for CER."""
    la, lb = len(a), len(b)
    if la == 0:
        return lb
    if lb == 0:
        return la
    dp = [[0] * (lb + 1) for _ in range(la + 1)]
    for i in range(la + 1):
        dp[i][0] = i
    for j in range(lb + 1):
        dp[0][j] = j
    for i in range(1, la + 1):
        for j in range(1, lb + 1):
            cost = 0 if a[i-1] == b[j-1] else 1
            dp[i][j] = min(
                dp[i-1][j] + 1,      # deletion
                dp[i][j-1] + 1,      # insertion
                dp[i-1][j-1] + cost  # substitution
            )
    return dp[la][lb]

@torch.no_grad()
def evaluate():
    # 1) Dataset + loader
    test_ds = CRNNDataset(DATA_ROOT, split="train", augment=None)
    test_loader = DataLoader(
        test_ds,
        batch_size=64,
        shuffle=False,
        num_workers=4,
        collate_fn=crnn_collate_fn_eval,
    )

    # 2) Model
    model = CRNN(img_h=IMG_HEIGHT, num_classes=NUM_CLASSES).to(DEVICE)
    with torch.no_grad():
        dummy_input = torch.randn(1, 3, 48, 100).to(DEVICE)
        _ = model(dummy_input)
    state = torch.load(CHECKPOINT, map_location=DEVICE)
    model.load_state_dict(state)
    model.eval()

    total_chars = 0
    total_errs = 0
    total_seqs = 0
    correct_seqs = 0

    for images, gt_texts, input_lengths in test_loader:
        images = images.to(DEVICE)
        input_lengths = input_lengths.to(DEVICE)

        log_probs = model(images)                          # T x B x C
        greedy_texts = ctc_greedy_decode(log_probs, input_lengths)
        # beam_texts = ctc_beam_search_decode(log_probs, input_lengths, beam_size=3, alpha=0.5)

        # for i in range(min(10, len(gt_texts))):
        #     print(f"GT:    {gt_texts[i]}")
        #     print(f"Greedy:{greedy_texts[i]}")
        #     print(f"Beam:  {beam_texts[i]}")
        #     print("----")
        for gt, pred in zip(gt_texts, greedy_texts):
            total_seqs += 1
            if gt == pred:
                correct_seqs += 1

            dist = levenshtein(gt, pred)
            total_errs += dist
            total_chars += len(gt)

    cer = total_errs / total_chars if total_chars > 0 else 0.0
    seq_acc = correct_seqs / total_seqs if total_seqs > 0 else 0.0

    print(f"Total sequences: {total_seqs}")
    print(f"Character Error Rate (CER): {cer*100:.2f}%")
    print(f"Sequence Accuracy: {seq_acc*100:.2f}%")

    return cer, seq_acc


if __name__ == "__main__":
    evaluate()


