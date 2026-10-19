import math
import random

import jiwer
import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageEnhance, ImageFilter, ImageOps
from torch.utils.data import Dataset
from torchvision import transforms

LAYOUTS = ("squash", "pad", "fold")


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def pad_to_square(img: Image.Image, fill=255) -> Image.Image:
    w, h = img.size
    side = max(w, h)
    canvas = Image.new(img.mode, (side, side), fill)
    canvas.paste(img, ((side - w) // 2, (side - h) // 2))
    return canvas


def fold_line(img: Image.Image, overlap: int = 16, fill=255) -> Image.Image:
    w, h = img.size
    k = max(1, round(math.sqrt(w / h)))
    if k == 1:
        return pad_to_square(img, fill)
    seg = math.ceil(w / k)
    canvas = Image.new(img.mode, (seg + overlap, k * h), fill)
    for i in range(k):
        right = w - i * seg
        left = max(0, right - seg - overlap)
        piece = img.crop((left, 0, right, h))
        canvas.paste(piece, (canvas.width - piece.width, i * h))
    return pad_to_square(canvas, fill)


class PrintAugment:
    def __init__(self, p: float = 0.8):
        self.p = p
        self.affine = transforms.RandomAffine(degrees=1.5, translate=(0.01, 0.05), scale=(0.9, 1.1),
                                              shear=(-8, 8), fill=255)

    def __call__(self, img: Image.Image) -> Image.Image:
        if random.random() > self.p:
            return img
        img = self.affine(img)
        r = random.random()
        if r < 0.25:
            img = img.filter(ImageFilter.MinFilter(3))
        elif r < 0.5:
            img = img.filter(ImageFilter.MaxFilter(3))
        if random.random() < 0.3:
            img = img.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.3, 1.2)))
        if random.random() < 0.5:
            img = ImageEnhance.Contrast(img).enhance(random.uniform(0.65, 1.4))
            img = ImageEnhance.Brightness(img).enhance(random.uniform(0.85, 1.1))
        if random.random() < 0.3:
            arr = np.asarray(img, dtype=np.float32)
            arr += np.random.normal(0, random.uniform(5, 20), arr.shape)
            img = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))
        return img


def load_image(path: str, layout: str = "squash", augment=None) -> Image.Image:
    img = ImageOps.grayscale(Image.open(path))
    if augment is not None:
        img = augment(img)
    if layout == "pad":
        img = pad_to_square(img)
    elif layout == "fold":
        img = fold_line(img)
    return img


class LineDataset(Dataset):
    def __init__(self, df: pd.DataFrame, processor, layout="squash", max_target_len=128, augment=False):
        self.df = df.reset_index(drop=True)
        self.processor = processor
        self.layout = layout
        self.max_target_len = max_target_len
        self.augment = PrintAugment() if augment else None

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img = load_image(row["path"], self.layout, self.augment)
        pixel_values = self.processor(images=img.convert("RGB"), return_tensors="pt").pixel_values[0]

        tok = self.processor.tokenizer
        ids = tok(str(row["text"]), add_special_tokens=False).input_ids[: self.max_target_len - 1]
        labels = ids + [tok.sep_token_id]
        labels += [-100] * (self.max_target_len - len(labels))
        return {"pixel_values": pixel_values, "labels": torch.tensor(labels)}


def configure_model(model, processor, max_new_tokens=128):
    tok = processor.tokenizer
    model.config.decoder_start_token_id = model.config.decoder.decoder_start_token_id or tok.sep_token_id
    model.config.pad_token_id = tok.pad_token_id
    model.config.eos_token_id = tok.sep_token_id
    model.config.vocab_size = model.config.decoder.vocab_size
    gen = model.generation_config
    gen.decoder_start_token_id = model.config.decoder_start_token_id
    gen.pad_token_id = tok.pad_token_id
    gen.eos_token_id = tok.sep_token_id
    gen.max_new_tokens = max_new_tokens
    return model


@torch.no_grad()
def predict(model, processor, df, device, layout="squash", num_beams=1, batch_size=32, progress=True):
    from tqdm import tqdm

    model.eval()
    preds = []
    for start in tqdm(range(0, len(df), batch_size), desc="tahmin", disable=not progress):
        batch = df.iloc[start:start + batch_size]
        images = [load_image(p, layout).convert("RGB") for p in batch["path"]]
        pixel_values = processor(images=images, return_tensors="pt").pixel_values.to(device)
        out = model.generate(pixel_values, num_beams=num_beams)
        preds.extend(processor.batch_decode(out, skip_special_tokens=True))
    return [p.strip() for p in preds]


def compute_metrics(refs, preds) -> dict:
    refs = [str(r) for r in refs]
    preds = [str(p) for p in preds]
    low_r, low_p = [r.lower() for r in refs], [p.lower() for p in preds]
    return {
        "CER": jiwer.cer(refs, preds),
        "WER": jiwer.wer(refs, preds),
        "CER_lower": jiwer.cer(low_r, low_p),
        "WER_lower": jiwer.wer(low_r, low_p),
        "line_accuracy": sum(r == p for r, p in zip(refs, preds)) / len(refs),
        "n": len(refs),
    }
