import copy
import os, json
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from datasets import load_dataset
from qmllm.utils.device import empty_cache

def load_image(image_path, image_size=None):
    # Load the image using use PIL, we don't support tcs_loader
    image = Image.open(image_path).convert('RGB')
    if image_size is not None and image_size > 0:
        image = image.resize((image_size, image_size), Image.Resampling.LANCZOS)
    return image


def override_user_prompt(data_item, prompt_text):
    if not prompt_text:
        return data_item

    item = copy.deepcopy(data_item)
    if "messages" in item:
        found_user = False
        for msg in item["messages"]:
            if msg.get("role") in ("user", "human"):
                msg["content"] = prompt_text
                found_user = True
                break
        if not found_user:
            item["messages"].insert(0, {"role": "user", "content": prompt_text})
        return item

    if "conversations" in item:
        found_user = False
        for conv in item["conversations"]:
            if conv.get("from") in ("human", "user"):
                conv["value"] = prompt_text
                found_user = True
                break
        if not found_user:
            item["conversations"].insert(0, {"from": "human", "value": prompt_text})
        return item

    item["conversations"] = [{"from": "human", "value": prompt_text}]
    return item


def limit_sample_images(data_item, max_images):
    if not max_images or max_images <= 0:
        return data_item

    item = copy.deepcopy(data_item)
    for key in ("image", "images"):
        images = item.get(key)
        if isinstance(images, list):
            item[key] = images[:max_images]
    return item


def _get_sample_images(data_item):
    images = data_item.get("image")
    if not images:
        images = data_item.get("images")
    if isinstance(images, list):
        return images
    if images:
        return [images]
    return []


def set_sample_images(data_item, images):
    item = copy.deepcopy(data_item)
    if "image" in item:
        item["image"] = images
    if "images" in item:
        item["images"] = images
    if "image" not in item and "images" not in item:
        item["image"] = images
    return item


def split_sample_by_image_chunks(data_item, image_chunk_size):
    if not image_chunk_size or image_chunk_size <= 0:
        return [data_item]

    images = _get_sample_images(data_item)
    if len(images) <= image_chunk_size:
        return [set_sample_images(data_item, images)]

    chunks = []
    for start in range(0, len(images), image_chunk_size):
        chunk_images = images[start:start + image_chunk_size]
        chunks.append(set_sample_images(data_item, chunk_images))
    return chunks


def build_calibration_items(dataset, n_samples, max_images=None, image_chunk_size=None):
    items = []
    total_images = 0
    for i in range(n_samples):
        idx = i % len(dataset)
        data_item = limit_sample_images(dataset[idx], max_images)
        image_count = len(_get_sample_images(data_item))
        total_images += image_count
        items.extend(split_sample_by_image_chunks(data_item, image_chunk_size))

    if image_chunk_size and image_chunk_size > 0:
        print(
            "[calib] expanded "
            f"{n_samples} source samples with {total_images} images "
            f"into {len(items)} image chunks (chunk_size={image_chunk_size})",
            flush=True,
        )
    return items


def get_multimodal_calib_dataset(
    data_path,
    image_folder,
    model,
    n_samples=128,
    few_shot_format=False,
    interleave_format=False,
    text_data_path=None,
    shuffle=True,
    micro_bs=16,
    image_size=None,
    max_images=None,
    image_chunk_size=None,
    prompt_text=None,
):
    # ------------- load dataset -------------
    if data_path.endswith(".jsonl"):
        dataset = []
        with open(data_path, "r") as json_file:
            for line in json_file:
                dataset.append(json.loads(line.strip()))
    elif data_path.endswith(".json"):
        with open(data_path, "r") as json_file:
            dataset = json.load(json_file)
    else:
        raise ValueError(f"Unsupported file type: {data_path}")

    if shuffle:
        rng = np.random.default_rng(seed=42)
        rng.shuffle(dataset)

    # ------------- sanity check -------------
    if few_shot_format and interleave_format:
        raise ValueError("You cannot specify both few_shot_format and interleave_format at the same time!")

    # ------------- build pure_text once (if needed) -------------
    pure_text = None
    if interleave_format:
        from datasets import load_dataset
        if not text_data_path:
            text_ds = load_dataset("mit-han-lab/pile-val-backup", split="validation")
        else:
            text_ds = load_dataset(text_data_path, split="validation")

        if shuffle:
            text_ds = text_ds.shuffle(seed=42)

        samples = []
        n_run = 0
        for item in text_ds:
            line = item["text"].strip()
            ids = model.tokenizer.encode(line)
            if len(ids) > 512:
                samples.append(torch.tensor(ids[:512], dtype=torch.long))
                n_run += 1
            if n_run == 128:
                break
        pure_text = samples

    # ------------- helpers: pad to target length -------------
    def pad_3d_to_len(x: torch.Tensor, target_len: int, pad_value: float = 0.0):
        # x: [B, N, C]
        cur = x.size(1)
        if cur == target_len:
            return x
        if cur > target_len:
            return x[:, :target_len, :]
        pad = target_len - cur
        return F.pad(x, (0, 0, 0, pad), value=pad_value)

    def pad_2d_to_len(x: torch.Tensor, target_len: int, pad_value):
        # x: [B, N]
        cur = x.size(1)
        if cur == target_len:
            return x
        if cur > target_len:
            return x[:, :target_len]
        pad = target_len - cur
        if x.dtype == torch.bool:
            x_u8 = x.to(torch.uint8)
            x_u8 = F.pad(x_u8, (0, pad), value=0)
            return x_u8.to(torch.bool)
        return F.pad(x, (0, pad), value=pad_value)

    calib_items = build_calibration_items(
        dataset,
        n_samples=n_samples,
        max_images=max_images,
        image_chunk_size=image_chunk_size,
    )
    n_calib_items = len(calib_items)

    # ------------- accumulate CPU chunks; pad/copy once at the end -------------
    chunks = []
    global_max_len = 0
    total_samples = 0

    for st in range(0, n_calib_items, micro_bs):
        micro_data_list = []
        ed = min(st + micro_bs, n_calib_items)
        print(f"[calib] building chunks {st}-{ed - 1} / {n_calib_items}", flush=True)

        for i in range(st, ed):
            data_item = calib_items[i]
            data_item = override_user_prompt(data_item, prompt_text)

            # load images
            if "image" in data_item and data_item["image"] and len(data_item["image"]) != 0:
                images = []
                if isinstance(data_item["image"], list):
                    for image_path in data_item["image"]:
                        full_image_path = os.path.join(image_folder, image_path)
                        images.append(load_image(full_image_path, image_size=image_size))
                else:
                    full_image_path = os.path.join(image_folder, data_item["image"])
                    images.append(load_image(full_image_path, image_size=image_size))
            else:
                images = None

            data_dict = model.preprocess_data(images, data_item)
            micro_data_list.append(data_dict)

        examples = model.data_collator(micro_data_list)

        if few_shot_format:
            examples = model.few_shot_data_samples(examples)

        if interleave_format:
            examples = model.interleave_data_samples(examples, pure_text=pure_text)

        # Generate each micro batch on the accelerator, then accumulate on CPU.
        prompt_inputs, prompt_kwargs = model.generate_input(examples)

        cur_embeds = prompt_inputs["inputs_embeds"].detach().cpu()       # [B, N, C]
        cur_labels = prompt_kwargs["labels"].detach().cpu()              # [B, N]
        cur_attn   = prompt_kwargs["attention_mask"].detach().cpu()      # [B, N]
        cur_vmask  = prompt_kwargs["vision_mask"].detach().cpu()         # [B, N]
        cur_cmask  = prompt_kwargs["caption_mask"].detach().cpu()        # [B, N]

        cur_len = cur_embeds.size(1)
        chunks.append(
            {
                "inputs_embeds": cur_embeds,
                "labels": cur_labels,
                "attention_mask": cur_attn,
                "vision_mask": cur_vmask,
                "caption_mask": cur_cmask,
            }
        )
        total_samples += cur_embeds.size(0)
        global_max_len = max(global_max_len, cur_len)

        # cleanup
        del micro_data_list, examples, prompt_inputs, prompt_kwargs
        empty_cache(getattr(model, "device", None))
        print(f"[calib] accumulated {ed}/{n_calib_items}, max_len={global_max_len}", flush=True)

    first_chunk = chunks[0]
    hidden_size = first_chunk["inputs_embeds"].size(-1)
    print(
        f"[calib] finalizing {total_samples} samples, max_len={global_max_len}, hidden_size={hidden_size}",
        flush=True,
    )

    prompt_inputs_all = {
        "inputs_embeds": torch.zeros(
            (total_samples, global_max_len, hidden_size),
            dtype=first_chunk["inputs_embeds"].dtype,
        )
    }
    prompt_kwargs_all = {
        "labels": torch.full(
            (total_samples, global_max_len),
            -100,
            dtype=first_chunk["labels"].dtype,
        ),
        "attention_mask": torch.zeros(
            (total_samples, global_max_len),
            dtype=first_chunk["attention_mask"].dtype,
        ),
        "vision_mask": torch.zeros(
            (total_samples, global_max_len),
            dtype=first_chunk["vision_mask"].dtype,
        ),
        "caption_mask": torch.zeros(
            (total_samples, global_max_len),
            dtype=first_chunk["caption_mask"].dtype,
        ),
    }

    offset = 0
    for chunk in chunks:
        bs, cur_len = chunk["inputs_embeds"].shape[:2]
        sl = slice(offset, offset + bs)
        prompt_inputs_all["inputs_embeds"][sl, :cur_len] = chunk["inputs_embeds"]
        prompt_kwargs_all["labels"][sl, :cur_len] = chunk["labels"]
        prompt_kwargs_all["attention_mask"][sl, :cur_len] = chunk["attention_mask"]
        prompt_kwargs_all["vision_mask"][sl, :cur_len] = chunk["vision_mask"]
        prompt_kwargs_all["caption_mask"][sl, :cur_len] = chunk["caption_mask"]
        offset += bs
        chunk.clear()
    chunks.clear()

    return prompt_inputs_all, prompt_kwargs_all
