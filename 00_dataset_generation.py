import os
import random
import webdataset as wds
import numpy as np
import pandas as pd
from tqdm import tqdm
import gc
import io
from PIL import Image
from config import word2idx10k
from PIL import Image, ImageDraw, ImageFont
import json
from config import fname, stimuli
import pickle


def CreateWordSetWebDataset(
    path_out,
    font_path,
    wordlist,
    num_train=200,
    num_val=50,
    num_shards=10,  # Number of shards to create
):
    # Define words, sizes, fonts

    sizes = np.linspace(40, 80, 21)
    rotations = np.linspace(-20, 20, 11)
    fonts = [font_path + os.path.splitext(f)[0] for f in os.listdir(font_path)]
    xshift = [-50, -40, -30, -20, -10, 0, 10, 20, 30, 40, 50]
    yshift = [-30, -15, 0, 15, 30]

    # Create output directory
    os.makedirs(path_out, exist_ok=True)

    # Generate all samples and shuffle them before creating tar files
    print("Generating all samples...")
    train_samples = []
    val_samples = []

    # For each word, create samples
    for word_idx, word in enumerate(tqdm(wordlist)):
        gc.collect()
        print(f"Processing word: {word} ({word_idx+1}/{len(wordlist)})")

        # Get class index for this word
        idx = word2idx10k[word]

        for n in range(num_train + num_val):
            # Select random parameters
            font = random.choice(fonts)
            size = random.choice(sizes)
            upper = random.choice([0, 1])
            x = random.choice(xshift)
            y = random.choice(yshift)
            angle = random.choice(rotations)

            # Create a sample info dict
            sample_info = {
                "word": word,
                "index": n,
                "font": font,
                "size": size,
                "upper": upper,
                "xshift": x,
                "yshift": y,
                "angle": angle,
                "class_idx": idx,
            }

            # Add to appropriate list
            if n < num_train:
                train_samples.append(sample_info)
            else:
                val_samples.append(sample_info)

    # Shuffle all samples
    print("Shuffling samples...")
    random.shuffle(train_samples)
    random.shuffle(val_samples)

    # Calculate samples per shard
    train_samples_per_shard = (len(train_samples) + num_shards - 1) // num_shards
    val_samples_per_shard = (len(val_samples) + num_shards - 1) // num_shards

    # Create shards for training data
    print(f"Writing training samples to {num_shards} tar files...")
    for shard_idx in range(num_shards):
        # Calculate start and end indices for this shard
        start_idx = shard_idx * train_samples_per_shard
        end_idx = min((shard_idx + 1) * train_samples_per_shard, len(train_samples))

        # Skip if no samples in this range
        if start_idx >= len(train_samples):
            continue

        # Create shard file name
        train_file = os.path.join(path_out, f"train-{shard_idx:05d}.tar")

        # Get samples for this shard
        shard_samples = train_samples[start_idx:end_idx]

        print(
            f"Writing training shard {shard_idx+1}/{num_shards} with {len(shard_samples)} samples..."
        )

        # Write samples to tar file
        with wds.TarWriter(train_file) as shard_sink:
            for i, sample_info in enumerate(
                tqdm(shard_samples, desc=f"Training shard {shard_idx+1}")
            ):
                # Create image using drawing function
                img = draw_rotated_text(
                    text=sample_info["word"],
                    fontname=sample_info["font"],
                    size=sample_info["size"],
                    xshift=sample_info["xshift"],
                    yshift=sample_info["yshift"],
                    angle=sample_info["angle"],
                    upper=sample_info["upper"],
                )

                # Convert the PIL image to bytes
                img_buffer = io.BytesIO()
                img.save(img_buffer, format="PNG")
                img_bytes = img_buffer.getvalue()

                # Create a unique key for this sample
                sample_key = f"{sample_info['word']}_{start_idx}_{i:06d}"

                # Create the sample
                sample = {
                    "__key__": sample_key,
                    "png": img_bytes,
                    "cls": sample_info["class_idx"],  # Store word as cls
                    # "cls_idx": sample_info["class_idx"],  # Store class index as cls_idx
                    "json": json.dumps(
                        {
                            "word": sample_info["word"],
                            "font": os.path.basename(sample_info["font"]),
                            "size": float(sample_info["size"]),
                            "upper": int(sample_info["upper"]),
                            "xshift": int(sample_info["xshift"]),
                            "yshift": int(sample_info["yshift"]),
                            "angle": float(sample_info["angle"]),
                            "class_idx": sample_info["class_idx"],
                            "shard": shard_idx,
                        }
                    ),
                }

                # Write the sample
                shard_sink.write(sample)

    # Create shards for validation data
    print(f"Writing validation samples to {num_shards} tar files...")
    for shard_idx in range(num_shards):
        # Calculate start and end indices for this shard
        start_idx = shard_idx * val_samples_per_shard
        end_idx = min((shard_idx + 1) * val_samples_per_shard, len(val_samples))

        # Skip if no samples in this range
        if start_idx >= len(val_samples):
            continue

        # Create shard file name
        val_file = os.path.join(path_out, f"val-{shard_idx:05d}.tar")

        # Get samples for this shard
        shard_samples = val_samples[start_idx:end_idx]

        print(
            f"Writing validation shard {shard_idx+1}/{num_shards} with {len(shard_samples)} samples..."
        )

        # Write samples to tar file
        with wds.TarWriter(val_file) as shard_sink:
            for i, sample_info in enumerate(
                tqdm(shard_samples, desc=f"Validation shard {shard_idx+1}")
            ):
                # Create image using drawing function
                img = draw_rotated_text(
                    text=sample_info["word"],
                    fontname=sample_info["font"],
                    size=sample_info["size"],
                    xshift=sample_info["xshift"],
                    yshift=sample_info["yshift"],
                    angle=sample_info["angle"],
                    upper=sample_info["upper"],
                )

                # Convert the PIL image to bytes
                img_buffer = io.BytesIO()
                img.save(img_buffer, format="PNG")
                img_bytes = img_buffer.getvalue()

                # Create a unique key for this sample
                sample_key = f"{sample_info['word']}_{start_idx}_{i:06d}"

                # Create the sample
                sample = {
                    "__key__": sample_key,
                    "png": img_bytes,
                    "cls": sample_info["class_idx"],  # Store word as cls
                    "json": json.dumps(
                        {
                            "word": sample_info["word"],
                            "font": os.path.basename(sample_info["font"]),
                            "size": float(sample_info["size"]),
                            "upper": int(sample_info["upper"]),
                            "xshift": int(sample_info["xshift"]),
                            "yshift": int(sample_info["yshift"]),
                            "angle": float(sample_info["angle"]),
                            "class_idx": sample_info["class_idx"],
                            "shard": shard_idx,
                        }
                    ),
                }

                # Write the sample
                shard_sink.write(sample)

    # Create a metadata file with dataset information
    metadata = {
        "train_samples": len(train_samples),
        "val_samples": len(val_samples),
        "train_shards": num_shards,
        "val_shards": num_shards,
        "samples_per_train_shard": train_samples_per_shard,
        "samples_per_val_shard": val_samples_per_shard,
        "num_classes": len(wordlist),
    }

    with open(os.path.join(path_out, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"Created {len(train_samples)} training samples across {num_shards} shards")
    print(f"Created {len(val_samples)} validation samples across {num_shards} shards")

    return "done"


# You'll need to implement this function to create an image in memory
def draw_rotated_text(
    text, size, fontname, angle=0, upper=1, xshift=0, yshift=0, W=500, H=500
):
    """
    Modified version of draw_rotated_text that returns a PIL Image instead of saving to disk

    Parameters:
    - text: String text to draw
    - size: Font size
    - fontname: Path to the font file (without .ttf extension)
    - angle: Float rotation angle in degrees (counter-clockwise)
    - upper: Whether to convert text to uppercase (1) or not (0)
    - xshift: Horizontal shift from center
    - yshift: Vertical shift from center
    - W: Image width
    - H: Image height

    Returns:
    - PIL Image with the rotated text
    """

    # Create a white background image
    image = Image.new("RGB", (W, H), color=(255, 255, 255))

    # Convert text to uppercase if needed
    if upper:
        text = text.upper()

    # Load the font
    try:
        font = ImageFont.truetype(f"{fname.fonts_dir}/{fontname}.ttf", int(size))
    except Exception as e:
        print(f"Font error: {e}, using default font")
        font = ImageFont.load_default()

    # Get text dimensions
    try:
        bbox = font.getbbox(text)  # Returns (x_min, y_min, x_max, y_max)
        w, h = bbox[2] - bbox[0], bbox[3] - bbox[1]
    except AttributeError:
        # Fall back for older PIL versions
        w, h = font.getsize(text)
        bbox = (0, 0, w, h)

    # Create a transparent image for the text
    text_img = Image.new("RGBA", (W, H), (255, 255, 255, 0))
    text_draw = ImageDraw.Draw(text_img)

    # Draw the text onto the transparent image, centered with the specified shifts
    text_draw.text(
        (xshift + (W - w) / 2, yshift + (H - h) / 2), text, font=font, fill="black"
    )

    # Rotate the text image
    rotated_text = text_img.rotate(angle, expand=0, resample=Image.BICUBIC)

    # Paste the rotated text onto the white background image using alpha channel as mask
    if image.mode != "RGBA":
        image = image.convert("RGBA")

    image.paste(rotated_text, (0, 0), rotated_text)

    # Convert back to RGB mode for compatibility
    image = image.convert("RGB")

    return image


def CreateStimuliWebDataset(path_out, font, type, stimuli):

    # Create output directory
    os.makedirs(path_out, exist_ok=True)

    if type != "stimuli":
        stimuli = stimuli[stimuli["type"] == type]
    stimlist = stimuli["stimuli"].tolist()
    baselist = stimuli["base"].tolist()
    typelist = stimuli["type"].tolist()
    typeidxlist = stimuli["index"].tolist()

    # Prepare list to hold all sample information
    samples = []

    # Generate sample information
    print(f"Generating sample information for {len(stimlist)} words...")
    for word_idx, word in enumerate(stimlist):
        base = baselist[word_idx]
        stimulus_type = typelist[word_idx]
        typeidx = typeidxlist[word_idx] - 1
        # Create a sample info dict
        sample_info = {
            "word": word,
            "type": stimulus_type,
            "type_idx": typeidx,
            "font": font,
            "size": 48,
            "upper": 1,
            "xshift": 0,
            "yshift": 0,
            "angle": 0,
            "base": base,
            "class_idx": word2idx10k[base.lower()],
        }
        samples.append(sample_info)

    # Shuffle all samples
    print("Shuffling samples...")
    random.shuffle(samples)

    # Define single tar file path
    tar_file = os.path.join(path_out, f"{type}.tar")

    # Write samples to a single tar file
    print(f"Writing {len(samples)} samples to tar file...")
    with wds.TarWriter(tar_file) as sink:
        for i, sample_info in enumerate(tqdm(samples)):
            # Create image using drawing function
            img = draw_rotated_text(
                text=sample_info["word"],
                size=sample_info["size"],
                fontname=sample_info["font"],
                angle=sample_info["angle"],
                upper=sample_info["upper"],
                xshift=sample_info["xshift"],
                yshift=sample_info["yshift"],
            )

            # Convert the PIL image to bytes
            img_buffer = io.BytesIO()
            img.save(img_buffer, format="PNG")
            img_bytes = img_buffer.getvalue()

            # Create a unique key for this sample
            sample_key = f"{sample_info['word']}_{i:06d}"

            # Create the sample
            sample = {
                "__key__": sample_key,
                "png": img_bytes,
                "cls": sample_info["class_idx"],
                # "cls.idx": sample_info["class_idx"],
                "json": json.dumps(
                    {
                        "word": sample_info["word"],
                        "base": sample_info["base"],
                        "font": os.path.basename(sample_info["font"]),
                        "size": float(sample_info["size"]),
                        "upper": int(sample_info["upper"]),
                        "xshift": int(sample_info["xshift"]),
                        "yshift": int(sample_info["yshift"]),
                        "angle": float(sample_info["angle"]),
                        "type": sample_info["type"],
                        "type_idx": sample_info["type_idx"],
                    }
                ),
            }

            # Write the sample
            sink.write(sample)

    # Create a metadata file with dataset information
    metadata = {
        "total_samples": len(samples),
        "classes": stimlist,
        "bases": baselist,
        "class_to_idx": word2idx10k,
        "num_classes": len(stimuli),
        "type": type,
    }

    with open(os.path.join(path_out, f"{type}_metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"Created {len(samples)} samples in a single tar file: {tar_file}")

    return "done"


if __name__ == "__main__":
    # create datasets for training and validation
    with open(fname.word2idx_dir, "rb") as file:
        word2idx10k = pickle.load(file)

    words1k = list(word2idx10k.keys())

    # generate wordset dataset for training and validation
    CreateWordSetWebDataset(
        path_out=fname.dataset_dir,
        font_path=fname.fonts_dir,
        wordlist=words1k,
    )

    # create stimuli dataset for testing
    stimuli = pd.read_csv(fname.stimuli_dir)
    stimuli = stimuli[stimuli["target"] == "0"]
    CreateStimuliWebDataset(
        path_out=fname.dataset_dir, stimuli=stimuli, font="courier", type="stimuli"
    )
