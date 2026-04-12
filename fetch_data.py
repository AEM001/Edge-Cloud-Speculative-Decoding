"""Download and save both datasets. Uses HF_TOKEN from env."""
import json
import os
import sys
from pathlib import Path
from datasets import load_dataset

# HF Token - set explicitly since bashrc export may not propagate
HF_TOKEN = os.environ.get("HF_TOKEN", "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")
os.environ["HF_TOKEN"] = HF_TOKEN

DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True)


def save_bespoke_stratos():
    """Save Bespoke-Stratos-17k (16.6K samples, small)."""
    out_file = DATA_DIR / "bespoke_stratos.json"
    if out_file.exists():
        print(f"[bespoke_stratos] Already exists: {out_file.stat().st_size / 1e6:.1f} MB")
        return

    print("[bespoke_stratos] Downloading...")
    ds = load_dataset("HuggingFaceH4/Bespoke-Stratos-17k", split="train", token=HF_TOKEN)
    print(f"[bespoke_stratos] {len(ds):,} samples loaded")

    prompts = []
    for i, sample in enumerate(ds):
        user_text = ""
        for msg in sample.get("conversations", []) or sample.get("messages", []):
            if msg.get("from") == "user" or msg.get("role") == "user":
                user_text = msg.get("value") or msg.get("content", "")
                break

        system = sample.get("system", "")[:500]
        parts = []
        if system:
            parts.append(f"<|im_start|>system\n{system}<|im_end|>")
        if user_text:
            parts.append(f"<|im_start|>user\n{user_text[:2000]}<|im_end|>")
        parts.append("<|im_start|>assistant\n")

        prompts.append({
            "source": "bespoke_stratos",
            "id": str(sample.get("id", f"sample_{i}")),
            "system": system[:200],
            "user_message": user_text[:600],
            "formatted_prompt": "\n".join(parts),
            "prompt_length": len(user_text),
        })

    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(prompts, f, indent=2, ensure_ascii=False)
    print(f"[bespoke_stratos] Saved {len(prompts):,} prompts ({out_file.stat().st_size / 1e6:.1f} MB)")


def save_conversation_chronicles():
    """Save ConversationChronicles (785K samples, large).
    Writes incrementally to avoid OOM on the 194MB JSON."""
    out_file = DATA_DIR / "conversation_chronicles.json"

    # Check if already complete
    if out_file.exists():
        try:
            with open(out_file) as f:
                data = json.load(f)
            if len(data) > 700000:
                print(f"[conversation_chronicles] Already complete: {len(data):,} prompts")
                return
            else:
                print(f"[conversation_chronicles] Incomplete ({len(data):,}), re-downloading...")
                out_file.unlink()
        except json.JSONDecodeError:
            print("[conversation_chronicles] Corrupt file, re-downloading...")
            out_file.unlink()

    print("[conversation_chronicles] Loading dataset from cache...")
    ds = load_dataset(
        "Dans-DiscountModels/ConversationChronicles-sharegpt",
        split="train",
        token=HF_TOKEN,
    )
    total = len(ds)
    print(f"[conversation_chronicles] {total:,} samples loaded, writing to file...")

    # Stream write: one object per line to avoid building huge list in memory
    with open(out_file, "w", encoding="utf-8") as f:
        f.write("[\n")
        for i in range(total):
            sample = ds[i]
            conv_id = sample.get("id", f"sample_{i}")
            conversations = sample.get("conversations", [])

            system_msg = ""
            first_user_msg = ""
            for msg in conversations:
                role = msg.get("from", "")
                val = msg.get("value", "")
                if role == "system" and not system_msg:
                    system_msg = val[:500]
                elif role == "human" and not first_user_msg:
                    first_user_msg = val[:1000]

            parts = []
            if system_msg:
                parts.append(f"<|im_start|>system\n{system_msg}<|im_end|>")
            if first_user_msg:
                parts.append(f"<|im_start|>user\n{first_user_msg}<|im_end|>")
            parts.append("<|im_start|>assistant\n")

            entry = {
                "source": "conversation_chronicles",
                "id": conv_id,
                "system": system_msg,
                "user_message": first_user_msg,
                "formatted_prompt": "\n".join(parts),
                "prompt_length": len(first_user_msg),
            }

            if i > 0:
                f.write(",\n")
            json.dump(entry, f, ensure_ascii=False)

            if (i + 1) % 50000 == 0:
                pct = 100 * (i + 1) / total
                print(f"  {i+1:,}/{total:,} ({pct:.0f}%)")

        f.write("\n]\n")

    size_mb = out_file.stat().st_size / (1024 * 1024)
    print(f"[conversation_chronicles] Saved {total:,} prompts ({size_mb:.1f} MB)")


def verify():
    """Quick verify both files."""
    print("\n--- Verification ---")
    for name in ["bespoke_stratos", "conversation_chronicles"]:
        fpath = DATA_DIR / f"{name}.json"
        if not fpath.exists():
            print(f"  {name}: MISSING")
            continue
        with open(fpath) as f:
            data = json.load(f)
        empty = sum(1 for p in data if not p["user_message"])
        print(f"  {name}: {len(data):,} prompts, empty: {empty}, size: {fpath.stat().st_size / 1e6:.1f} MB")
        if data:
            print(f"    First: {data[0]['user_message'][:80]}...")


if __name__ == "__main__":
    print("=" * 60)
    print("DATASET DOWNLOAD")
    print("=" * 60)
    save_bespoke_stratos()
    save_conversation_chronicles()
    verify()
    print("\nDone!")
