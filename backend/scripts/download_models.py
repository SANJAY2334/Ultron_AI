"""Script to download pretrained neural model weights for ULTRON local inference."""

from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer

def main() -> None:
    target_dir = Path(__file__).resolve().parent.parent / "app" / "ai" / "local_models" / "ultron_distilgpt2"
    target_dir.mkdir(parents=True, exist_ok=True)
    weights_path = target_dir / "model.safetensors"

    if weights_path.exists():
        print(f"Pretrained weights already exist at {weights_path} ({weights_path.stat().st_size / (1024*1024):.1f} MB).")
        return

    print("Downloading distilbert/distilgpt2 pretrained weights and tokenizer...")
    tok = AutoTokenizer.from_pretrained("distilbert/distilgpt2")
    model = AutoModelForCausalLM.from_pretrained("distilbert/distilgpt2")

    tok.save_pretrained(str(target_dir))
    model.save_pretrained(str(target_dir))
    print(f"Saved model and tokenizer to {target_dir} successfully!")

if __name__ == "__main__":
    main()
