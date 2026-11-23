from reasoning_from_scratch.qwen3 import download_qwen3_small, Qwen3Tokenizer
from pathlib import Path
import torch


def get_device(enable_tensor_cores=True):
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print("USING CUDA")

        if enable_tensor_cores:
            major, minor = map(int, torch.__version__.split(".")[:2])

            if (major, minor) >= (2, 9):
                torch.backends.cuda.matmul.fp32_precision = "tf32"
                torch.backends.cudnn.conv.fp32_precision = "tf32"
            else:
                torch.backends.cuda.matmul.allow_tf32 = True
                torch.backends.cudnn.allow_tf32 = True

        elif torch.backends.mps.is_available():
            device = torch.device("mps")
            print("using apple silicon GPUS")

        elif torch.xpu.is_available():
            device = torch.device("xpu")
            print("intel gpu")
        else:
            device = torch.device("cpu")
            print("using cpu")

        return device


def main():
    download_qwen3_small(kind="base", tokenizer_only=True, out_dir="qwen3")

    # load the tokenizer settings from the tokenizer file into Qwen3Tokenizer path
    tokenizer_path = Path("qwen3") / "tokenizer-base.json"
    tokenizer = Qwen3Tokenizer(tokenizer_file_path=tokenizer_path)

    # doing  asimple tokenizer round trip, encode a text into token IDs and then decode those IDs back to text
    prompt = "Explain large language models."
    input_token_ids_list = tokenizer.encode(prompt)

    text = tokenizer.decode(input_token_ids_list)
    print(text)

    # now print the token ids that are generate from the list
    for k, v in enumerate(input_token_ids_list):
        print(f" {k} | {v} -> {tokenizer.decode([v])}")


if __name__ == "__main__":
    main()
