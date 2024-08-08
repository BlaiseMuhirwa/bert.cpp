import torch
import argparse
from datetime import datetime
import os
import time
from model import Bert
from trainer import Trainer
from dataset import BertDataset

DATASET_PATH = os.path.join(os.getcwd(), "data/IMDB_Dataset.csv")
CHECK_POINT_DIR = os.path.join(os.getcwd(), "data/bert_checkpoints")
LOGS_DIR = os.path.join(
    os.getcwd(), f"data/logs/bert_experiment_{datetime.now().timestamp()}"
)
os.makedirs(CHECK_POINT_DIR, exist_ok=True)
os.makedirs(LOGS_DIR, exist_ok=True)


device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else ("mps" if torch.backends.mps.is_available() else "cpu")
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("Parameters for BERT training.")
    parser.add_argument("--hidden-size", type=int, default=36)
    parser.add_argument("--num-epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--num-heads", type=int, default=6)
    parser.add_argument("--embedding-dim", type=int, default=64)

    # Optimization
    parser.add_argument("--learning_rate", type=float, default=3e-4)

    # others
    parser.add_argument(
        "--use-tensor-cores", type=bool, default=False, help="Use tensor cores."
    )

    return parser.parse_args()


def main():
    args = parse_args()

    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)
        torch.cuda.empty_cache()

    if args.use_tensor_cores:
        torch.set_float32_matmul_precision("high")

    dataset = BertDataset(data_path=DATASET_PATH, should_include_text=False)
    model = Bert(
        vocab_size=len(dataset.vocab),
        dim_in=args.embedding_dim,
        dim_out=args.hidden_size,
        attn_heads=args.num_heads,
        use_learnable_pos_embed=True,
    )
    # Move the model onto the GPU
    model.to(device)

    trainer = Trainer(
        model=model,
        dataset=dataset,
        log_dir=LOGS_DIR,
        checkpoint_dir=CHECK_POINT_DIR,
        print_progress_every=20,
        print_accuracy_every=200,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        epochs=args.num_epochs,
    )

    trainer.print_summary()
    trainer()


if __name__ == "__main__":
    start = time.time()
    main()
    end = time.time()

    print(f"Total training time: {end - start:.4f}")
