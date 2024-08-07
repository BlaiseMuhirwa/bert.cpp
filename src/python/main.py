import torch
import argparse
from datetime import datetime
import os

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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("Parameters for BERT training.")
    parser.add_argument("--hidden-size", type=int, default=36)
    parser.add_argument("--num-epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=12)
    parser.add_argument("--num-heads", type=int, default=6)
    parser.add_argument("--embedding-dim", type=int, default=64)

    # Optimization
    parser.add_argument("--learning_rate", type=float, default=1e-4)

    return parser.parse_args()


def main():
    args = parse_args()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    dataset = BertDataset(data_path=DATASET_PATH, should_include_text=False)
    model = Bert(
        vocab_size=len(dataset.vocab),
        dim_in=args.embedding_dim,
        dim_out=args.hidden_size,
        attn_heads=args.num_heads,
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
    main()
