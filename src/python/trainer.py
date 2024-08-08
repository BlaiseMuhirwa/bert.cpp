from model import Bert
from dataset import BertDataset
import torch
from torch import Tensor
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

import os
import time
from datetime import datetime

device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else ("mps" if torch.backends.mps.is_available() else "cpu")
)

def percentage(batch_size: int, max_index: int, current_index: int):
    """
    Calculate epoch progress percentage

    Args:
        batch_size: batch size
        max_index: max index in epoch
        current_index: current index

    Returns:
        Passed percentage of dataset
    """
    batched_max = max_index // batch_size
    return round(current_index / batched_max * 100, 2)


def nsp_accuracy(result: Tensor, target: Tensor) -> float:
    """
    Accuracy for the next sentence prediction task.
    """
    s = (result.argmax(1) == target.argmax(1)).sum()
    return round(float(s / result.size(0)), 2)


def mlm_accuracy(result: Tensor, target: Tensor, inverse_token_mask: Tensor) -> float:
    """
    MLM accuracy between masked words.
    """
    r = result.argmax(-1).masked_select(~inverse_token_mask)
    t = target.masked_select(~inverse_token_mask)
    s = (r == t).sum()
    return round(float(s / (result.size(0) * result.size(1))), 2)


class Trainer:
    def __init__(
        self,
        model: Bert,
        dataset: BertDataset,
        log_dir: str,
        checkpoint_dir: str = None,
        print_progress_every: int = 10,
        print_accuracy_every: int = 50,
        batch_size: int = 24,
        learning_rate: float = 0.005,
        epochs: int = 5,
    ):
        self.model = model
        self.dataset = dataset
        self.batch_size = batch_size
        self.epochs = epochs
        self.current_epoch = 0
        self.data_loader = DataLoader(
            self.dataset, batch_size=self.batch_size, shuffle=True
        )

        self.writer = SummaryWriter(log_dir=log_dir)
        self.checkpoint_dir = checkpoint_dir

        # Sigmoid + Binary cross entry for the NSP objective
        self.loss_fn = torch.nn.BCEWithLogitsLoss().to(device)

        # Negativ log-likelihood for the MLM objective
        self.mlm_loss_fn = torch.nn.NLLLoss(ignore_index=0).to(device)
        self.optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

        self._splitter_size = 50
        self._dataset_len = len(self.dataset)
        self._num_batches = self._dataset_len // self.batch_size
        self._print_prog_every = print_progress_every
        self._print_acc_every = print_accuracy_every

    def print_summary(self):
        num_trainable_params = sum(
            p.numel() for p in self.model.parameters() if p.requires_grad
        )
        print("Model Summary\n")
        print("=" * self._splitter_size)
        print(f"Device: {device}")
        print(f"Training dataset len: {self._dataset_len}")
        print(f"Max / Optimal sentence len: {self.dataset.optimal_sentence_length}")
        print(f"Vocab size: {len(self.dataset.vocab)}")
        print(f"Batch size: {self.batch_size}")
        print(f"Num batches: {self._num_batches}")
        print(f"Trainable params: {num_trainable_params}")
        print("=" * self._splitter_size)
        print()

    def __call__(self):
        for self.current_epoch in range(self.current_epoch, self.epochs):
            loss = self.train(self.current_epoch)
            self.save_checkpoint(self.current_epoch, step=-1, loss=loss)

    def train(self, epoch: int) -> float:
        print(f"Epoch: {epoch}")

        start = time.time()
        avg_nsp_loss, avg_mlm_loss = 0, 0
        for i, value in enumerate(self.data_loader):
            index = i + 1
            input, mask, inverse_token_mask, token_target, nsp_target = value
            self.optimizer.zero_grad()

            token, nsp = self.model(input, mask)

            tm = inverse_token_mask.unsqueeze(-1).expand_as(token)
            token = token.masked_fill(tm, 0)

            # 1D tensor as target is required
            loss_token = self.mlm_loss_fn(token.transpose(1, 2), token_target)
            loss_nsp = self.loss_fn(nsp, nsp_target.float())

            loss = loss_token + loss_nsp
            avg_mlm_loss += loss_token
            avg_nsp_loss += loss_nsp

            loss.backward()
            self.optimizer.step()

            if index % self._print_prog_every == 0:
                elapsed = time.gmtime(time.time() - start)
                summary = self.training_summary(
                    elapsed, i + 1, avg_nsp_loss, avg_mlm_loss
                )

                if index % self._print_acc_every == 0:
                    summary += self.acc_summary(
                        i + 1, token, nsp, token_target, nsp_target, inverse_token_mask
                    )

                print(summary)

                avg_nsp_loss, avg_mlm_loss = 0, 0

        return loss

    def training_summary(
        self,
        elapsed: float,
        index: int,
        average_nsp_loss: float,
        average_mlm_loss: float,
    ) -> str:
        passed = percentage(self.batch_size, self._dataset_len, index)
        global_step = self.current_epoch * len(self.data_loader) + index

        print_nsp_loss = average_nsp_loss / self._print_prog_every
        print_mlm_loss = average_mlm_loss / self._print_prog_every

        s = f"{time.strftime('%H:%M:%S', elapsed)}"
        s += (
            f" | Epoch {self.current_epoch + 1} | {index} / {self._num_batches} ({passed}%) | "
            f"NSP loss {print_nsp_loss:6.2f} | MLM loss {print_mlm_loss:6.2f}"
        )

        self.writer.add_scalar("NSP loss", print_nsp_loss, global_step=global_step)
        self.writer.add_scalar("MLM loss", print_mlm_loss, global_step=global_step)
        return s

    def acc_summary(
        self,
        index: int,
        token: Tensor,
        nsp: Tensor,
        token_target: Tensor,
        nsp_target: Tensor,
        inverse_token_mask: Tensor,
    ) -> str:
        global_step = self.current_epoch * len(self.data_loader) + index
        nsp_acc = nsp_accuracy(nsp, nsp_target)
        token_acc = mlm_accuracy(token, token_target, inverse_token_mask)

        self.writer.add_scalar("NSP train accuracy", nsp_acc, global_step=global_step)
        self.writer.add_scalar(
            "Token train accuracy", token_acc, global_step=global_step
        )

        return f" | NSP accuracy {nsp_acc} | Token accuracy {token_acc}"

    def save_checkpoint(self, epoch: int, step: int, loss: float) -> None:
        if not self.checkpoint_dir:
            return
        prev = time.time()
        name = f"bert_epoch{epoch}_step{step}_{datetime.now().timestamp():.0f}.pt"
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "loss(nsp+mlm)": loss,
            },
            os.path.join(self.checkpoint_dir, name),
        )
        print()
        print(f"Model saved as '{name}' in {time.time() - prev:.2f}s")
        print()

    def load_checkpoint(self, path: str):
        print("=" * self._splitter_size)
        checkpoint = torch.load(path)
        self.current_epoch = checkpoint["epoch"]
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        print("=" * self._splitter_size)
