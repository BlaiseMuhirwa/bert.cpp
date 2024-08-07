import torch
import random
from typing import Optional
from torch.utils.data import Dataset
import torchtext
from torchtext.data import get_tokenizer
from torch import Tensor
import pandas as pd
import numpy as np
import tqdm
from collections import Counter
import os
import logging 

logger = logging.getLogger(__name__)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.debug(f"Using device: {device}")


class BertDataset(Dataset):
    MASK_PERCENTAGE = 0.15
    MASKED_INDICES_COLUMN = "masked_indices"
    TARGET_COLUMN = "indices"
    TOKEN_MASK_COLUMN = "token_mask"
    NSP_TARGET_COLUMN = "is_next"
    OPTIMAL_LENGTH_PERCENTILE = 70

    CLS = "[CLS]"
    MASK = "[MASK]"
    PAD = "[PAD]"
    SEP = "[SEP]"
    UNK = "[UNK]"

    SPECIAL_TOKENS = [CLS, MASK, PAD, SEP, UNK]

    SERIALIZED_TOKENIZED_DATASET_PATH = os.path.join(os.getcwd(), "data/tokenized.csv")

    def __init__(
        self,
        data_path: str,
        mask_pctg: Optional[float] = None,
        chunk_range: Optional[tuple] = None,
        should_include_text: bool = False,
        serialize_tokenized_dataset: bool = False
    ):
        self.ds: pd.Series = pd.read_csv(data_path)["review"]
        if chunk_range:
            assert len(chunk_range) == 2
            self.ds = self.ds[chunk_range[0] : chunk_range[1]]
        self.mask_pctg = mask_pctg if mask_pctg else self.MASK_PERCENTAGE
        self.tokenizer = get_tokenizer("basic_english")
        self.counter = Counter()
        self.should_include_text = should_include_text

        if should_include_text:
            self.columns = [
                "masked_sentence",
                self.MASKED_INDICES_COLUMN,
                "sentence",
                self.TARGET_COLUMN,
                self.TOKEN_MASK_COLUMN,
                self.NSP_TARGET_COLUMN,
            ]
        else:
            self.columns = [
                self.MASKED_INDICES_COLUMN,
                self.TARGET_COLUMN,
                self.TOKEN_MASK_COLUMN,
                self.NSP_TARGET_COLUMN,
            ]

        self.df = self._prepare_dataset()
        if serialize_tokenized_dataset:
            self.df.to_csv(self.SERIALIZED_TOKENIZED_DATASET_PATH)

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Tensor:
        item = self.df.iloc[idx]
        input = torch.tensor(item[self.MASKED_INDICES_COLUMN]).long()
        token_mask = torch.tensor(item[self.TOKEN_MASK_COLUMN]).bool()

        mask_target = torch.tensor(item[self.TARGET_COLUMN]).long()
        # Directly set all non-masked tokens to 0 
        mask_target = mask_target.masked_fill_(token_mask, 0)
        attention_mask = (input == self.vocab[self.PAD]).unsqueeze(0)

        if item[self.NSP_TARGET_COLUMN] == 0:
            nsp_target = torch.tensor([1, 0])
        else:
            nsp_target = torch.tensor([0, 1])

        return (
            input.to(device),
            attention_mask.to(device),
            token_mask.to(device),
            mask_target.to(device),
            nsp_target.to(device),
        )

    def _fill_vocab(self):
        # Adds to the vocabulary words that appear >= 2 times in the
        # training corpus.
        self.vocab: torchtext.vocab.Vocab = torchtext.vocab.vocab(
            self.counter, min_freq=2, specials=self.SPECIAL_TOKENS
        )
        self.vocab.set_default_index(4)

    def _prepare_dataset(self) -> pd.DataFrame:
        sentences, nsp, sentence_lens = [], [], []
        for review in self.ds:
            review_sentences = review.split(". ")
            sentences.extend(review_sentences)
            self._update_length(review_sentences, sentence_lens)

        self.optimal_sentence_length = self._find_optimal_sentence_length(sentence_lens)

        for sentence in tqdm.tqdm(sentences):
            s = self.tokenizer(sentence)
            self.counter.update(s)

        self._fill_vocab()

        for review in tqdm.tqdm(self.ds):
            review_sentences = review.split(". ")
            if not len(review_sentences):
                continue

            for i in range(len(review_sentences) - 1):
                # True NSP item
                first, second = self.tokenizer(review_sentences[i]), self.tokenizer(
                    review_sentences[i + 1]
                )
                nsp.append(self._create_item(first, second, 1))

                # False NSP item
                first, second = self._select_false_nsp_sentence(sentences)
                first, second = self.tokenizer(first), self.tokenizer(second)
                nsp.append(self._create_item(first, second, 0))

        return pd.DataFrame(nsp, columns=self.columns)

    def _find_optimal_sentence_length(self, sentence_lens: list[int]) -> int:
        return int(
            np.percentile(np.array(sentence_lens), self.OPTIMAL_LENGTH_PERCENTILE)
        )

    def _update_length(self, sentences: list[str], lengths: list[int]) -> None:
        for v in sentences:
            l = len(v.split())
            lengths.append(l)
        return lengths

    def _create_item(self, first: list[str], second: list[str], target: int) -> tuple:
        # Create masked sentence item
        updated_first, first_mask = self._preprocess_sentence(first.copy())
        updated_second, second_mask = self._preprocess_sentence(second.copy())

        nsp_sentence = updated_first + [self.SEP] + updated_second
        nsp_indices = self.vocab.lookup_indices(nsp_sentence)
        inverse_token_mask = first_mask + [True] + second_mask

        # Create sentence item without masking random words
        first, _ = self._preprocess_sentence(first.copy(), apply_mask=False)
        second, _ = self._preprocess_sentence(second.copy(), apply_mask=False)

        original_nsp_sentence = first + [self.SEP] + second
        original_nsp_indices = self.vocab.lookup_indices(original_nsp_sentence)

        if self.should_include_text:
            return (
                nsp_sentence,
                nsp_indices,
                original_nsp_sentence,
                original_nsp_indices,
                inverse_token_mask,
                target,
            )
        else:
            return nsp_indices, original_nsp_indices, inverse_token_mask, target

    def _select_false_nsp_sentence(self, sentences: list[str]):
        length = len(sentences)
        sentence_idx = random.randint(0, length - 1)
        next_sentence_idx = random.randint(0, length - 1)
        while sentence_idx + 1 == next_sentence_idx:
            next_sentence_idx = random.randint(0, length - 1)

        return sentences[sentence_idx], sentences[next_sentence_idx]

    def _preprocess_sentence(
        self, sentence: list[str], apply_mask: bool = True
    ) -> tuple:
        inverse_token_mask = None
        if apply_mask:
            sentence, inverse_token_mask = self._mask_sentence(sentence)
        inverse_token_mask = [True] + inverse_token_mask if inverse_token_mask else None
        sentence, inverse_token_mask = self._pad_sentence(
            [self.CLS] + sentence, inverse_token_mask
        )
        return sentence, inverse_token_mask

    def _mask_sentence(self, sentence: list[str]) -> tuple:
        length = len(sentence)
        inverse_token_mask = [
            True for _ in range(max(length, self.optimal_sentence_length))
        ]
        mask_amount = round(length * self.mask_pctg)

        for _ in range(mask_amount):
            i = random.randint(0, length - 1)
            randn = random.random()
            if randn < 0.1:
                # 10% of the time we don't change the token
                continue
            elif randn < 0.9:
                # 80% of the time we replace it with the mask token
                sentence[i] = self.MASK
            else:
                # 10% of the time we replace it with a random token in the vocab
                j = random.randint(5, len(self.vocab) - 1)
                sentence[i] = self.vocab.lookup_token(j)
            inverse_token_mask[i] = False

        return sentence, inverse_token_mask

    def _pad_sentence(self, sentence: list[str], inverse_token_mask: list[bool] = None):
        length = len(sentence)
        if length >= self.optimal_sentence_length:
            s = sentence[: self.optimal_sentence_length]
        else:
            s = sentence + [self.PAD] * (self.optimal_sentence_length - length)

        if inverse_token_mask:
            length_mask = len(inverse_token_mask)
            if length_mask >= self.optimal_sentence_length:
                inverse_token_mask = inverse_token_mask[: self.optimal_sentence_length]
            else:
                inverse_token_mask = inverse_token_mask + [True] * (
                    self.optimal_sentence_length - length_mask
                )

        return s, inverse_token_mask


if __name__ == "__main__":
    dataset = BertDataset(
        data_path=os.path.join(os.getcwd(), "data/IMDB Dataset.csv"),
        # chunk_range=(0, 10),
        should_include_text=True,
        serialize_tokenized_dataset=True
    )
    stce = ["[CLS]", "this", "works", "[MASK]", "well"]
    indices = dataset.vocab.lookup_indices(stce)
    print(f"indices: {indices}")
    for i in range(20):
        print(f"token[{i}]: {dataset.vocab.lookup_token(i)}")

    print(f"1st item: {dataset[0]}")
