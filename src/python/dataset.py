from typing import Optional
from torch.utils.data import Dataset
from torchtext.data import get_tokenizer
from torch import Tensor 
import pandas as pd 

class BertDataset(Dataset):
    MASK_PERCENTAGE = 0.15
    MASKED_INDICES_COLUMN = "masked_indices"
    TARGET_COLUMN = "indices"
    TOKEN_MASK_COLUMN = "token_mask"
    NSP_TARGET_COLUMN = "is_next"

    def __init__(
        self,
        data_path: str,
        mask_pctg: Optional[float] = None,
        should_include_text: bool = False,
    ):
        self.ds: pd.Series = pd.read_csv(data_path)["review"]
        self.mask_pctg = mask_pctg if mask_pctg else self.MASK_PERCENTAGE
        self.tokenizer = get_tokenizer("basic_english")

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

        self.df = None 

    def __len__(self) -> int:
        return len(self.df)
    
    def __getitem__(self, idx: int) -> Tensor:
        pass 

    def _prepare_dataset(self) -> pd.DataFrame:
        pass 

