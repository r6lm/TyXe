import torch
from torch.utils.data import Dataset
import pandas as pd


class ASMGMovieLens(Dataset):
  def __init__(self, input_path, period_start, period_end=None, x_cols=None):
    
    self.period_start = period_start
    self.period_end = period_end if period_end is not None else period_start
    self.x_cols = ["userId", "movieId", "period"] if x_cols is None else x_cols
    
    # load data but do not keep in instance
    all_data_df = pd.read_csv(input_path)

    # select period
    data_df = all_data_df.loc[
      lambda df:df.period.between(self.period_start, self.period_end)]

    # select columns
    x = data_df.loc[:, self.x_cols].values
    y = data_df.loc[:, "label"].values

    self.x = torch.tensor(x, dtype=torch.int32)
    self.y = torch.tensor(y, dtype=torch.float32)

  def __len__(self):
    return len(self.y)
   
  def __getitem__(self,idx):
    return self.x[idx], self.y[idx]

if __name__ == "__main__":
    d = ASMGMovieLens(
        "/home/rodrigo/Dropbox/Documents/MSc/dissertation/increcs/data/preprocessed/ml_processed.csv", 
        15, 24)
    d[31]
    print(d[0])
