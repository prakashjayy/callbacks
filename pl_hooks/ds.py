__all__ = ['add_transforms', 'collate_fn', 'LightDL']

import torch
import fastcore.all as fc
import torchvision.transforms.functional as TF
import pytorch_lightning as pl
from datasets import load_dataset
from functools import partial

def add_transforms(b, tfsm, train=True):
    if train: b['image'] = [tfsm(o) for o in b["image"]]
    b["image"] = [TF.to_tensor(o) for o in b["image"]]
    return b

def collate_fn(b):
    return torch.stack([o["image"] for o in b]), torch.tensor([o["label"] for o in b])

class LightDL(pl.LightningDataModule):
    def __init__(self, ds_name, tfsm=None, bs=16): 
        super().__init__(); fc.store_attr();
        dsd = load_dataset(self.ds_name)
        train, val = dsd["train"], dsd["test"]
        self.tds = train.with_transform(partial(add_transforms, tfsm=self.tfsm, train=True))
        self.vds = val.with_transform(partial(add_transforms, tfsm=self.tfsm, train=False))
    
    __repr__ = fc.basic_repr(flds="ds_name, bs")
    def train_dl(self): return torch.utils.data.DataLoader(self.tds, collate_fn=collate_fn,\
                                                           batch_size=self.bs, num_workers=8, drop_last=True,
                                                           shuffle=True, persistent_workers=True)
    def val_dl(self): return torch.utils.data.DataLoader(self.vds, collate_fn=collate_fn, batch_size=self.bs, \
                                                         num_workers=8, persistent_workers=True)        
