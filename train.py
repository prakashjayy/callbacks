import torch 
import torch.nn.functional as F 
import pytorch_lightning as pl
from pl_hooks.utils import import_module, locate_cls
from mmcv import Config

import warnings
warnings.filterwarnings("ignore")


class Trainer(pl.LightningModule):
    def __init__(self, cfg):
        super().__init__()
        self.save_hyperparameters()
        self.cfg = cfg 
        self.model = import_module(cfg.network)
        self.validation_step_outputs = []
    
    def forward(self, images):
        return self.model(images) 
    
    def training_step(self, batch, batch_idx):
        images, gts = batch 
        outputs = self(images)
        loss = F.cross_entropy(outputs, gts)
        self.log(name="train/loss", value=loss, prog_bar=True, batch_size=images.shape[0], on_step=True, on_epoch=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        images, gts = batch 
        with torch.no_grad(): outputs = self(images)
        loss = F.cross_entropy(outputs, gts)
        self.log(name="val/loss", value=loss, prog_bar=True, batch_size=images.shape[0], on_step=True, on_epoch=True)
        self.validation_step_outputs.append([gts.to("cpu"), torch.argmax(outputs, axis=1).to("cpu")])
        return loss 
    
    def on_validation_epoch_end(self):
        gts = torch.hstack([i[0] for i in self.validation_step_outputs])
        preds = torch.hstack([i[1] for i in self.validation_step_outputs])
        acc = ((gts == preds).sum())/gts.shape[0]
        self.log(name="val/acc", value=acc)
        self.validation_step_outputs.clear() 

    def configure_optimizers(self):
        optimizer = import_module(
            self.cfg.optimizer,
            params=[x for x in self.model.parameters() if x.requires_grad],
            #lr=self.learning_rate#remove
        )
        self.optimizer = optimizer

        if "scheduler" not in self.cfg.keys():
            return optimizer
        
        scheduler = import_module(self.cfg.scheduler, optimizer=optimizer)

        return [optimizer], [{"scheduler": scheduler, "interval": "epoch", "frequency": 1}]
    

def main(cfg_path: str):
    cfg = Config.fromfile(cfg_path)
    model = Trainer(cfg)
    dl = import_module(cfg.dataloader)
    
    trainer = pl.Trainer(
        devices=cfg.devices if cfg.accelerator != "cpu" else None,
        accelerator=cfg.accelerator, 
        max_epochs=cfg.epochs,
        log_every_n_steps=10,
        callbacks=[locate_cls(s) for s in cfg.callbacks],
        logger=locate_cls(cfg.logger),
    )
    # tune to find lr #remove
    trainer.fit(model, train_dataloaders=dl.train_dl(), val_dataloaders=dl.val_dl())

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Model training")
    parser.add_argument("--cfg_path", type=str, required=True)
    args = parser.parse_args()
    main(args.cfg_path)