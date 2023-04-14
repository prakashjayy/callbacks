import torchvision

tfsm = torchvision.transforms.RandAugment(num_ops= 2, magnitude= 9, num_magnitude_bins= 31)

dataloader = dict(type="pl_hooks.ds.LightDL", 
                  ds_name="fashion_mnist", 
                  tfsm=tfsm,
                  bs=2048)

#network =dict(type="pl_hooks.network.cnn_layers")
#https://huggingface.co/docs/timm/quickstart
#model_names = timm.list_models('*resne*t*')
network = dict(type="timm.create_model", 
               model_name="resnet10t", 
               pretrained=False, 
               num_classes=10,
               in_chans=1)

optimizer = dict(
  type = "torch.optim.AdamW",
  lr = 3.7e-4,
  betas=(0.9,0.96), 
  weight_decay=4.5e-2, 
  amsgrad=True
)

# lightning stuff
devices = 1
accelerator = "gpu"
epochs = 100
save_top_k = 1
#distributed = (accelerator == "gpu") & (devices >= 2 or devices == -1)


logger = dict(__class_fullname__="pytorch_lightning.loggers.TensorBoardLogger", 
              save_dir= "./lightning_logs", 
              name="fashion")

## lightning callbacks 
checkpoint = dict(__class_fullname__="pytorch_lightning.callbacks.ModelCheckpoint", 
                  monitor="val/acc",
                  mode="max",
                  save_top_k=save_top_k,
                  filename="{epoch}-{step}-{val/acc:.3f}",
                  save_last=True)
lr_monitor = dict(__class_fullname__="pytorch_lightning.callbacks.LearningRateMonitor", 
                 logging_interval="step")
tqdmp = dict(__class_fullname__ = "pytorch_lightning.callbacks.TQDMProgressBar", 
            refresh_rate=4)

reluhook = dict(__class_fullname__="pl_hooks.hooks.ForwardHookCallback", 
              name="relu",
              mod_filter="torch.nn.ReLU", 
              type=["percent"], 
              pre=False)

callbacks = [checkpoint, lr_monitor, tqdmp, reluhook]