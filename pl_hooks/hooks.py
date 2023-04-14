#PrakashJay
from pytorch_lightning.callbacks import Callback
from functools import partial 
import fastcore.all as fc 
import pydoc 
import pandas as pd 


class ForwardHookCallback(Callback):
    def __init__(self, name, mod_filter=fc.noop, type="percent", pre=False): 
        fc.store_attr() 
        self.mod_filter = pydoc.locate(mod_filter) if isinstance(mod_filter, str) else mod_filter
        self.hook = False 
        self.type = self.type if isinstance(self.type, list) else [self.type]
        self.values = []

    def reset(self, name):
        if name not in self.m.keys(): self.m[name] = [] 

    def hookmean(self, name, mod, inp, outp):
        if self.hook:
            self.reset(name)
            outp = outp.detach().to("cpu")
            self.m[name] = float(outp.mean())
    
    def hookstd(self, name, mod, inp, outp):
        if self.hook:
            self.reset(name)
            outp = outp.detach().to("cpu")
            self.m[name] = float(outp.std())

    def hookpercent(self, name, mod, inp, outp):
        if self.hook:
            self.reset(name)
            self.m[name] = float(100- (outp.nonzero().shape[0]/outp.numel())*100)
    
    def hookhist(self, name, mod, inp, outp):
        if self.hook:
            self.reset(name)
            val = outp.histc(40).log1p().detach().to("cpu")
            self.m[name] = val.tolist()

    def setup(self, trainer, pl_module, stage):
        if stage == "fit":
            #we need to register all the hooks from pl_module.model
            print("Regitering hooks")
            self.m = dict()
            mods = fc.filter_ex(pl_module.model.modules(), fc.risinstance(self.mod_filter))
            for n, k in enumerate(mods):
                for type in self.type: 
                    nt = f"{self.name}-{n}-{type}"
                    print(f"registering: {nt} for {k}")
                    k.register_forward_hook(partial(getattr(self, f"hook{type}"), nt))
                    self.m[nt] = []
            
    
    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        tt = {k: (v if isinstance(v, list) else round(v, 3)) for k, v in self.m.items()}
        self.values.append(tt)

        for k in self.m.keys(): self.m[k] = []# making the list free of metrics as this is already logged.
        self.hook = False 
    
    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx): 
        self.hook = True 
    
    def on_train_epoch_end(self, trainer, pl_module):
        pd.DataFrame(self.values).to_csv(pl_module.logger.log_dir+f"/{self.name}_epoch.csv", index=False)
    
    def on_fit_end(self, trainer, pl_module) -> None:
        pd.DataFrame(self.values).to_csv(pl_module.logger.log_dir+f"{self.name}/fit_end.csv", index=False)