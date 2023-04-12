#PrakashJay
from pytorch_lightning.callbacks import Callback
from functools import partial 
import fastcore.all as fc 
import pydoc 


class ForwardHookCallback(Callback):
    def __init__(self, mod_filter=fc.noop, type="percent"): 
        fc.store_attr() 
        self.mod_filter = pydoc.locate(mod_filter) if isinstance(mod_filter, str) else mod_filter
        self.hook = False 

    def reset(self, name):
        if name not in self.m.keys(): self.m[name] = [] 

    def hookpercent(self, name, mod, inp, outp):
        if self.hook:
            self.reset(name)
            self.m[name].append(100- (outp.nonzero().shape[0]/outp.numel())*100)
    
    def hookhist(self, name, mod, inp, outp):
        if self.hook:
            self.reset(name)
            val = outp.histc(40).log1p().detach().to("cpu")
            self.m[name].append(val)

    def setup(self, trainer, pl_module, stage):
        if stage == "fit":
            #we need to register all the hooks from pl_module.model
            print("Regitering hooks")
            self.m = dict()
            mods = fc.filter_ex(pl_module.model.modules(), fc.risinstance(self.mod_filter))
            for n, k in enumerate(mods): 
                k.register_forward_hook(partial(getattr(self, f"hook{self.type}"), n))
                self.m[n] = []
            
    
    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        # we need to gather all the parameters and log them to logger. 
        for k, v in self.m.items(): 
            name = f"{self.type}hook/{k}"
            if self.type == "hist": pl_module.logger.experiment.add_histogram(name, values=v[-1], global_step=pl_module.current_epoch*(1+batch_idx))
            elif self.type == "percent": pl_module.log(name=name, value=v[-1], on_step=True)
            else: raise NotImplementedError(f"type:{self.type} is not implemented")
        
        for k in self.m.keys(): self.m[k] = []# making the list free of metrics as this is already logged.
        self.hook = False 
    
    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx): self.hook = True 
