# callbacks
A repo to quickly experiment with hyper parameters for a simple classification problem. 

- To train `python train.py --cfg_path="configs/exp1.py"`

##TODO:
- [x] Log forward values conv2d => hist, ReLU => %zeros 
- [ ] Log values before and after layer outputs in both forward and backward books. 
- [ ] performace transforms like `cutout` or `randomerase`
- [ ] can we reach 90+ accuracy.
- [ ] time different layers using callbacks 