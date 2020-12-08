# Contents
- **utils.py** : A code containing common codes
- **train.py** : A normal training code without 'raytune'
- **train_tune.py** : A training code with 'raytune'

# Required libraries
- **pytorch**
- **raytune** <br>-Installation:
```
$ pip install 'ray[tune]'
```

# Run
### 1. clone
```bash
$ git clone https://github.com/machine-intelligence-lab/raytune_example.git
```
### 2. run
<train.py>
```bash
$ python3 train.py
```

<train_tune.py>
```bash
$ CUDA_VISIBLE_DEVICES=x,x python3 train_tune.py
```
- It runs total 5 trials and each trial runs maximum 30 epochs
- It assigns 2 CPUs and 0.25 GPUs to each trial (That is, 4 trials share one gpu)
- It saves logs into ".ray_result/*expr_name*" <br> (expr_name has a name like 'DEFAULT_yyyy_mm_dd_hh_mm_ss')

# Tensorboard
If tensorboard is installed, you can visualize trial results\
Command:
```bash
$ tensorboard --logdir=.ray_result/expr_name
```
If you run trials on a server, then it might be more convinient just copying the experiment directory to your local computer where a web-browser works
