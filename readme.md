# Towards Initialization-dependent and Non-vacuous  Generalization Bounds for Overparameterized Shallow Neural Networks
This repository contains the code to train neural networks and calculate measures and generalization bounds shown in the Empirical Studies section in the paper "Towards Initialization-dependent and Non-vacuous  Generalization Bounds for Overparameterized Shallow Neural Networks".

## Requirements
```
torch >= 2.5.1
torch >= 0.20.1
numpy >= 2.0.2
```

## Usage
As a sample, the following command is used to train a 2-layer shallow neural networks with widths range from $2^6$ to $2^{18}$ on MNIST dataset for a binary classification (1 or 7) and output norms, capacity, and generalization bounds derived in existing studies and our paper:

```
python main_binary.py --dataset MNIST --nclasses 1 \
                      --nunits 64,128,256,512,1024,2048,4096,8192,16384,32768,65536,131072,262144 \              
                      --epochs 20 --batchsize 256 --learningrate 0.001 --momentum 0.9 --stopcond 0.1 --random 42
```

The input arguments are given as follows:

```
usage: main_binary.py [-h] 
    [--dataset  { MNIST, ijcnn, CIFAR10, CIFAR100, SVHN, RCV1, GISETTE, w1a, a1a}] 
    [--nclasses  Number of classes]  
    [--nunits_list  Comma-separated list of hidden units] 
    [--epochs  Epoch] 
    [--stopcond  Stopping condtion] 
    [--batchsize  Batch size] 
    [--momentum  Momentum] 
    [--learningrate  Learning rate] 
    [--random_list  Comma-separated list of random seeds] 
    [--datadir] [--outputdir] [--modeldir] [--no-cuda]
```

## Output Bounds derived in our paper
- Path norm:
  - "standard_path_norm"
  - "path_norm"
- Capacity bounds:
  - "Our capacity (std path norm)"
  - "Our capacity"
- Generalization bounds:
  - "Our generalization (std path norm)"
  - "Our generalization"
- Simplified generalization bounds:
    - "Our simp"
