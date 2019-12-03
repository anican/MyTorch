# MyTorch
Personal computing framework for deep learning. 
## `nn` Package
### Linear Layers
- [x] `nn.Linear()`
### Convolutional Layers
- [ ] `nn.Conv1d()`
- [ ] `nn.Conv2d()`
- [ ] `nn.Conv3d()`
- [ ] `nn.ConvTranspose2d()`
### Non-Linear Activation Layers
- [ ] `nn.Sigmoid()`
- [ ] `nn.ReLU()`
- [ ] `nn.LeakyReLU()`
- [ ] `nn.LogSigmoid()`
- [ ] `nn.PReLU()`
- [ ] `nn.Tanh()`
- [ ] `nn.Softplus()`
### Loss Layers
- [ ] `nn.CrossEntropyLoss()`
- [ ] `nn.MSELoss()`
- [ ] `nn.RMSELoss()`
### Pooling Layers
- [ ] `nn.MaxPool2d()`
- [ ] `nn.MaxPool3d()`
### Normalization Layers
- [ ] `nn.BatchNorm1d()`
- [ ] `nn.BatchNorm2d()`
## `optim` Package
- [x] `optim.SGD()`
- [ ] `optim.Adam()`
- [ ] learning rate decay scheduler
## Tests

## Installation 
First install Miniconda (https://docs.conda.io/en/latest/miniconda.html).

```bash
conda create -n my-torch python=3.6.9
git clone <insert-repository-link>
cd <insert-repository-link>
conda deactivate
conda env update -n my-torch -f environment.yml
conda activate dl-class
```
