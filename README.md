# Federated Learning Baselines with FedFeat

This repository provides implementations and experimental settings for various federated learning methods with and without **FedFeat**, evaluated across multiple model architectures and datasets.

## 📚 Available Baselines

- **FedAvg**
- **FedProx**
- **MOON**
- **FedMa**
- **FedOpt**
- **FedNova**
- **FedDyn**

### ✅ FedFeat: Module
Enable with: `--fedfeat True`  
Disable with: `--fedfeat False`

---

## 🧠 Model Architectures

| Model Name | Argument |
|------------|----------|
| MLP        | `mlp`    |
| CNN        | `cnn`    |
| ResNet18   | `resnet` |
| ViT-B-16   | `vit`    |

---

## 📊 Datasets

| Dataset         | Argument    |
|-----------------|-------------|
| CIFAR-10        | `cifar10`   |
| CIFAR-100       | `cifar100`  |
| MNIST           | `mnist`     |
| Fashion-MNIST   | `fashionmnist`    |

## 📚 Dataset Specifications

- **MNIST**
  - `num_classes`: `10`
  - `data_shape`: `[1, 28, 28]`

- **Fashion-MNIST**
  - `num_classes`: `10`
  - `data_shape`: `[1, 28, 28]`

- **CIFAR-10**
  - `num_classes`: `10`
  - `data_shape`: `[3, 32, 32]`

- **CIFAR-100**
  - `num_classes`: `100`
  - `data_shape`: `[3, 32, 32]`

---

## ⚙️ Learning Rate Settings

### 🔹 CIFAR-10 / CIFAR-100

| Baseline  | Model(s)             | Local LR (`lr`) | Global LR (`g_lr`) |
|-----------|----------------------|------------------|---------------------|
| FedAvg    | MLP, CNN, ResNet     | 0.05             | -                   |
| FedAvg    | FedFeat              | 0.05             | 0.001               |
| FedProx   | MLP, CNN, ResNet     | 0.05             | -                   |
| FedProx   | FedFeat              | -                | 0.001               |
| MOON      | MLP, CNN, ResNet     | 0.05             | -                   |
| MOON      | ResNet (CIFAR-100)   | 0.01             | -                   |
| MOON      | FedFeat              | -                | 0.0001              |
| FedMa     | MLP, CNN, ResNet     | 0.05             | -                   |
| FedMa     | FedFeat              | -                | 0.001               |
| FedOpt    | MLP, CNN, ResNet     | 0.05             | -                   |
| FedOpt    | FedFeat              | -                | 0.0001              |
| FedNova   | MLP                  | 0.05             | -                   |
| FedNova   | CNN, ResNet          | 0.01             | -                   |
| FedNova   | FedFeat              | -                | 0.001               |
| FedDyn    | MLP, CNN             | 0.05             | -                   |
| FedDyn    | ResNet (CIFAR-100)   | 0.01             | -                   |
| FedDyn    | FedFeat              | -                | 0.001               |
| **All**   | ViT                  | 0.05             | 0.001               |

---

### 🔹 MNIST / Fashion MNIST

- For **MLP**:
  - Local LR (`lr`) = **0.01**
  - Global LR (`g_lr`) = **0.001**

- For **CNN** and **ResNet**:
  - Local LR (`lr`) = **0.01**
  - Global LR (`g_lr`) = **0.001**
  - for MOON Global LR (`g_lr`) = **0.001** / **0.0001**
---


📦 Dependencies
The following libraries are required:

torch==1.8.1

torchvision==0.6.0

timm

numpy

scipy

matplotlib

tqdm

ipython

## 🚀 Example Usage

```bash
# Example: Useage
 python main.py --config=exps/fedfeat.json

 
```bash
sh main.sh
