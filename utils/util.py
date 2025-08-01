import copy
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import IPython.display
from torch.utils.data import DataLoader


def view_10(img, label):
    """ view 10 labelled examples from tensor"""
    fig, axes = plt.subplots(2, 5, figsize=(10, 4))
    for i, ax in enumerate(axes.flat):
        ax.axis("off")
        ax.set_title(label[i].cpu().numpy())
        ax.imshow(img[i][0], cmap="gray")
    IPython.display.display(fig)
    plt.close(fig)

def view_10_color(img, label):
    """ View 10 labeled examples from tensor for color images """
    fig, axes = plt.subplots(2, 5, figsize=(12, 6))

    for i, ax in enumerate(axes.flat):
        ax.axis("off")
        ax.set_title(f"Label: {label[i].cpu().numpy()}")
        ax.imshow(img[i].permute(1, 2, 0).cpu().numpy())  # Assuming img is a PyTorch tensor
    IPython.display.display(fig)
    plt.close(fig)    


def num_params(model):
    """ """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


