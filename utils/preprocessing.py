import numpy as np

def normalize(x):
    return (x - x.min()) / (x.max() - x.min() + 1e-8)