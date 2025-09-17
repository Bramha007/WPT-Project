import os, json
import numpy as np
import matplotlib.pyplot as plt

def save_size_hist(sides, out_path="outputs/size_hist.png"):
    plt.figure(figsize=(5,4))
    plt.hist(sides, bins=[0,12,24,48,96,192,384], edgecolor="k")
    plt.xlabel("Side length (px)"); plt.ylabel("Count"); plt.title("Square size histogram")
    plt.tight_layout(); plt.savefig(out_path, dpi=140, bbox_inches="tight"); plt.close()
    return out_path
