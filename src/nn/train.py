from typing import TypedDict
import pickle
import matplotlib.pyplot as plt
import os
import numpy as np

class CifarBatch(TypedDict):
    batch_label: str
    labels: list[int]
    data: np.ndarray
    filenames: list[str]

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

def readCifarBatch(fileName: str) -> CifarBatch:
    # Load the original data with byte keys
    with open(os.path.join(CURRENT_DIR, '..', '..', 'data', fileName), 'rb') as fo:
        raw = pickle.load(fo, encoding='bytes')

    # Convert keys and values to normal Python types
    return {
        "batch_label": raw[b"batch_label"].decode(),
        "labels": raw[b"labels"],
        "data": np.array(raw[b"data"]),
        "filenames": [name.decode() for name in raw[b"filenames"]]
    }

def saveImage(img_flat: np.ndarray, fileName: str):
    img = img_flat.reshape(3, 32, 32).transpose(1, 2, 0)

    plt.imshow(img)
    plt.axis("off")
    plt.savefig(os.path.join(CURRENT_DIR, '..', '..', 'public', fileName))

trainBatch1 = readCifarBatch('data_batch_1')

print(trainBatch1.keys())

saveImage(trainBatch1["data"][2], trainBatch1["filenames"][2])