from typing import TypedDict
import pickle
import matplotlib.pyplot as plt
import os
import numpy as np
import copy

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

def initialize_with_zeros(dim):
    w = np.zeros((dim, 1))
    b = 0

    return w, b

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def propagate(w, b, X, Y):
    m = X.shape[1]

    A = sigmoid(np.sum(X * w, axis = 0, keepdims = True) + b)
    cost = (np.dot(Y, np.log(A).T) + (np.dot((1 - Y), np.log(1 - A).T))) / -m

    dw = np.dot(X, (A - Y).T) / m
    db = np.sum(A - Y) / m

    cost = np.squeeze(np.array(cost))

    grads = {"dw": dw,
             "db": db}

    return grads, cost


def optimize(w, b, X, Y, num_iterations=100, learning_rate=0.01, print_cost=True):
    w = copy.deepcopy(w)
    b = copy.deepcopy(b)

    costs = []

    for i in range(num_iterations):
        grads, cost = propagate(w, b, X, Y)

        dw = grads["dw"]
        db = grads["db"]

        w -= (dw * learning_rate)
        b -= (db * learning_rate)

        if i % 100 == 0:
            costs.append(cost)

            # Print the cost every 100 training iterations
            if print_cost:
                print("Cost after iteration %i: %f" % (i, cost))

    params = {"w": w,
              "b": b}

    grads = {"dw": dw,
             "db": db}

    return params, grads, costs


def predict(w, b, X):
    A = sigmoid(np.sum(X * w, axis=0, keepdims=True) + b)
    return (A > 0.5).astype(int).squeeze()

trainBatch1 = readCifarBatch('data_batch_1')
trainBatch2 = readCifarBatch('data_batch_2')
trainBatch3 = readCifarBatch('data_batch_3')
trainBatch4 = readCifarBatch('data_batch_4')
trainBatch5 = readCifarBatch('data_batch_5')
testBatch = readCifarBatch('test_batch')

Y = np.array([1 if label == 3 else 0 for label in (trainBatch1["labels"] + trainBatch2["labels"] + trainBatch3["labels"] + trainBatch4["labels"]) + trainBatch5["labels"]])
Y_test = np.array([1 if label == 3 else 0 for label in testBatch["labels"]])
X = np.concatenate([trainBatch1["data"].T, trainBatch2["data"].T, trainBatch3["data"].T, trainBatch4["data"].T, trainBatch5["data"].T], axis=1) / 255
X_test = testBatch["data"].T / 255
w,b = initialize_with_zeros(X.shape[0])

# none_cat_indexes = np.where(Y == 0)[0]
# cat_indexes = np.where(Y == 1)[0]
#
# np.random.seed(42)
# num_to_keep = len(cat_indexes)
# selected_none_cat_indexes = np.random.choice(none_cat_indexes, size=num_to_keep * 2, replace=False)
#
# balanced_indices = np.concatenate([cat_indexes, selected_none_cat_indexes])
# balanced_indices.sort()
#
# X_balanced = X[:, balanced_indices]
# Y_balanced = Y[balanced_indices]

params, grads, costs = optimize(w, b, X, Y, 2000)
predictions = predict(params["w"], params["b"], X_test)

print(np.where(predictions == 1)[0])

# saveImage((X_balanced.T[0] * 255).astype(np.uint8), testBatch["filenames"][438])
# saveImage((X_balanced.T[1] * 255).astype(np.uint8), testBatch["filenames"][439])
# saveImage((X_balanced.T[10] * 255).astype(np.uint8), testBatch["filenames"][440])
# saveImage((X_balanced.T[9997] * 255).astype(np.uint8), testBatch["filenames"][441])
# saveImage((X_balanced.T[9998] * 255).astype(np.uint8), testBatch["filenames"][442])
# saveImage((X_balanced.T[9999] * 255).astype(np.uint8), testBatch["filenames"][442])

# saveImage((X_balanced.T[2] * 255).astype(np.uint8), testBatch["filenames"][438])
# saveImage((X_balanced.T[3] * 255).astype(np.uint8), testBatch["filenames"][439])
# saveImage((X_balanced.T[4] * 255).astype(np.uint8), testBatch["filenames"][440])
# saveImage((X_balanced.T[9994] * 255).astype(np.uint8), testBatch["filenames"][441])
# saveImage((X_balanced.T[9995] * 255).astype(np.uint8), testBatch["filenames"][442])
# saveImage((X_balanced.T[9996] * 255).astype(np.uint8), testBatch["filenames"][442])

# saveImage(testBatch["data"][438], testBatch["filenames"][438])
# saveImage(testBatch["data"][2698], testBatch["filenames"][2698])
# saveImage(testBatch["data"][5441], testBatch["filenames"][5441])
# saveImage(testBatch["data"][8183], testBatch["filenames"][8183])
# saveImage(testBatch["data"][9084], testBatch["filenames"][9084])