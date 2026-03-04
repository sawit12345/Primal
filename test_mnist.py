import numpy as np
import urllib.request
import gzip
import os
from agent import Agent

def load_mnist_images(filename):
    with gzip.open(filename, 'rb') as f:
        data = np.frombuffer(f.read(), np.uint8, offset=16)
    return data.reshape(-1, 28, 28)

def load_mnist_labels(filename):
    with gzip.open(filename, 'rb') as f:
        data = np.frombuffer(f.read(), np.uint8, offset=8)
    return data

def download_mnist():
    base_url = "https://storage.googleapis.com/cvdf-datasets/mnist/"
    files = [
        "train-images-idx3-ubyte.gz",
        "train-labels-idx1-ubyte.gz",
        "t10k-images-idx3-ubyte.gz",
        "t10k-labels-idx1-ubyte.gz"
    ]
    for filename in files:
        if not os.path.exists(filename):
            print(f"Downloading {filename}...")
            urllib.request.urlretrieve(base_url + filename, filename)

def test_mnist():
    print("Preparing MNIST data...")
    download_mnist()

    train_images = load_mnist_images("train-images-idx3-ubyte.gz")
    train_labels = load_mnist_labels("train-labels-idx1-ubyte.gz")
    test_images = load_mnist_images("t10k-images-idx3-ubyte.gz")
    test_labels = load_mnist_labels("t10k-labels-idx1-ubyte.gz")

    agent = Agent()

    print("Training 1-shot per class...")
    # Find one sample per class
    classes_found = set()
    for i in range(len(train_labels)):
        lbl = train_labels[i]
        if lbl not in classes_found:
            classes_found.add(lbl)
            agent.train_mnist_shot(train_images[i], lbl)
            print(f"Trained class {lbl}")
        if len(classes_found) == 10:
            break

    print("Evaluating on the full 10k test set...")
    correct = 0
    num_test = len(test_labels)
    for i in range(num_test):
        pred = agent.predict_mnist(test_images[i])
        if pred == test_labels[i]:
            correct += 1

        if (i+1) % 1000 == 0:
            print(f"Evaluated {i+1}/{num_test} samples, current accuracy: {correct / (i+1) * 100:.2f}%")

    print(f"Accuracy on tested samples: {correct / num_test * 100:.2f}%")

if __name__ == "__main__":
    test_mnist()
