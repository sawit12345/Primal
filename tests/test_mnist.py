import numpy as np
from primal.agent import PrimalAgent

def test_mnist_mvp():
    import urllib.request, gzip, os
    
    # Fallback downloader
    def load_mnist_fallback(path="/tmp/mnist"):
        os.makedirs(path, exist_ok=True)
        base = "https://storage.googleapis.com/cvdf-datasets/mnist/"
        files = {
            "train_images": "train-images-idx3-ubyte.gz",
            "train_labels": "train-labels-idx1-ubyte.gz",
            "test_images":  "t10k-images-idx3-ubyte.gz",
            "test_labels":  "t10k-labels-idx1-ubyte.gz",
        }
        for key, fname in files.items():
            fpath = os.path.join(path, fname)
            if not os.path.exists(fpath):
                urllib.request.urlretrieve(base + fname, fpath)

        def read_images(fname):
            with gzip.open(os.path.join(path, fname)) as f:
                f.read(16)
                return np.frombuffer(f.read(), dtype=np.uint8).reshape(-1, 784).astype(np.float32) / 255.0

        def read_labels(fname):
            with gzip.open(os.path.join(path, fname)) as f:
                f.read(8)
                return np.frombuffer(f.read(), dtype=np.uint8)

        return (
            read_images("train-images-idx3-ubyte.gz"),
            read_labels("train-labels-idx1-ubyte.gz"),
            read_images("t10k-images-idx3-ubyte.gz"),
            read_labels("t10k-labels-idx1-ubyte.gz"),
        )
        
    train_X, train_y, test_X, test_y = load_mnist_fallback()

    agent = PrimalAgent((28, 28), n_actions=10, is_mnist=True)
    agent.reset()
    
    # CRITICAL: disable BMR during learning
    agent.gmm.bmr_enabled = False
    agent.gmm.novelty_threshold = 0.0
    
    slot_to_class = {}
    print("Learning phase (1 sample per class)...")
    for label in range(10):
        sample = train_X[train_y == label][0].reshape(28, 28)
        agent.update(obs=sample, action=0, reward=1.0, next_obs=sample, done=False)
        
        # find which slot just opened
        features = agent.extract_features(sample)
        resp = agent.gmm.e_step(features)
        winning_slot = int(np.argmax(resp))
        slot_to_class[winning_slot] = label
        
    agent.gmm.bmr_enabled = True
    print(f"Slots after learning: {agent.gmm.n_components} (need exactly 10)")
    
    print("Testing on 10,000 samples...")
    correct = 0
    total = min(len(test_X), 10000)
    for i in range(total):
        sample = test_X[i].reshape(28, 28)
        
        # MVP classification
        winning_slot = agent.act(sample)
        predicted_label = slot_to_class.get(winning_slot, -1)
        
        if predicted_label == test_y[i]:
            correct += 1
            
        if (i+1) % 1000 == 0:
            print(f"Processed {i+1} samples, accuracy so far: {correct/(i+1):.4f}")

    accuracy = correct / total
    print(f"Final Accuracy: {accuracy:.4f} (need > 0.70 for MVP)")
    if accuracy > 0.70:
        print("PASS: MNIST MVP accuracy >= 70%")
    else:
        print("FAIL: MNIST MVP accuracy < 70%")

if __name__ == "__main__":
    test_mnist_mvp()
