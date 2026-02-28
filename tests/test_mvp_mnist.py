"""MVP MNIST test for Phase -1."""

import numpy as np
from primal.agent import PrimalAgent


def load_mnist():
    """Load MNIST dataset using sklearn or fallback."""
    try:
        from sklearn.datasets import fetch_openml
        print("Loading MNIST via fetch_openml...")
        mnist = fetch_openml("mnist_784", version=1, parser="auto")
        X = mnist.data.values if hasattr(mnist.data, 'values') else mnist.data
        y = mnist.target.values if hasattr(mnist.target, 'values') else mnist.target
        y = y.astype(np.uint8)
        
        # Split train/test
        X_train, X_test = X[:60000], X[60000:]
        y_train, y_test = y[:60000], y[60000:]
        
        return X_train, y_train, X_test, y_test
        
    except Exception as e:
        print(f"fetch_openml failed: {e}")
        print("Using fallback MNIST loader...")
        return load_mnist_fallback()


def load_mnist_fallback(path="/tmp/mnist"):
    """Fallback MNIST loader using direct download."""
    import os
    import urllib.request
    import gzip
    
    os.makedirs(path, exist_ok=True)
    base = "https://storage.googleapis.com/cvdf-datasets/mnist/"
    files = {
        "train_images": "train-images-idx3-ubyte.gz",
        "train_labels": "train-labels-idx1-ubyte.gz",
        "test_images": "t10k-images-idx3-ubyte.gz",
        "test_labels": "t10k-labels-idx1-ubyte.gz",
    }
    
    for key, fname in files.items():
        fpath = os.path.join(path, fname)
        if not os.path.exists(fpath):
            print(f"  Downloading {fname}...")
            urllib.request.urlretrieve(base + fname, fpath)

    def read_images(fname):
        with gzip.open(os.path.join(path, fname)) as f:
            f.read(16)
            data = np.frombuffer(f.read(), dtype=np.uint8).reshape(-1, 784)
            return data.astype(np.float32) / 255.0

    def read_labels(fname):
        with gzip.open(os.path.join(path, fname)) as f:
            f.read(8)
            return np.frombuffer(f.read(), dtype=np.uint8)

    X_train = read_images("train-images-idx3-ubyte.gz")
    y_train = read_labels("train-labels-idx1-ubyte.gz")
    X_test = read_images("t10k-images-idx3-ubyte.gz")
    y_test = read_labels("t10k-labels-idx1-ubyte.gz")
    
    return X_train, y_train, X_test, y_test


def train_agent_on_samples(agent, X_train, y_train):
    """Train agent with 1 sample per class."""
    print("\nTraining on 1 sample per class...")
    
    slot_to_class = {}
    
    for label in range(10):
        # Get first sample of this class
        idx = np.where(y_train == label)[0][0]
        sample = X_train[idx].reshape(28, 28)
        
        print(f"  Training on digit {label}...")
        
        # Update agent with this sample
        for _ in range(5):  # Multiple updates to reinforce
            action = agent.act(sample)
            fe = agent.update(sample, action, 1.0, sample, False)
        
        # Find which slot was most activated
        features = agent.extract_features(sample)
        resp = agent.gmm.e_step(features)
        winning_slot = int(np.argmax(resp))
        slot_to_class[winning_slot] = label
        
        print(f"    Assigned to slot {winning_slot}, FE={fe:.2f}")
    
    print(f"\nSlots after learning: {agent.gmm.n_components}")
    print(f"Slot-to-class mapping: {slot_to_class}")
    
    return slot_to_class


def test_agent(agent, X_test, y_test, slot_to_class):
    """Test agent on all 10,000 test samples."""
    print("\nTesting on 10,000 samples...")
    
    correct = 0
    unknown = 0
    n_test = len(X_test)
    
    for i in range(n_test):
        sample = X_test[i].reshape(28, 28)
        true_label = y_test[i]
        
        # Get prediction
        features = agent.extract_features(sample)
        resp = agent.gmm.e_step(features)
        winning_slot = int(np.argmax(resp))
        
        if winning_slot in slot_to_class:
            pred_label = slot_to_class[winning_slot]
            if pred_label == true_label:
                correct += 1
        else:
            unknown += 1
        
        # Progress
        if (i + 1) % 1000 == 0:
            acc = correct / (i + 1)
            print(f"  Processed {i+1}/{n_test}, accuracy so far: {acc:.3f}")
    
    accuracy = correct / n_test
    print(f"\nFinal accuracy: {accuracy:.3f} ({correct}/{n_test})")
    print(f"Unknown predictions: {unknown}")
    
    return accuracy


def main():
    print("=" * 60)
    print("MVP MNIST TEST - Phase -1")
    print("=" * 60)
    
    # Load data
    X_train, y_train, X_test, y_test = load_mnist()
    print(f"Train: {X_train.shape}, Test: {X_test.shape}")
    
    # Create agent
    agent = PrimalAgent(
        obs_shape=(28, 28),
        n_actions=10,
        mode='mnist'
    )
    print(f"Agent initialized: {agent.feature_dim} feature dimensions")
    
    # Train
    slot_to_class = train_agent_on_samples(agent, X_train, y_train)
    
    # Test
    accuracy = test_agent(agent, X_test, y_test, slot_to_class)
    
    # Results
    print("\n" + "=" * 60)
    print("MVP MNIST RESULTS")
    print("=" * 60)
    
    # Criteria check
    acc_pass = accuracy > 0.70
    
    status = "PASS" if acc_pass else "FAIL"
    print(f"{status}: accuracy > 70% (got {accuracy*100:.1f}%)")
    
    print("=" * 60)
    
    return acc_pass


if __name__ == "__main__":
    main()
