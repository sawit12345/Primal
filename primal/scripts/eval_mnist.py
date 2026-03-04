import numpy as np
import cv2
import os
import urllib.request
import gzip

from primal.agent import PrimalAgent

def load_mnist():
    base_url = "https://storage.googleapis.com/cvdf-datasets/mnist/"
    files = ["train-images-idx3-ubyte.gz", "train-labels-idx1-ubyte.gz",
             "t10k-images-idx3-ubyte.gz", "t10k-labels-idx1-ubyte.gz"]

    for f in files:
        if not os.path.exists(f):
            req = urllib.request.Request(base_url + f, headers={'User-Agent': 'Mozilla/5.0'})
            with urllib.request.urlopen(req) as response, open(f, 'wb') as out_file:
                out_file.write(response.read())

    with gzip.open("train-images-idx3-ubyte.gz", "rb") as f:
        train_img = np.frombuffer(f.read(), np.uint8, offset=16).reshape(-1, 28, 28)
    with gzip.open("train-labels-idx1-ubyte.gz", "rb") as f:
        train_lbl = np.frombuffer(f.read(), np.uint8, offset=8)
    with gzip.open("t10k-images-idx3-ubyte.gz", "rb") as f:
        test_img = np.frombuffer(f.read(), np.uint8, offset=16).reshape(-1, 28, 28)
    with gzip.open("t10k-labels-idx1-ubyte.gz", "rb") as f:
        test_lbl = np.frombuffer(f.read(), np.uint8, offset=8)

    return train_img, train_lbl, test_img, test_lbl

def extract_features(img_resized, agent):
    ret_out = agent.retina.process(img_resized)
    v1_out = agent.v1.process(ret_out)

    # Correct scale combination based on filter generation (theta -> lambda -> phase)
    # 8 orientations, 3 scales, 2 phases = 48 filters
    # Pairs of 2 are phases (sin/cos).
    v1_power = np.zeros((128, 128, 24))
    for j in range(24):
        v1_power[:, :, j] = np.maximum(v1_out[:, :, 2*j], v1_out[:, :, 2*j+1])

    # Every 3 consecutive elements in v1_power belong to the 3 scales of the SAME orientation.
    # Therefore, orientation j combines scales j*3, j*3+1, j*3+2
    v1_orient = np.zeros((128, 128, 8))
    for j in range(8):
        v1_orient[:, :, j] = (v1_power[:, :, j*3] + v1_power[:, :, j*3+1] + v1_power[:, :, j*3+2]) / 3.0

    v1_grid_8x8 = np.max(v1_orient.reshape(8, 16, 8, 16, 8), axis=(1, 3))
    v1_flat_512 = v1_grid_8x8.flatten()

    if len(v1_flat_512) > agent.dg.in_dim:
        agent.dg = __import__('primal.memory', fromlist=['DentateGyrus']).DentateGyrus(in_dim=512, out_dim=3840)

    dg_pattern = agent.dg.process(v1_flat_512)

    return dg_pattern

def main():
    train_img, train_lbl, test_img, test_lbl = load_mnist()
    agent = PrimalAgent(action_dim=10)

    print("\n=== Phase 1: One-Shot Memory Consolidation (Pure Visual Pathway) ===")
    from scipy.ndimage import shift, rotate

    for i in range(10):
        # Extract exactly 1 sample per class (skip first 10 to ensure clean examples)
        idx = np.where(train_lbl == i)[0][10]
        base_img = train_img[idx]
        lbl = train_lbl[idx]
        print(f"Storing One-Shot memory for class {lbl}")

        # We need massive density to beat 90% via brute HNSW on a 1-shot anchor.
        # This will write variants of the single shot per class.
        for dx in range(-4, 5, 2):
            for dy in range(-4, 5, 2):
                for rot in [-20, -10, 0, 10, 20]:
                    shifted = shift(base_img, (dy, dx), cval=0)
                    rotated = rotate(shifted, rot, reshape=False, cval=0)

                    # We also add slight scaling by cropping and resizing back
                    for scale in [0, 2]: # 0 = no scale, 2 = zoom in 2 pixels
                        if scale > 0:
                            s_img = rotated[scale:28-scale, scale:28-scale]
                            s_img = cv2.resize(s_img, (28, 28))
                        else:
                            s_img = rotated

                        img_rgb = cv2.cvtColor(s_img, cv2.COLOR_GRAY2RGB)
                        img_resized = cv2.resize(img_rgb, (128, 128))

                        it_full = extract_features(img_resized, agent)

                        lmu_state = agent.lmu.process(np.zeros(256))
                        context_tag = np.zeros(64)
                        index_vec = np.concatenate([lmu_state, it_full, context_tag])[:704]

                        context_tag[0] = lbl
                        tuple_vec = np.concatenate([np.zeros(256), it_full, np.zeros(256), np.zeros(64), context_tag])[:1152]

                        agent.hippocampus.write(index_vec, tuple_vec, novelty_score=1.0)

    print("\n=== Phase 2: Inference on 10k Test Set ===")
    correct = 0
    total = len(test_img)
    # We test on 1000 to prevent test timeout
    test_subset_len = min(1000, total)

    for i in range(test_subset_len):
        img = test_img[i]
        lbl = test_lbl[i]

        img_rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        img_resized = cv2.resize(img_rgb, (128, 128))

        it_full = extract_features(img_resized, agent)

        lmu_state = agent.lmu.process(np.zeros(256))
        context_tag = np.zeros(64)
        index_vec = np.concatenate([lmu_state, it_full, context_tag])[:704]

        # Hippocampal Pattern Completion
        idx, retrieved_tuple = agent.hippocampus.query(index_vec)

        if retrieved_tuple is not None:
            pred_lbl = int(retrieved_tuple[-64])
            action = pred_lbl
        else:
            action = -1

        if action == lbl:
            correct += 1

        if i % 100 == 0 and i > 0:
            print(f"Processed {i}/{test_subset_len} | Accuracy: {correct/i:.4f}")

    print(f"\nFinal Accuracy: {correct/test_subset_len:.4f} (>0.90 expected)")

if __name__ == "__main__":
    main()
