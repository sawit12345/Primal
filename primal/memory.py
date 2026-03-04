import numpy as np
import scipy.linalg
import hnswlib

class LMUIndex:
    def __init__(self, history_len=200, lmu_dim=128, input_dim=256):
        self.history_len = history_len
        self.lmu_dim = lmu_dim
        self.input_dim = input_dim

        Q = np.arange(lmu_dim, dtype=np.float64)
        R = (2*Q + 1)[:, None] / history_len
        j, i = np.meshgrid(Q, Q)
        A = np.where(i < j, -1, (-1.0)**(i - j + 1)) * R
        B = (-1.0)**Q[:, None] * R

        self.Ad = scipy.linalg.expm(A)
        self.Bd = np.linalg.inv(A).dot(self.Ad - np.eye(lmu_dim)).dot(B)

        self.state = np.zeros(lmu_dim)

    def process(self, x):
        if self.Bd.shape[1] != self.input_dim:
            self.W_in = np.random.randn(1, self.input_dim) / np.sqrt(self.input_dim)
            x_proj = np.dot(self.W_in, x)
            self.state = np.dot(self.Ad, self.state) + np.dot(self.Bd, x_proj).flatten()
        else:
            self.state = np.dot(self.Ad, self.state) + np.dot(self.Bd, x).flatten()

        return self.state

class DentateGyrus:
    def __init__(self, in_dim=384, out_dim=3840, sparsity=0.02):
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.sparsity = sparsity
        self.W = np.random.randn(out_dim, in_dim) / np.sqrt(in_dim)

    def process(self, x):
        proj = np.dot(self.W, x)
        threshold_idx = int((1 - self.sparsity) * self.out_dim)
        if threshold_idx >= self.out_dim: threshold_idx = self.out_dim - 1
        threshold_val = np.partition(proj, threshold_idx)[threshold_idx]
        output = np.where(proj >= threshold_val, 1.0, 0.0)
        return output

class HippocampalBuffer:
    def __init__(self, max_elements=200000):
        self.index_dim = 704
        self.tuple_dim = 1152
        self.max_elements = max_elements

        self.index = hnswlib.Index(space='cosine', dim=self.index_dim)
        self.index.init_index(max_elements=max_elements, ef_construction=400, M=64)

        self.tuples = np.zeros((max_elements, self.tuple_dim))
        self.social_flags = np.zeros(max_elements, dtype=bool)
        self.retention_scores = np.zeros(max_elements)
        self.novelty_scores = np.zeros(max_elements)

        self.current_size = 0
        self.write_idx = 0
        self.step_counter = 0

    def write(self, index_vec, tuple_vec, novelty_score=0.0, is_social=False):
        if self.current_size >= self.max_elements:
            idx = np.argmin(self.retention_scores)
        else:
            idx = self.current_size
            self.current_size += 1

        self.tuples[idx] = tuple_vec
        self.social_flags[idx] = is_social
        self.novelty_scores[idx] = novelty_score

        self.retention_scores[idx] = self.step_counter

        self.index.add_items(index_vec, np.array([idx]))
        self.write_idx = idx
        self.step_counter += 1
        return idx

    def query(self, partial_index_vec, k=1):
        if self.current_size == 0:
            return None, None

        labels, distances = self.index.knn_query(partial_index_vec, k=k)
        retrieved_idx = labels[0][0]

        self.retention_scores[retrieved_idx] += 1

        return retrieved_idx, self.tuples[retrieved_idx]

    def prune(self):
        if self.current_size < 100:
            return
        num_to_prune = int(self.current_size * 0.1)
        indices_to_prune = np.argsort(self.retention_scores)[:num_to_prune]
        for idx in indices_to_prune:
            self.index.mark_deleted(idx)
            self.retention_scores[idx] = -np.inf

class CA1Mismatch:
    def __init__(self, threshold=0.6):
        self.threshold = threshold

    def process(self, retrieved_pattern, current_pattern):
        diff = retrieved_pattern - current_pattern
        norm = np.linalg.norm(diff)
        mismatch_score = norm / (np.linalg.norm(current_pattern) + 1e-8)
        write_flag = mismatch_score > self.threshold
        return write_flag, mismatch_score

class SpatialMap:
    def __init__(self, grid_size=64, cell_dim=32):
        self.grid_size = grid_size
        self.cell_dim = cell_dim
        self.egocentric_map = np.zeros((grid_size, grid_size, cell_dim))
        self.allocentric_map = np.zeros((grid_size, grid_size, cell_dim))
        self.rsc_transform = np.random.randn(grid_size*grid_size, grid_size*grid_size) / (grid_size)

    def update_attended(self, loc_x, loc_y, feature_vec):
        if 0 <= loc_x < self.grid_size and 0 <= loc_y < self.grid_size:
            self.egocentric_map[loc_x, loc_y] += 0.1 * (feature_vec - self.egocentric_map[loc_x, loc_y])

    def rsc_update(self, vestibular_input, position_error):
        pass
