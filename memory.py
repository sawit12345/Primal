import numpy as np
import hnswlib
import scipy.linalg as linalg

class LMU:
    def __init__(self, q=128, history_len=200):
        self.q = q
        self.theta = history_len
        # Analytic Legendre kernel initialization
        Q = np.arange(q, dtype=np.float64)
        R = (2 * Q + 1)[:, None] / self.theta
        j, i = np.meshgrid(Q, Q)
        A = np.where(j < i, -1, (-1.0) ** (i - j + 1)) * R
        B = (-1.0) ** Q[:, None] * R

        # Discretize using zero-order hold (ZOH) approx
        self.Ad = linalg.expm(A)
        self.Bd = np.linalg.inv(A) @ (self.Ad - np.eye(q)) @ B

        self.state = np.zeros(q)

    def step(self, u):
        # u is expected to be a scalar or averaged summary of cortical activation
        # To match blueprint: "LMU kernel matrix of dimension 128 by 256 to the cortical layer 5 activation"
        # We assume u is 256-d layer 5 state. Let's project it to scalar or use multi-dim LMU.
        # Simplest: use mean activation.
        u_scalar = np.mean(u)
        self.state = self.Ad @ self.state + self.Bd.flatten() * u_scalar
        return self.state

class DentateGyrus:
    def __init__(self, input_dim=384, output_dim=3840):
        self.input_dim = input_dim
        self.output_dim = output_dim
        # Fixed sparse random projection
        self.W = np.random.randn(output_dim, input_dim) / np.sqrt(input_dim)

    def process(self, x):
        # x: 256-d belief + 128-d A2 = 384
        proj = self.W @ x
        # Hard threshold keeping top 2 percent
        threshold = np.percentile(proj, 98)
        binary_out = (proj >= threshold).astype(np.float32)
        return binary_out

class CA3_CA1:
    def __init__(self, pattern_dim=3840):
        # Dense associative memory (Modern Hopfield / Ramsauer et al.)
        self.patterns = [] # Will store DG patterns

    def process(self, current_pattern):
        # CA3 Pattern completion
        if len(self.patterns) == 0:
            self.patterns.append(current_pattern)
            return current_pattern, 1.0 # mismatch = 1.0

        patterns_mat = np.stack(self.patterns)
        # Attention-like readout (Softmax(beta * X @ Y.T) @ Y)
        beta = 8.0
        scores = np.exp(beta * (patterns_mat @ current_pattern) - np.max(beta * (patterns_mat @ current_pattern)))
        scores /= np.sum(scores)
        retrieved_pattern = scores @ patterns_mat

        # CA1 Mismatch detection
        mismatch = np.linalg.norm(current_pattern - retrieved_pattern) / np.linalg.norm(current_pattern)

        return retrieved_pattern, mismatch

    def store(self, pattern):
        self.patterns.append(pattern)
        if len(self.patterns) > 10000:
            self.patterns.pop(0)

class HippocampalBuffer:
    def __init__(self, max_elements=10000, index_dim=704, tuple_dim=1152):
        self.max_elements = max_elements
        self.index_dim = index_dim
        self.tuple_dim = tuple_dim

        # HNSW Index
        self.index = hnswlib.Index(space='cosine', dim=index_dim)
        self.index.init_index(max_elements=max_elements, ef_construction=200, M=16)
        self.index.set_ef(50)

        self.tuples = []
        self.current_id = 0
        self.retention_scores = []

    def _create_index_vector(self, lmu_state, it_vector, context_tag):
        # 128 (LMU) + 512 (IT MAP one-hot) + 64 (context) = 704
        return np.concatenate([lmu_state, it_vector[:512], context_tag])

    def write(self, tuple_data, index_vector, novelty_score):
        if self.current_id < self.max_elements:
            self.index.add_items(index_vector, self.current_id)
            self.tuples.append(tuple_data)
            self.retention_scores.append(novelty_score)
            self.current_id += 1
        else:
            # Replace lowest retention score
            lowest_idx = np.argmin(self.retention_scores)
            self.index.add_items(index_vector, lowest_idx) # Overwrite in hnswlib
            self.tuples[lowest_idx] = tuple_data
            self.retention_scores[lowest_idx] = novelty_score

    def retrieve(self, partial_index_vector):
        if self.current_id == 0:
            return np.zeros(self.tuple_dim)

        labels, distances = self.index.knn_query(partial_index_vector, k=1)
        best_id = labels[0][0]
        # Increase retention score slightly on retrieval
        self.retention_scores[best_id] += 0.01
        return self.tuples[best_id]

class MemorySystem:
    def __init__(self):
        self.lmu = LMU()
        self.dg = DentateGyrus()
        self.ca3_ca1 = CA3_CA1()
        self.buffer = HippocampalBuffer()

    def process(self, l5_state, it_vector, belief, a2_vector, l6_state, pfc_goal, context_tag):
        # LMU update
        lmu_state = self.lmu.step(l5_state)

        # DG & CA3/CA1 (Pattern separation & Mismatch)
        # "input is 256-d belief distribution + 128-d a2"
        dg_input = np.concatenate([belief, a2_vector])
        dg_pattern = self.dg.process(dg_input)
        retrieved_dg, mismatch = self.ca3_ca1.process(dg_pattern)

        # Buffer operations
        # Tuple dimension = 256 (L5) + 512 (IT) + 256 (L6) + 64 (PFC) + 64 (Context) = 1152
        tuple_data = np.concatenate([l5_state, it_vector, l6_state, pfc_goal, context_tag])
        index_vector = self.buffer._create_index_vector(lmu_state, it_vector, context_tag)

        retrieved_tuple = self.buffer.retrieve(index_vector)

        # Norepinephrine controls writing (novelty mismatch > 0.6)
        norepinephrine = mismatch
        if norepinephrine > 0.6:
            self.buffer.write(tuple_data, index_vector, norepinephrine)
            self.ca3_ca1.store(dg_pattern)

        return retrieved_tuple, norepinephrine
