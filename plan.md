1. Setup environment and basic structure.
   - Define config and hyperparameters (128x128 input, etc.).
   - Create main `Agent` class and loop.
2. Implement Sensory Pipelines.
   - Retina (DoG, Color Opponency).
   - V1 (Gabor filter bank).
   - V2-IT (RGM hierarchy or a functional equivalent using sparse dictionary learning / VMP if possible).
   - V5 (Lucas-Kanade or Spatiotemporal Gabor).
   - Auditory (Mel filterbank, zeroed for these tasks).
3. Implement Cortical & Subcortical structures.
   - Thalamus routing.
   - Cortical predictive coding hierarchy (Layers 1-6).
   - LMU Entorhinal Index.
   - Dentate Gyrus & CA3 (Pattern separation & completion).
   - Hippocampal Buffer (HNSW via `hnswlib`).
4. Implement Action Selection & Modulators.
   - Basal Ganglia (TD-lambda).
   - Neuromodulators (DA, NE, 5-HT, ACh).
   - Amygdala, Cerebellum, Insula, OFC, SMA.
5. Integration & Training Loop.
   - Connect components according to "UPDATE SCHEDULE".
   - Implement "Spelke's priors" as initial weights or fixed biases.
   - Handle replay (Levels 1-3).
6. Testing.
   - Gymnasium Breakout test (render or headless).
   - MNIST test (1 sample per class, 10-shot total).
7. Pre-commit & Submit.
   - Use pre_commit_instructions tool to verify.
   - Submit the branch.