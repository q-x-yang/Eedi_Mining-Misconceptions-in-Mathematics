# Eedi - Mining Misconceptions in Mathematics

This project tackles the Kaggle competition "Eedi - Mining Misconceptions in Mathematics". The goal is to infer which misconceptions a student is likely demonstrating when answering a mathematics question incorrectly, using the question text, answer choices, and metadata such as subject and construct.

## Problem summary

The dataset contains mathematics questions and student answer options. Each incorrect answer can be linked to one or more misconceptions, and the task is to rank likely misconceptions for each question-answer pair. In other words, this is a retrieval/ranking problem: for a given question-context + answer, find the misconception concepts most likely to explain the mistake.

This is a challenging multi-label semantic matching task because misconceptions are nuanced, and the same mathematical mistake can be expressed in different wording across questions.

## Solution workflow

The workflow is built around sentence embeddings and nearest-neighbor retrieval:

1. Preprocess each question into a unified text string containing:
   - construct name
   - subject name
   - question text
   - selected answer text
   - a lightweight answer tag such as A/B/C/D

2. Concatenate these fields into a single natural-language representation:
   - `<Construct> ... <Subject> ... <Question> ... <Answer> ...`

3. Encode both the generated question-answer texts and the misconception names using transformer-based sentence embedding models.

4. Use FAISS to retrieve the nearest misconception embeddings for each question-answer embedding.

5. Combine several model outputs through weighted ensembling to improve retrieval quality.

6. Produce the final submission by selecting the top K misconception IDs for each incorrect answer instance.

## Model and logic

The final solution uses an ensemble of three sentence-transformer models:

- `BAAI/bge-large-en-v1.5`
- `sentence-transformers/all-mpnet-base-v2`
- `Alibaba-NLP/gte-base-en-v1.5`

Each model is used to encode:
- the formatted question-answer text
- the misconception labels/names

The embeddings are normalized and merged with weighted combination:

- GTE: 0.718365
- BGE: 0.198723
- MPNet: 0.310873

These weights were tuned to maximize leaderboard performance. A nearest-neighbor search over the misconception embedding space then retrieves the most relevant misconceptions for each question-answer pair.

A major practical detail in this competition was model compatibility: the GTE and MPNet embeddings had smaller dimensionality than the target representation, so the vectors were padded to a consistent 1024-dimensional space before FAISS indexing.

## Key tuned parameters

The notebook and experiments used these core settings:

- `K = 25` nearest misconceptions retrieved per sample
- `BS = 16` encoding batch size
- `D = 1024` embedding dimension used during the FAISS search setup
- Weighted ensemble: GTE/BGE/MPNet as above
- `normalize_embeddings=True` for better similarity calibration
- Use of FAISS `IndexFlatL2` for nearest-neighbor retrieval

These parameters were selected through empirical leaderboard tuning and validation experiments.

## Results

This solution achieved a silver medal in the Kaggle competition, validating the effectiveness of the embedding-based retrieval plus ensemble strategy.

In the notebook, the individual model scores were reported as:

- BGE-large: LB = 0.257
- MPNet-v2: LB = 0.251
- GTE-base: LB = 0.281

The ensemble and retrieval workflow improved the final leaderboard result beyond the single-model baselines, reaching the silver-medal level.

## Repository summary

The project is centered around a Kaggle notebook:

- `ensemble-mpnetv2-bge-gte-with-faiss-gpu-0858c4.ipynb`

This notebook contains the full pipeline: dataset preprocessing, embedding generation, FAISS search, weighted ensembling, and final submission export.

## Takeaway

The core idea is simple but powerful: convert the question-answer text and misconception concepts into the same embedding space, then use vector similarity to retrieve the likely misconceptions behind a wrong answer. The ensemble of multiple sentence transformers plus FAISS retrieval was the key to achieving a strong competitive result.
