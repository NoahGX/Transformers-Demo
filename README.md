# Transformers-Demo

## Overview
  - **Part 1**: Demonstrates a manual implementation of the multi-head attention sublayer in a Transformer encoder. It shows how queries, keys, and values are computed and how attention scores are derived and applied to the value vectors.
  - **Part 2**: Uses a pre-trained BERT model (from Hugging Face Transformers) to tokenize text, derive embeddings for specific tokens (e.g., "entropy"), compare sentence embeddings, and generate summaries using a summarization pipeline.

## Features
  - **Multi-Head Attention**: Manual computation of Q, K, V matrices, attention scores, and the final weighted outputs.
  - **Token Embeddings**: Utilize a BERT model to extract embeddings for specific words.
  - **Sentence Embeddings**: Tokenize multiple sentences simultaneously and compare [CLS] (or sentence-level) embeddings using cosine similarity.
  - **Summarization**: Leverage the Hugging Face `pipeline` for text summarization.

## Usage 
  1. Clone or download the repository to your local machine.
  2. Open the notebook using Jupyter Lab or Jupyter Notebook.
  3. Run the cells in order to see the step-by-step implementation and outputs.  

## Prerequisites
  - **Python 3.7+**
  - **NumPy**
  - **SciPy**
  - **Jupyter Notebook vs JupyterLab**
  - [**PyTorch**](https://pytorch.org/) (version compatible with your CUDA setup if using GPUs)
  - [**Hugging Face Transformers**](https://github.com/huggingface/transformers) (for additional transformer-based NLP tasks)
  - Install required packages via pip:
      ```bash
        pip install torch numpy scipy transformers
      ```

## Input
  - The notebook uses randomly generated data (`x`) to demonstrate multi-head attention calculations.
  - For exploring **transformers** library tasks, you can provide your own texts or datasets for inference.  

## Output
  - Intermediate print statements show the shape and sample contents of Q, K, V matrices, attention scores, and weighted outputs.
  - Log outputs from Hugging Face Transformers (if you run additional cells focusing on NLP tasks).

## Notes
  - The random seeds are set for reproducibility.
  - This notebook focuses on a conceptual understanding of multi-head attention in Transformers.
  - In practice, you would replicate the attention mechanism across multiple heads, then concatenate or project the combined output.
  - For large-scale tasks, consider using GPU-accelerated hardware to speed up computations.
