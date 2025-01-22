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
  - **Part 1 (Manual Multi-Head Attention)**:
      1. Open the notebook in Jupyter.
      2. Run all cells to see the step-by-step implementation of multi-head attention.
      3. Observe printed shapes, attention scores, and weighted outputs.  
  - **Part 2 (BERT Embeddings & Summarization)**:
      1. Open the notebook in Jupyter).
      2. Make sure the Hugging Face `transformers` library is installed (`pip install transformers`).
      3. Run all cells to tokenize text, extract embeddings, compare similarities, and summarize.
      4. Observe the final text summaries and similarity scores printed in the output cells.

## Prerequisites
  - Python 3.7+
  - Jupyter Notebook or JupyterLab environment  
  - **Part 1**:
      - `numpy`
      - `torch`
      - `torch.nn`
      - `scipy` (for `softmax`)
  - **Part 2**:
      - `torch`
      - `transformers`
      - `tokenizers` (installed alongside `transformers`)

## Input
  - **Part 1**: Synthetic input vectors (`x`) for demonstrating multi-head attention calculations.  
  - **Part 2**:
      - A sample text paragraph for BERT tokenization and summarization.
      - Splitting text into sentences for separate encoding.  

## Output
  - **Part 1**:
      - Intermediate Q, K, V matrices.
      - Attention score matrices and the final weighted output vectors.
  - **Part 2**:
      - Prints the shape of tokenized input, hidden states from BERT, and compares cosine similarity among [CLS] embeddings of sentences.
      - Summarized version of the input text using the Hugging Face pipeline.  

## Notes
  - Make sure you have enough memory and a proper GPU (optional) for faster execution of the BERT model.
  - The code in Part 1 is a conceptual demonstration. In production-level applications, use efficient PyTorch or TensorFlow implementations of multi-head attention.
  - Part 2 showcases only a small subset of BERT’s capabilities. Fine-tuning on specific tasks usually yields better performance than zero-shot usage.
  - References: [PyTorch documentation](https://pytorch.org/) and [Hugging Face Transformers documentation](https://github.com/huggingface/transformers)
