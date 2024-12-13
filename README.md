# Project 2: BERT Question-Answering System

This project involves fine-tuning the BERT model to serve as a Question-Answering (QA) system. 

## Dataset Description

This project uses a simplified version of the dataset from the paper **"A BERT Baseline for the Natural Questions"** with the following features:

1. **Answer Types**: Includes only two types—**No Answer** and **Short Answer**.
2. **Direct Contexts**: Each example provides a single context for extracting the answer span (if applicable).

### Key Points
- The model works directly on provided contexts during inference, bypassing span-ranking complexities from the original paper.
- Answer spans are provided using **character indices** relative to the context.


## Model Overview: DistilBERTQA

The **DistilBERTQA** model fine-tunes the lightweight and efficient **DistilBERT** architecture ('distilbert-base-uncased') for span-based Question-Answering (QA) tasks. It predicts:

1. **Answer Span**: Identifies the start and end tokens of the answer in the context.
2. **Answer Type**: Classifies the type of answer (e.g., no answer or short answer).

## Training and Evaluation

### Training Loop
The `train_loop` function trains the QA model over 2 epochs using the AdamW optimizer. It computes training losses and evaluates the model on a validation dataset after each epoch.

- **Loss Function**: Combines cross-entropy losses for start span, end span, and answer type classification.
- **Output**: Lists of training and validation losses for each epoch.

### Evaluation Loop
The `eval_loop` function evaluates the trained model on the validation dataset and computes span-level metrics:

Metrics: Precision, recall, and F1 score for span predictions.
Output: Precision, recall, and F1 score.

## Main Function

The `main` function handles the entire process of training and evaluating the QA model. Here’s what it does:

1. Sets up file paths and checks if a GPU is available.
2. Loads the model, tokenizer, and data.
3. Prepares data loaders for training and validation.
4. Trains the model and evaluates it, reporting precision, recall, and F1 score.

## Results

### Training and Validation
- **Epoch 1**:
  - **Train Loss**: 3.7811
  - **Validation Loss**: 2.7316
  
- **Epoch 2**:
  - **Train Loss**: 2.2436
  - **Validation Loss**: 2.4962
  
### Final Evaluation
- **Precision**: 0.7111
- **Recall**: 0.7312
- **F1-Score**: 0.6965

Here's a video that gives an overview of the project and dives into the challenges I tackled along the way. [[link]](https://drive.google.com/file/d/1yAa656HerqUJ5O9EV2_AJWz6EEarKcyz/view?usp=drive_link)