# 🧠 Mental Health Issue Classification using LSTM and GloVe Embeddings

This project leverages **Natural Language Processing (NLP)** and **Deep Learning** to classify user-submitted descriptions of mental health issues into predefined categories. Using **Bidirectional LSTM networks** and **GloVe embeddings**, the model learns semantic patterns in counseling data to assist in early-stage mental health assessment.

---

## 📖 Overview

Mental health disorders are a growing global concern, and early detection is key to prevention and treatment. This project introduces a system that classifies user input from mental health forums into 32 categories using:

- LSTM (Long Short-Term Memory) neural networks
- GloVe (Global Vectors for Word Representation) embeddings
- NLP preprocessing techniques

It aims to automate initial assessments and improve accessibility in mental health care systems.

---

## ✨ Features

- Classifies counseling conversation text into **32 mental health topics**
- Utilizes **bidirectional LSTM** for sequential modeling
- Incorporates **GloVe embeddings** for contextual understanding
- Compares performance with ensemble learning models
- Clean and modular implementation using **Keras**, **Pandas**, and **NLTK**

---

## 📊 Dataset

- **Source:** CounselChat Dataset via [Nicolas Bertagnolli’s GitHub](https://github.com/nbertagnolli/CounselChat)
- **Samples:** 1376 cleaned entries
- **Classes:** 32 mental health categories (e.g., depression, anxiety, relationships)
- **Attributes Used:** `questionText` and `topic`

---

## 🛠 Preprocessing

1. Noise removal (HTML tags, punctuation, etc.)
2. Tokenization
3. Lemmatization
4. Stop-word removal
5. Token-to-index mapping using Keras Tokenizer
6. Padding sequences to equal length

---

## 🧠 Model Architecture

The model is designed for multi-class classification of user-submitted mental health text into 32 categories. It uses a deep learning pipeline with **Bidirectional LSTM** and **GloVe embeddings** to capture semantic and sequential information.

```text
Input → Embedding Layer (GloVe, 200D) 
     → Bidirectional LSTM (100 units, dropout=0.1) 
     → Dense Layer (500 units, ReLU) 
     → Output Layer (32 units, Softmax)
