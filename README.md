# Generative AI & Deep Learning Assessment Portfolio 🚀

Welcome to the **Generative AI & Deep Learning Assessment Portfolio**. This repository contains 6 production-grade, modular modules covering core concepts in Deep Learning (MLPs, CNNs, Computer Vision), Real-Time Systems, Retrieval-Augmented Generation (RAG), Autonomous ReAct AI Agents with Function Calling, and LLM Fine-Tuning & Quantitative Evaluation.

---

## 📂 Repository Structure

| File Name | Description | Key Technologies |
| :--- | :--- | :--- |
| **`01_mlp_mnist_classification.py`** | Multi-Layer Perceptron (MLP) for Handwritten Digit Classification | TensorFlow / Keras, Regularization, Callbacks |
| **`02_cnn_image_classification.py`** | Convolutional Neural Network (CNN) Classifier with Full Evaluation Suite | Keras Conv2D, Scikit-Learn, Confusion Matrix |
| **`03_realtime_emotion_recognition.py`** | Real-Time Facial Emotion Recognition & Automated Feedback System | OpenCV Haar Cascades, Keras CNN, Live Overlay |
| **`04_rag_vector_search_qa.py`** | Retrieval-Augmented Generation (RAG) Document QA Pipeline | TF-IDF / Embedding Vector Store, Text Chunker, LLM Synthesis |
| **`05_genai_agent_function_calling.py`** | Autonomous ReAct Generative AI Agent with Tool & Function Calling | ReAct Pattern, Function Calling, System Prompting |
| **`06_llm_finetuning_evaluation.py`** | LLM Fine-Tuning & Quantitative Evaluation Benchmark | ROUGE-1/2/L, BLEU-4, Model Perplexity (PPL) |

---

## ⚙️ Installation & Prerequisites

Ensure Python 3.9+ is installed on your system. Install the required dependencies using pip:

```bash
pip install tensorflow opencv-python scikit-learn numpy matplotlib
```

---

## 🚀 Execution Guide

### Module 1: MLP MNIST Classification
Train and evaluate the MLP model on MNIST digits:
```bash
python 01_mlp_mnist_classification.py
```

### Module 2: CNN Image Classification & Metrics
Build a 2D CNN, export performance plots, and confusion matrix:
```bash
python 02_cnn_image_classification.py
```

### Module 3: Real-Time Emotion Recognition with Feedback
Launch live facial emotion analysis via webcam (or headless test mode):
```bash
python 03_realtime_emotion_recognition.py
```

### Module 4: Retrieval-Augmented Generation (RAG) System
Perform document ingestion, vector indexing, similarity search, and prompt synthesis:
```bash
python 04_rag_vector_search_qa.py
```

### Module 5: GenAI Agent & Tool Calling
Execute multi-step reasoning using ReAct agent logic with tool execution:
```bash
python 05_genai_agent_function_calling.py
```

### Module 6: LLM Fine-Tuning & Metrics Evaluation
Compute ROUGE, BLEU, and Perplexity metrics for language models:
```bash
python 06_llm_finetuning_evaluation.py
```

---

## 🛠️ Summary of Modules

1. **MLP Classification**: Demonstrates dense neural network design, dropout regularization, and learning rate scheduling.
2. **CNN Architecture**: Utilizes 2D spatial feature extraction with Batch Normalization and comprehensive classification metrics.
3. **Facial Emotion Recognition**: Integrates real-time OpenCV Haar Cascade face tracking with CNN inference and dynamic feedback text overlay.
4. **RAG Pipeline**: Implements semantic document chunking, vector similarity search, context retrieval, and structured QA prompt construction.
5. **ReAct Agent**: Features an autonomous agent that parses structured JSON action intents, executes external Python functions, and iterates until goal completion.
6. **LLM Evaluation**: Evaluates text generation accuracy using ROUGE (1, 2, L), BLEU-4, and Perplexity metrics.

---

## 📄 License
This project is open-source and available under the MIT License.
