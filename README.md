# Quantum-Enhanced Multimodal Temporal Plant Disease Progression Prediction

A multimodal temporal deep-learning project for predicting plant disease progression from sequential wheat leaf observations using classical recurrent neural networks and a Quantum Long Short-Term Memory (QLSTM) model.

---

## 1. Project Overview

Plant diseases develop over time rather than appearing as independent events in individual images. Therefore, predicting disease progression requires understanding:

- Visual characteristics of plant leaves
- Disease severity
- Lesion development
- Environmental/metadata information
- Temporal relationships between observations

This project investigates whether a Quantum-enhanced recurrent architecture, specifically a Quantum Long Short-Term Memory (QLSTM), can improve temporal disease progression prediction compared with classical recurrent models such as LSTM and GRU.

The project uses sequential wheat leaf observations derived from the ETH Zurich Sequential Wheat Dataset.

The central research question is:

> Can a hybrid QLSTM improve temporal plant disease progression prediction over classical recurrent models when using sequential wheat leaf images and environmental metadata?

---

## 2. Main Objective

The main objective is to develop a multimodal temporal prediction pipeline:

RAW / PROCESSED WHEAT DATA
        |
        v
Image / Feature Preprocessing
        |
        v
Vision / Metadata Feature Extraction
        |
        v
Temporal Sequence Construction
        |
        v
Multimodal Feature Fusion
        |
        v
+-----------------------------+
| Temporal Recurrent Models   |
|                             |
|  Classical LSTM             |
|  Classical GRU              |
|  Quantum LSTM (QLSTM)       |
+-----------------------------+
        |
        v
Multi-Task Prediction
        |
        +--------------------+
        |                    |
        v                    v
Disease Severity       Lesion Area
        |
        v
Evaluation + Visualization