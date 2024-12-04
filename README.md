# README

## Introduction

This project implements several dimensionality reduction and data processing techniques, namely **PCA**, **RPCA**, and **RPCA on Graphs**. These methods are widely used to extract meaningful insights from high-dimensional data while reducing its complexity.

---

## Contents

- [PCA (Principal Component Analysis)](#pca)
- [RPCA (Robust Principal Component Analysis)](#rpca)
- [RPCA on Graphs](#rpca-on-graphs)

---

## PCA (Principal Component Analysis)

**PCA** is a dimensionality reduction technique that transforms a high-dimensional dataset into a set of new variables, called **principal components**, which maximize the variance of the projected data.

### Key Points:
- **Main goal**: Reduce the dimensionality of the data while retaining as much variance as possible.
- **Method**: Finds axes (principal components) in the data space where the variance is maximized.
- **Common uses**: Data compression, noise reduction, data exploration, preprocessing for machine learning algorithms.
- **Example application**: Dimensionality reduction for images, data preprocessing for classification algorithms.

### Procedure:
1. Center the data (subtract the mean).
2. Compute the covariance matrix of the centered data.
3. Perform eigenvalue and eigenvector decomposition.
4. Select the principal components that explain the most variance.
5. Project the data onto these components.

---

## RPCA (Robust Principal Component Analysis)

**RPCA** is a generalization of PCA that can handle data corrupted by noise or outliers. It decomposes the data into two parts: a low-rank component (structured) and a sparse component (outliers/noise).

### Key Points:
- **Main goal**: Handle corrupted or noisy data by separating the low-rank structure from the sparse errors.
- **Method**: Models the data as the sum of two matrices: a low-rank matrix and a sparse matrix.
- **Common uses**: Anomaly detection, missing data analysis, and image or video denoising.
- **Example application**: Detecting anomalies in financial datasets or sensor data.

### Procedure:
1. Assume the data can be decomposed into two matrices: one low-rank (structured) and one sparse (containing anomalies or noise).
2. Use factorization algorithms such as **proximal gradient methods** to estimate these two components.
3. Minimize a cost function combining the low-rank norm and sparse norm.

---

## RPCA on Graphs

**RPCA on Graphs** is an extension of RPCA that applies dimensionality reduction and component separation techniques to data structured as graphs. This allows handling data where the relationships between points are as important as the points themselves.

### Key Points:
- **Main goal**: Apply RPCA to graph-structured data (e.g., social networks, object relationships).
- **Method**: Graphs capture both local and global relationships between data points and allow for structured matrix processing.
- **Common uses**: Product recommendations, social network analysis, anomaly detection in graphs.
- **Example application**: Analyzing interactions between users on social media to detect abnormal behavior.

### Procedure:
1. Represent the data as a graph (adjacency matrices, weighted graphs, etc.).
2. Apply RPCA to decompose the data into a low-rank component (structure) and a sparse component (outliers/noise).
3. Integrate graph structure into the decomposition process to preserve relationships between data points.

---

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/username/repository.git
    ```
2. Install the requirements
   ```bash
   pip install -r requirements.txt
   ```
---

## Low Rank Recovery

 1. Load the dataset according to a time frame
   ```python
   VideoLoader.read_videos(video_name, time_interval=time_interval_you_want)
   ```

2. Apply the remove background method
   ```python
   # method_you_want is among 'rpca' and 'rpca_augmented'
   VideoLoader.remove_background(method=method_you_want, dataset_name=video_name, plot=False)
   ```

