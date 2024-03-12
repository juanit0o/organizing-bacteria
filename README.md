# Bacterial Cell Image Analysis

## Objective
The objective of this assignment is to analyze a set of bacterial cell images using machine learning techniques, including feature extraction, feature selection, and clustering, to assist biologists in organizing similar images.

## Data
- **Dataset:** 563 PNG images in the `images/` folder (ommited due their size)

The images provided for this assignment were obtained by automatic segmentation and include cells in different stages of their life cicle as well as segmentation errors, not corresponding to real cells.
<p align="center">
       <img src="https://i.imgur.com/hSkfA0w.png" width="900" height="400" alt="Layout of the website">
        <br>
       <em>Sample of the images provided</em>
</p>

## Implementation
1. Load images and extract features using three methods:
    - Principal Component Analysis (PCA)
    - t-Distributed Stochastic Neighbor Embedding (t-SNE)
    - Isometric mapping with Isomap
2. Extract six features per method, totaling 18 features.
3. Use `tp2_aux.py` module for image matrix retrieval and feature extraction.
4. Use `labels.txt` for cell cycle phase information (0 for unlabeled cells). The labels.txt has information on the identification of the cell cycle phase for each cell. The first column has the cell identifier and the second column a label for the cell cycle phase. These cells were manually labelled by biologists. There are 3 phases, labelled with integers 1, 2 and 3. The first phase before the cell starts to divide, the second covers the first part of the division, with the formation of a membrane ring where the cells will divide, and the third phase corresponds to the final stage of cell division. However, note that only some images are labelled. Images that were not labelled have a label value of 0 in this file.



## Clustering Algorithms
- **DBSCAN:** Implement parameter selection method based on the original paper by Martin Ester et al. (1996).
- **K-Means:** Vary parameters (ε for DBSCAN, k for K-Means) and evaluate performance using silhouette score and external indices (Rand index, Precision, Recall, F1 measure, and adjusted Rand index).

## Optional Exercise
- Implement bissecting K-Means hierarchical clustering algorithm.

## Guidelines for Implementation
- Ensure proper handling of features extraction time.
- Save extracted features for efficient experimentation.
- Utilize `tp2_aux.py` functions for cluster report generation.
- Provide clear and concise answers to associated questions.

## Questions
- Answer questions based on the assignment instructions in TP2.txt.

## Contributors
- João Funenga & André Costa
