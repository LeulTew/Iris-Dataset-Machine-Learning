# Iris-Dataset-Machine-Learning
Iris Dataset Machine Learning: Decision Trees, KNN, Perceptron, and Clustering with Visualizations
# README: Machine Learning Models on the Iris Dataset

## Project Overview

This notebook contains the implementation of six machine learning problems involving Decision Trees, K-Nearest Neighbors (KNN), Perceptron, K-Means Clustering, and K-Medoids Clustering using the Iris dataset. Each problem was solved step-by-step with clear instructions, and performance was evaluated using various metrics. The notebook also includes plots and visualizations to analyze the results.

## Requirements

- **Python 3.7+**
- **VS Code** (or any Jupyter-supported environment)
- Required Python libraries:
  - `numpy`
  - `pandas`
  - `matplotlib`
  - `seaborn`
  - `scikit-learn`
  - `sklearn_extra` (for K-Medoids)

You can install the required dependencies using:
```bash
pip install numpy pandas matplotlib seaborn scikit-learn scikit-extra
```

## Problems Overview

### 1. Decision Tree Depth Optimization
- **Objective:** Understand the impact of tree depth on model performance and how it affects overfitting/underfitting.
- **Key Steps:**
  1. Trained decision trees with various max depths.
  2. Evaluated models using accuracy, precision, recall, F1-score.
  3. Plotted performance metrics and visualized decision boundaries of the best model.

### 2. K-Nearest Neighbors (KNN) Hyperparameter Tuning
- **Objective:** Optimize the KNN model by adjusting the number of neighbors and distance metrics.
- **Key Steps:**
  1. Trained KNN classifiers with different k-values and distance metrics.
  2. Evaluated each model.
  3. Visualized decision boundaries and analyzed performance metrics.

### 3. Perceptron Learning Algorithm
- **Objective:** Understand the Perceptron algorithm and its limitations.
- **Key Steps:**
  1. Trained a Perceptron model.
  2. Evaluated its performance.
  3. Plotted decision boundaries and convergence over iterations.
---

## Problem #4: Comparing Decision Tree, KNN, and Perceptron

This section focuses on comparing three different classification algorithms: Decision Tree, K-Nearest Neighbors (KNN), and Perceptron, using the Iris dataset.

### Steps:
1. **Data Preparation**:
   - Loaded the Iris dataset.
   - Converted species names to numerical labels.
   - Selected the first 100 samples for binary classification (Setosa vs. Versicolor).
   - Used two features (`SepalLengthCm`, `SepalWidthCm`) for visualization.
   - Split the dataset into training and testing sets.

2. **Feature Standardization**:
   - Applied standardization to features for the Perceptron and KNN models.

3. **Model Implementation**:
   - Implemented Decision Tree, KNN, and Perceptron from scratch.

4. **Training and Evaluation**:
   - Trained each model using the training set.
   - Evaluated the models using accuracy, precision, recall, and F1 score.

5. **Results**:
   - **Decision Tree**: Accuracy: 1.00, Precision: 1.00, Recall: 1.00, F1 Score: 1.00
   - **KNN**: Accuracy: 1.00, Precision: 1.00, Recall: 1.00, F1 Score: 1.00
   - **Perceptron**: Accuracy: 0.60, Precision: 0.50, Recall: 1.00, F1 Score: 0.67

6. **Visualization**:
   - Plotted decision boundaries for each model.
   - Compared the performance metrics of the classifiers.

### Visualization:
- **Decision Boundaries**: Displayed the decision boundaries for each model.
- **Performance Metrics**: Visualized the accuracy, precision, recall, and F1 scores for each classifier.

---

## Problem #5: K-Means Clustering and Visualization

This problem focuses on applying K-Means clustering to the Iris dataset and visualizing the results.

### Steps:
1. **Data Preparation**:
   - Loaded the Iris dataset.
   - Applied feature normalization using `StandardScaler`.

2. **K-Means Clustering**:
   - Applied K-Means clustering with 3 clusters.
   - Visualized the clustering results in 2D and 3D.

3. **Cluster Comparison**:
   - Compared the cluster assignments with the actual class labels.

4. **Silhouette Score**:
   - Calculated the silhouette score to evaluate clustering quality.
   - **Silhouette Score**: 0.46

5. **Silhouette Plot**:
   - Generated a silhouette plot to visualize the silhouette coefficient values for each cluster.

---

## Problem #6: K-Medoids Clustering and Comparison with K-Means

This problem extends the clustering analysis by comparing K-Means with K-Medoids clustering.

### Steps:
1. **Data Preparation**:
   - Loaded the Iris dataset.
   - Normalized the dataset using `StandardScaler`.

2. **K-Medoids Clustering**:
   - Implemented K-Medoids clustering using the PAM (Partitioning Around Medoids) algorithm.
   - Visualized the clustering results in 2D and 3D.

3. **Cluster Comparison**:
   - Compared K-Medoids cluster assignments with K-Means and the actual class labels.

4. **Silhouette Score**:
   - Calculated silhouette scores for both K-Means and K-Medoids.
   - **K-Means Silhouette Score**: 0.46
   - **K-Medoids Silhouette Score**: 0.45

5. **Silhouette Plot**:
   - Generated silhouette plots for both clustering methods.

---


## Running the Notebook

1. Open the notebook in **VS Code** or another Jupyter notebook environment.
2. Ensure the required libraries are installed.
3. Run each cell sequentially to execute the code, view outputs, and generate plots.
4. Visualizations and performance metrics will be displayed within the notebook.

## Outputs and Explanations

- **Performance Metrics:** For each model, the notebook provides accuracy, precision, recall, F1-score, and confusion matrices where applicable.
- **Decision Boundaries:** Plots to help visualize how each classifier separates the Iris dataset.
- **Clustering Visualizations:** 2D and 3D plots for both K-Means and K-Medoids clustering, along with silhouette score comparisons.

## Screenshots and Presentation

This notebook complements a **PowerPoint presentation**, which includes:
- Screenshots of key outputs and visualizations.
- Performance comparisons and insights for each model.
- Summary of the key takeaways from the experiments.

Note that the presentation is used for illustrating the results, while this README provides a clear explanation of the code and its usage.

## Submission Package

- **Python Notebook:** `AI_Assign.ipynb` (contains all implementation code and visualizations).
- **PowerPoint Presentation:** A complementary visual summary of the project.

