

# Comparative Study of Bayesian and Non-Parametric Multiclass Classification

**Dataset:** UCI Wine Dataset  
**Methods:** Gaussian Bayesian / Quadratic Discriminant Analysis (QDA) and k-Nearest Neighbors (k-NN)

## Executive Summary

This project develops a multiclass pattern-recognition system and compares two classification approaches: a parametric Gaussian Bayesian classifier using Quadratic Discriminant Analysis (QDA) and the non-parametric k-Nearest Neighbors (k-NN) classifier.

The UCI Wine dataset contains **178 observations, 13 numerical features, and 3 classes**. A stratified 80:20 train-test split was used, with standardisation fitted only on the training data. The models were evaluated using accuracy, macro precision, macro recall, macro F1-score, and confusion matrices.

For k-NN, values of **k = 1, 3, 5, 7, and 9** were tested using both **Euclidean and Manhattan distance**. Five-fold stratified cross-validation was also conducted to assess model robustness.

On the fixed hold-out test set, QDA achieved **100% accuracy**. Several k-NN configurations also achieved 100% accuracy. Five-fold cross-validation produced a mean accuracy of approximately **98.89% for QDA**, while the strongest k-NN configurations achieved mean accuracies of approximately **97.76%**.

The results demonstrate that both approaches can perform very well on this dataset and that model selection should consider distributional assumptions, data geometry, sample size, computational requirements, and validation performance.

## 1. Problem Formulation and Dataset

The task is a three-class supervised pattern-recognition problem. Each wine observation is represented by a 13-dimensional feature vector, and the classifier assigns each observation to one of three wine classes.

The UCI Wine dataset contains:

- **178 observations**
- **13 continuous chemical measurements**
- **3 target classes**

### Class Distribution

| Class | Count | Proportion |
|---|---:|---:|
| Class 0 | 59 | 33.1% |
| Class 1 | 71 | 39.9% |
| Class 2 | 48 | 27.0% |

## 2. Data Representation and Statistical Analysis

For each class, the sample mean and covariance matrix were calculated. The covariance matrix captures both individual feature variability and relationships between features.

### Feature Statistics

| Feature | Mean | Std. Dev. | Min | Max |
|---|---:|---:|---:|---:|
| Alcohol | 13.001 | 0.812 | 11.030 | 14.830 |
| Malic Acid | 2.336 | 1.117 | 0.740 | 5.800 |
| Ash | 2.367 | 0.274 | 1.360 | 3.230 |
| Alcalinity of Ash | 19.495 | 3.340 | 10.600 | 30.000 |
| Magnesium | 99.742 | 14.282 | 70.000 | 162.000 |
| Total Phenols | 2.295 | 0.626 | 0.980 | 3.880 |
| Flavanoids | 2.029 | 0.999 | 0.340 | 5.080 |
| Nonflavanoid Phenols | 0.362 | 0.124 | 0.130 | 0.660 |
| Proanthocyanins | 1.591 | 0.572 | 0.410 | 3.580 |
| Color Intensity | 5.058 | 2.318 | 1.280 | 13.000 |
| Hue | 0.957 | 0.229 | 0.480 | 1.710 |
| OD280/OD315 of Diluted Wines | 2.612 | 0.710 | 1.270 | 4.000 |
| Proline | 746.893 | 314.907 | 278.000 | 1680.000 |

The strongest absolute Pearson correlation was between **total phenols and flavanoids (r = 0.865)**, followed by **flavanoids and OD280/OD315 of diluted wines (r = 0.787)**. These relationships suggest that covariance structure can provide useful information for classification.

## 3. Preprocessing and Experimental Design

The dataset was divided into training and testing sets using an **80:20 stratified split** with `random_state=42` for reproducibility.

- Training observations: **142**
- Test observations: **36**

`StandardScaler` was fitted only on the training data and then applied to both training and test sets. This prevents information from the test set from influencing the learned transformation.

Scaling is particularly important for k-NN because distance-based methods combine feature-wise differences. Without scaling, high-magnitude variables such as proline could dominate lower-magnitude variables.

## 4. Bayesian / Parametric Classifier: Gaussian QDA

Quadratic Discriminant Analysis models each class using a multivariate Gaussian distribution with its own covariance matrix.

For each class, the likelihood can be expressed as:

$$
p(x|c) = \frac{1}{(2\pi)^{d/2}|\Sigma_c|^{1/2}}
\exp\left[-\frac{1}{2}(x-\mu_c)^T\Sigma_c^{-1}(x-\mu_c)\right]
$$

Bayes' rule gives:

$$
P(c|x) \propto p(x|c)P(c)
$$

Classification selects the class with the largest posterior probability.

Because QDA estimates a separate covariance matrix for each class, it generally produces **quadratic decision boundaries**.

### Main Assumptions

- Approximate multivariate normality within each class
- Independent observations
- Sufficiently reliable covariance estimation
- Feature interactions represented through covariance matrices rather than conditional independence

## 5. Non-Parametric Classifier: k-NN

k-Nearest Neighbors classifies a query observation according to the majority class among its **k nearest training observations**.

Two distance measures were evaluated.

### Euclidean Distance

$$
d_E(x,z) = \sqrt{\sum_j (x_j-z_j)^2}
$$

### Manhattan Distance

$$
d_1(x,z) = \sum_j |x_j-z_j|
$$

The experiment tested:

- k = 1
- k = 3
- k = 5
- k = 7
- k = 9

Both Euclidean and Manhattan distances were evaluated.

### Hold-Out Test Results

| Distance | k | Accuracy | Macro Precision | Macro Recall | Macro F1 |
|---|---:|---:|---:|---:|---:|
| Euclidean | 1 | 0.9722 | 0.9744 | 0.9762 | 0.9743 |
| Euclidean | 3 | 0.9722 | 0.9744 | 0.9762 | 0.9743 |
| Euclidean | 5 | 0.9722 | 0.9697 | 0.9762 | 0.9718 |
| Euclidean | 7 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| Euclidean | 9 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| Manhattan | 1 | 0.9722 | 0.9744 | 0.9762 | 0.9743 |
| Manhattan | 3 | 0.9722 | 0.9744 | 0.9762 | 0.9743 |
| Manhattan | 5 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| Manhattan | 7 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| Manhattan | 9 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |

## 6. Quantitative Comparative Evaluation

On the fixed hold-out test set:

- QDA achieved **100% accuracy**.
- Euclidean k-NN achieved **97.22%** for k = 1, 3 and 5.
- Euclidean k-NN achieved **100%** for k = 7 and 9.
- Manhattan k-NN achieved **100%** for k = 5, 7 and 9.

Because the test set contained only 36 observations, five-fold stratified cross-validation was used as an additional robustness check.

### Five-Fold Cross-Validation

| Model | Mean Accuracy | Standard Deviation |
|---|---:|---:|
| QDA | 0.9889 | 0.0136 |
| k-NN Manhattan, k=1 | 0.9776 | 0.0324 |
| k-NN Manhattan, k=9 | 0.9775 | 0.0276 |
| k-NN Manhattan, k=7 | 0.9775 | 0.0276 |
| k-NN Euclidean, k=7 | 0.9719 | 0.0252 |
| k-NN Euclidean, k=9 | 0.9719 | 0.0252 |
| k-NN Euclidean, k=5 | 0.9717 | 0.0181 |
| k-NN Manhattan, k=5 | 0.9663 | 0.0326 |

QDA produced a mean cross-validation accuracy of approximately **0.9889**, while the strongest k-NN configuration achieved approximately **0.9776**.

Both approaches nevertheless demonstrated strong performance on the dataset.

## 7. Interpretation of Decision Boundaries

QDA constructs decision boundaries by comparing class discriminant functions. Because each class has its own covariance matrix, the resulting boundaries can be curved and can reflect different orientations within each class.

k-NN does not define a single global mathematical decision boundary. Instead, its classification boundary emerges from the local geometry of neighbouring observations and can therefore be irregular.

Increasing the value of k generally produces smoother decision boundaries and reduces sensitivity to individual observations. However, excessively large values of k can oversmooth genuine local patterns.

The strong performance of both methods indicates that the wine classes are well separated in the feature space. QDA benefits from class-specific mean and covariance information, while k-NN benefits from local geometric similarity.

## 8. When Can Parametric Classification or k-NN Be Appropriate?

Parametric Gaussian classification can be useful when:

- Class distributions are reasonably close to the assumed Gaussian form.
- The available sample size supports stable parameter estimation.
- A compact statistical model can adequately describe the class structure.

k-NN can be useful when:

- Classes are strongly non-Gaussian.
- Classes are multi-modal.
- Class boundaries have irregular geometric structures.
- Local similarity provides useful information for classification.

However, k-NN is sensitive to irrelevant features, feature scaling, the choice of k, and the curse of dimensionality.

## 9. Limitations and Threats to Validity

Several limitations should be considered when interpreting the results:

- Only **178 observations** are available, so parameter and neighbourhood estimates may be sensitive to sampling variability.
- QDA covariance estimation can become unstable when the feature dimension is large relative to the sample size.
- k-NN prediction becomes increasingly computationally expensive as the training dataset grows.
- k-NN performance depends on the choice of k, distance metric, and scaling.
- The Wine dataset is relatively clean and well separated, so the observed performance may be more optimistic than performance on noisy real-world datasets.
- A single hold-out split may be unstable; therefore, five-fold stratified cross-validation was included.
- Perfect hold-out accuracy does not demonstrate universally perfect generalisation.
- A two-dimensional PCA visualisation is only an illustration and does not represent the complete 13-dimensional decision boundary.

## 10. Conclusion

This experiment demonstrates how assumptions about the underlying distribution of data can affect classification performance.

QDA explicitly models each class using a multivariate Gaussian distribution with a class-specific covariance matrix, resulting in quadratic decision boundaries. k-NN makes fewer global assumptions about the data distribution and instead relies heavily on local geometry.

On the UCI Wine dataset, QDA produced the strongest cross-validation result, while appropriately tuned k-NN also performed very well and achieved perfect hold-out accuracy for several configurations.

The findings support a data-dependent approach to model selection. The choice between parametric and non-parametric classification should consider:

- Plausibility of distributional assumptions
- Geometry of the data
- Sample size
- Computational requirements
- Validation performance

## 11. Project Deliverables

The project includes:

- Technical report covering problem formulation, dataset description, exploratory statistics, preprocessing, mathematical models, experimental design, results, interpretation, and limitations.
- Executable Jupyter Notebook containing the Python implementation.
- Evaluation using accuracy, precision, recall, F1-score, and confusion matrices.
- Experiments using five k values: 1, 3, 5, 7, and 9.
- Comparison of Euclidean and Manhattan distance measures.
- Five-fold stratified cross-validation.
- Visualisations of class distributions, feature correlations, k-NN performance, and confusion matrices.
