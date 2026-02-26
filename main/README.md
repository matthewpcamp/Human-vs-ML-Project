# Human vs Machine Learning Project

This project challenges you to explore the differences between human-designed algorithms and machine learning models. You will first create a human algorithm (pseudo-code) to classify data based on features, then translate that algorithm into Python. Next, you will train a K-Nearest Neighbors (KNN) classifier on the same dataset and compare your results. Finally, you will record a short screen-share with narration explaining your methods and observations.

You may work alone or with a partner. You may choose to work with the provided Penguins dataset, or select your own pre-cleaned dataset from the links below (I have suggested a few datasets as a guide, but you are welcome to select something different with approval).  The most important detail regarding your data-set is that your data needs to lend itself to classification.  For example, an iris with a sepal length of x and a petal width of y can be classified as ‘Setosa’. I also recommend that you use github codespaces, as you will need access to command-line tools that are unavailable in VS Code for EDU.

[UCI Machine Learning Repository](https://archive.ics.uci.edu/datasets)
 - Iris (classic 3-class classification)
 - Mushroom (binary classification: edible/poisonous)
 - Student Performance (predict grades, numeric features)

[Kaggle Datasets](https://www.kaggle.com/datasets)   *Note: For Kaggle, I will have to download the data for you and post on a shared drive.
 - Titanic survival dataset (binary classification)
 - Heart disease dataset (binary classification)
 - Breast cancer diagnosis (binary)
 - Penguins dataset (same as Kira, already cleaned)

---

**Team Members:**  
- Name 1  
- Name 2 (if applicable)  

**Dataset Used:**  
(e.g. Penguins, Iris, Titanic, etc.)

**Source:**  
(UCI / Kaggle / Provided dataset)

**Target Variable (What we are predicting):**  
(e.g. species, survival, diagnosis)

**Features Used:**  
- Feature 1  
- Feature 2  
- Feature 3

**[Video Review](https://)**

## Human Algorithm

### Pseudo-Code
```text
Write your human decision rules here.
```

When examining the data and visualizations, we focused on the features ___ and ___ because ___.

The plots/tables suggested a possible threshold for ___, and we considered values above or below this point to see how they might relate to ___.

From the summary tables and visualizations, it appeared that ___ could influence classification, which led us to ___ in our decision rules.

### Confusion Matrix

Accuracy: ?

| Actual \ Predicted | Class 1 | Class 2 | Class 3 |
|-------------------|---------|---------|---------|
| **Class 1**       |         |         |         |
| **Class 2**       |         |         |         |
| **Class 3**       |         |         |         |

One example where our algorithm worked well is when the inputs were ___, leading to a correct prediction of ___ because ___.

An example where the algorithm did not perform as expected is when the inputs were ___, resulting in a prediction of ___ instead of ___, which may have happened because ___.

These examples of success and failure highlight patterns in the data or limitations in our rules, such as ___.

<img width="315" height="334" alt="image" src="https://github.com/user-attachments/assets/23ee1e49-da76-47c2-97b8-c8fbcbef179c" />

## Machine Learning Model

We chose a value of k = ___ after comparing model performance across different values of k and observing that ___.

When analyzing the outputs and metrics, we noticed that changing k affected ___, which influenced our final choice.

Based on the results shown in the tables or visualizations, k = ___ best matched our goals for model performance because ___.

### Confusion Matrix

Accuracy: ?

| Actual \ Predicted | Class 1 | Class 2 | Class 3 |
|-------------------|---------|---------|---------|
| **Class 1**       |         |         |         |
| **Class 2**       |         |         |         |
| **Class 3**       |         |         |         |

The table/visualization shows a clear pattern where the model predicts ___ when ___, indicating a strong relationship between these features.

The confusion matrix reveals that the model most often confuses ___ with ___, suggesting these classes have similar feature values.

Compared to the human algorithm, the KNN model shows different behavior when ___, as seen in the ___ visualization.

<img width="315" height="334" alt="image" src="https://github.com/user-attachments/assets/199ae59d-3470-40c6-9669-60e62b211619" />
