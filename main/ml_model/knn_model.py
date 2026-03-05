import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

# bring in the helper functions from other parts of the project
from main.data.fetch_data import load_student_data
from main.human_algorithm.human_calssifier import human_classify

# Load the data
student_df, _ = load_student_data()          # dataframe with G1,G2,G3 etc.
print("shape", student_df.shape)
print(student_df.head(2))                    # show first couple rows

# Make the grade category target from G3
def make_category(grade):
    if grade < 8:
        return 'Low'
    elif grade < 12:
        return 'Medium'
    else:
        return 'High'

student_df['actual_category'] = student_df['G3'].apply(make_category)
print("Category counts:")
print(student_df['actual_category'].value_counts())

# Prepare features and train/test split
# use the first two grades as predictors
feature_cols = ['G1', 'G2']
X = student_df[feature_cols]
y = student_df['actual_category']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

# Try several values of k and choose the best one
print("Checking odd k values from 1 to 15")
best_k = None
best_acc = 0.0
for k in range(1, 16, 2):               
    clf = KNeighborsClassifier(n_neighbors=k)
    clf.fit(X_train, y_train)
    acc = clf.score(X_test, y_test)
    print(f" k={k} accuracy={acc:.3f}")
    if acc > best_acc:
        best_acc = acc
        best_k = k

print(f"selected k={best_k} with accuracy {best_acc:.3f}")

# Train the final model and make predictions on the test set 
knn = KNeighborsClassifier(n_neighbors=best_k)
knn.fit(X_train, y_train)
predictions = knn.predict(X_test)

# combine test features with predictions so we can evaluate
result_df = X_test.copy()
result_df['actual_category'] = y_test
result_df['KNN_prediction'] = predictions
result_df['correct'] = result_df['KNN_prediction'] == result_df['actual_category']

print("\nKNN accuracy on test set:", result_df['correct'].mean())
print(pd.crosstab(result_df['actual_category'], result_df['KNN_prediction']))


# Compare against the simple human classifier
human_preds = result_df.apply(
    lambda r: human_classify(r['G1'], r['G2'], r['actual_category']),
    axis=1
)
hum_acc = (human_preds == result_df['actual_category']).mean()
print("Human algorithm accuracy:", hum_acc)
print(pd.crosstab(result_df['actual_category'], human_preds))

# Plot training and test results (correct vs incorrect predictions)
plots_dir = os.path.join(os.path.dirname(__file__), '..', 'plots')
if not os.path.isdir(plots_dir):
    os.makedirs(plots_dir)

# training set scatter
train_plot_df = X_train.copy()
train_plot_df['actual_category'] = y_train
train_plot_df['KNN_prediction'] = knn.predict(X_train)
train_plot_df['correct'] = train_plot_df['KNN_prediction'] == train_plot_df['actual_category']

plt.figure(figsize=(8, 6))
sns.scatterplot(
    data=train_plot_df,
    x='G1', y='G2',
    hue='correct', style='correct',
    palette={True: 'green', False: 'red'},
    s=100
)
plt.title(f'KNN (k={best_k}) training set results')
plt.xlabel('G1')
plt.ylabel('G2')
plt.legend(title='Prediction Correct')
plt.grid(True)
plt.savefig(os.path.join(plots_dir, 'knn_train_results.png'), dpi=150)
plt.close()

# test set scatter
plt.figure(figsize=(8, 6))
sns.scatterplot(
    data=result_df,
    x='G1', y='G2',
    hue='correct', style='correct',
    palette={True: 'green', False: 'red'},
    s=100
)
plt.title(f'KNN (k={best_k}) test set results')
plt.xlabel('G1')
plt.ylabel('G2')
plt.legend(title='Prediction Correct')
plt.grid(True)
plt.savefig(os.path.join(plots_dir, 'knn_test_results.png'), dpi=150)
plt.close()
