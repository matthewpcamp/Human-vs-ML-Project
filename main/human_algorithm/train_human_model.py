import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
from main.human_algorithm.human_calssifier import human_classify
from main.data.fetch_data import load_student_data
from sklearn.model_selection import train_test_split

# get the data
df, _ = load_student_data()   # grades and features in a DataFrame

# convert the numeric final grade (G3) into a category string
# this matches the categories used by the human classifier

def categorize_g3(grade):
    if grade < 8:
        return 'Low'
    elif grade < 12:
        return 'Medium'
    else:
        return 'High'

# keep original copy of data safe
df = df.copy()
df['actual_category'] = df['G3'].apply(categorize_g3)

# split into train/test for evaluation
train_df, test_df = train_test_split(
    df,
    test_size=0.3,
    random_state=42,
    stratify=df['actual_category']
)

# run the human model on the test version
# human_classify looks at G1 and G2 and returns Low/Medium/High
#                         (it ignores G3 even though we pass it)
test_df['human_prediction'] = test_df.apply(
    lambda r: human_classify(r['G1'], r['G2'], r['G3']),
    axis=1
)

test_df['correct'] = test_df['human_prediction'] == test_df['actual_category']
accuracy = test_df['correct'].mean()
print(f"Human classifier accuracy: {accuracy:.2%}")

# show confusion matrix so we can see where it makes mistakes
conf_matrix = pd.crosstab(
    test_df['actual_category'],
    test_df['human_prediction'],
    rownames=['Actual'],
    colnames=['Predicted']
)
print(conf_matrix)

# print one sample failure if there is one
if not test_df[test_df['correct']].empty:
    failure_row = test_df[~test_df['correct']].iloc[0]
    print("\nFAILURE EXAMPLE")
    print(failure_row[['G1', 'G2', 'G3', 'actual_category', 'human_prediction']])

# make a scatter plot of the results
plots_dir = os.path.join(os.path.dirname(__file__), '..', 'plots')
if not os.path.isdir(plots_dir):
    os.makedirs(plots_dir)

plt.figure(figsize=(8, 6))
sns.scatterplot(
    data=test_df,
    x='G1', y='G2',
    hue='correct', style='correct',
    palette={True: 'green', False: 'red'},
    s=100
)
plt.title('Human Algorithm: Correct vs Incorrect Predictions')
plt.xlabel('G1')
plt.ylabel('G2')
plt.legend(title='Prediction Correct')
plt.grid(True)

out_file = os.path.join(plots_dir, 'human_model_training_results.png')
plt.savefig(out_file, dpi=150)
plt.close()
