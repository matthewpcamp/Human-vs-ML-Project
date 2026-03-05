import os
from main.data.fetch_data import load_student_data
import matplotlib.pyplot as plt
import seaborn as sns

# -----------------------------------------------------------------------------
# simple plotting helper for the "getting started" notebook/assignment
# -----------------------------------------------------------------------------
# the idea is: take two numeric columns (e.g. G1 and G2) and colour each
# point according to a toy rule that tries to predict the final grade.
# the colors help students see the relationship visually.
# -----------------------------------------------------------------------------

def make_plot(factor_1, factor_2):
    # human‑readable labels for titles/filenames
    f1_label = factor_1.replace('_', ' ')
    f2_label = factor_2.replace('_', ' ')

    # load data every time so this module can be run standalone
    df, _ = load_student_data()

    # handwritten rule copied from pseudo-code
    def predict_grade(row):
        if row[factor_1] < 8 and row[factor_2] < 8:
            return 'Low'
        elif row[factor_1] < 12 and row[factor_2] < 12:
            return 'Medium'
        else:
            return 'High'

    df = df.copy()  # avoid changing original dataframe
    df['predicted_grade'] = df.apply(predict_grade, axis=1)

    # make sure output directory exists relative to this file
    plots_dir = os.path.normpath(os.path.join(os.path.dirname(__file__), '..', 'plots'))
    os.makedirs(plots_dir, exist_ok=True)

    plt.figure(figsize=(8, 6))
    sns.scatterplot(
        data=df,
        x=factor_1,
        y=factor_2,
        hue='predicted_grade',
        palette={'Low': 'red', 'Medium': 'blue', 'High': 'green'},
        s=90,
    )

    plt.title(f'Student Grades: {f1_label} vs {f2_label}')
    plt.xlabel(f1_label)
    plt.ylabel(f2_label)
    plt.legend(title='Predicted Grade')
    plt.grid(True)

    filename = f"{f1_label}_v_{f2_label}.png"
    plt.savefig(os.path.join(plots_dir, filename), dpi=150)
    plt.close()


# create all three plots
make_plot('G1', 'G2')
make_plot('G1', 'G3')
make_plot('G2', 'G3')