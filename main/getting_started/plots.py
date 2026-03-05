import os
from main.data.fetch_data import load_student_data
import matplotlib.pyplot as plt
import seaborn as sns


def make_plot(factor_1, factor_2):
    factor_1_label = factor_1.replace('_', ' ')
    factor_2_label = factor_2.replace('_', ' ')
    
    df, target_name = load_student_data()

    # create a predicted grade category based on the provided pseudo-code;
    # this will override the original target coloring
    def _predict_category(row):
        if row[factor_1] < 8 and row[factor_2] < 8:
            return 'Low'
        elif row[factor_1] < 12 and row[factor_2] < 12:
            return 'Medium'
        else:
            return 'High'

    # apply classification and add to dataframe
    df = df.copy()
    df['predicted_grade'] = df.apply(_predict_category, axis=1)

    # determine output directory inside the package so the
    # plots/ folder always lives under `main` regardless of current
    # working directory
    plots_dir = os.path.join(os.path.dirname(__file__), '..', 'plots')
    plots_dir = os.path.normpath(plots_dir)
    os.makedirs(plots_dir, exist_ok=True)

    plt.figure(figsize=(8, 6))
    sns.scatterplot(
        data=df,
        x=factor_1,
        y=factor_2,
        hue='predicted_grade',
        palette={'Low': 'red', 'Medium': 'blue', 'High': 'green'},
        s=90
    )

    plt.title(f'Student Grades: {factor_1_label} vs {factor_2_label}')
    plt.xlabel(f'{factor_1_label}')
    plt.ylabel(f'{factor_2_label}')
    plt.legend(title='Predicted Grade')
    plt.grid(True)
    out_path = os.path.join(plots_dir, f"{factor_1_label}_v_{factor_2_label}.png")
    plt.savefig(out_path, dpi=150)
    plt.close()

make_plot('G1', 'G2')
make_plot('G1', 'G3')
make_plot('G2', 'G3')