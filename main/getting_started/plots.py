import os
from fetch_data import load_student_data
import matplotlib.pyplot as plt
import seaborn as sns


def make_plot(factor_1, factor_2):
    factor_1_label = factor_1.replace('_', ' ')
    factor_2_label = factor_2.replace('_', ' ')
    
    df, target_name = load_student_data()

    os.makedirs("plots", exist_ok=True)

    plt.figure(figsize=(8, 6))
    sns.scatterplot(
        data=df,
        x=factor_1,
        y=factor_2,
        hue=target_name,
        style=target_name,
        s=90
    )

    plt.title(f'Student Grades: {factor_1_label} vs {factor_2_label}')
    plt.xlabel(f'{factor_1_label}')
    plt.ylabel(f'{factor_2_label}')
    plt.legend(title='Student Grades')
    plt.grid(True)
    plt.savefig(f'main/plots/{factor_1_label}_v_{factor_2_label}.png', dpi=150)
    plt.close()

make_plot('G1', 'G2')
make_plot('G1', 'G3')
make_plot('G2', 'G3')