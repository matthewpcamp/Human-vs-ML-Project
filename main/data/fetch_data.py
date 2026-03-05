from ucimlrepo import fetch_ucirepo
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os

from ucimlrepo import fetch_ucirepo 


def load_student_data():
    # fetch the UCI repository dataset for student performance
    student_performance = fetch_ucirepo(id=320)

    # pull out feature names and the (first) target variable name
    feature_names = (
        student_performance.variables[
            student_performance.variables['role'] == 'Feature'
        ]['name']
        .tolist()
    )
    target_name = (
        student_performance.variables[
            student_performance.variables['role'] == 'Target'
        ]['name']
        .values[0]
    )

    # assemble feature dataframe
    df = pd.DataFrame(student_performance.data.features, columns=feature_names)

    # the dataset actually contains three separate grade columns
    # (G1, G2, G3). `target_name` from the metadata is just the first
    # entry, so `student_performance.data.targets` may be a 2D array. We
    # need to merge all of the columns into the dataframe rather than
    # attempting to assign a multi-column array to a single column.
    targets = student_performance.data.targets
    if targets.ndim == 1 or (targets.ndim == 2 and targets.shape[1] == 1):
        # single target column is simple
        df[target_name] = targets
    else:
        # multiple target columns; pull their names from the metadata
        target_names = (
            student_performance.variables[
                student_performance.variables['role'] == 'Target'
            ]['name'].tolist()
        )
        # build a small DataFrame and concatenate
        targets_df = pd.DataFrame(targets, columns=target_names)
        df = pd.concat([df, targets_df], axis=1)
        # keep the original target_name for compatibility (first entry)
        target_name = target_names[0]

    return df, target_name
