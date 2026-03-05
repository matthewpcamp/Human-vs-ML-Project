from ucimlrepo import fetch_ucirepo
import pandas as pd


def load_student_data():
    raw = fetch_ucirepo(id=320)

    # figure out which columns are features vs targets using the metadata
    feature_names = (
        raw.variables[raw.variables['role'] == 'Feature']['name'].tolist()
    )
    target_name = (
        raw.variables[raw.variables['role'] == 'Target']['name'].values[0]
    )

    # create a DataFrame for just the features
    df = pd.DataFrame(raw.data.features, columns=feature_names)

    targets = raw.data.targets
    if targets.ndim == 1 or (targets.ndim == 2 and targets.shape[1] == 1):
        df[target_name] = targets
    else:
        target_names = (
            raw.variables[raw.variables['role'] == 'Target']['name'].tolist()
        )
        targets_df = pd.DataFrame(targets, columns=target_names)
        df = pd.concat([df, targets_df], axis=1)
        target_name = target_names[0]

    return df, target_name
