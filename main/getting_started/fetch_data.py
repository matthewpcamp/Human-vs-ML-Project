from ucimlrepo import fetch_ucirepo
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os

from ucimlrepo import fetch_ucirepo 


def load_student_data():
	student_performance = fetch_ucirepo(id=320) 
  

	feature_names = student_performance.variables[student_performance.variables['role'] == 'Feature']['name'].tolist()
	target_name = student_performance.variables[student_performance.variables['role'] == 'Target']['name'].values[0]

	df = pd.DataFrame(student_performance.data.features, columns=feature_names)
	df[target_name] = student_performance.data.targets

	return df, target_name
	out_file = os.path.join('plots', 'G2_vs_G3.png')
	try:
		plot_G2_vs_G3(save_path=out_file)
		print(f"Saved plot to {out_file}")
	except Exception as e:
		print('Failed to create plot:', e)