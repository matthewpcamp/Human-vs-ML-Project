from ucimlrepo import fetch_ucirepo
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os

from ucimlrepo import fetch_ucirepo 

student_performance = fetch_ucirepo(id=320) 
  
df = student_performance.variables

# data (as pandas dataframes) 
X = student_performance.data.features 
y = student_performance.data.targets 


df['good student'] = "yes"

# metadata 
print(student_performance.metadata) 
  
# variable information 
print(student_performance.variables) 

feature_names = student_performance.variables[student_performance.variables['role'] == 'Feature']['name'].tolist()


def plot_G2_vs_G3(features=None, targets=None, save_path=None, show=False):
	"""Plot `G2` vs `G3` using the provided features/targets.

	If `targets` contains `G3` as a column or series it will be combined
	with `features` for plotting.
	"""
	# Prepare dataframe
	if features is None:
		df_feat = X.copy()
	else:
		df_feat = features.copy()

	if targets is None:
		df_tgt = y
	else:
		df_tgt = targets

	# Ensure targets is a DataFrame
	if df_tgt is None:
		df_full = df_feat.copy()
	else:
		df_tgt_df = pd.DataFrame(df_tgt).reset_index(drop=True)
		df_full = pd.concat([df_feat.reset_index(drop=True), df_tgt_df], axis=1)

	if 'G2' not in df_full.columns or 'G3' not in df_full.columns:
		raise KeyError("Required columns 'G2' and/or 'G3' not found in the combined dataframe")

	plt.figure(figsize=(8, 6))
	sns.scatterplot(data=df_full, x='G2', y='G3')
	plt.xlabel('Grade 2 (G2)')
	plt.ylabel('Final grade (G3)')
	plt.title('Grade 2 vs Final grade (G3)')
	plt.tight_layout()

	if save_path:
		os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)
		plt.savefig(save_path)
	if show:
		plt.show()
	plt.clf()


if __name__ == '__main__':
	out_file = os.path.join('plots', 'G2_vs_G3.png')
	try:
		plot_G2_vs_G3(save_path=out_file)
		print(f"Saved plot to {out_file}")
	except Exception as e:
		print('Failed to create plot:', e)