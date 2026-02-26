from ucimlrepo import fetch_ucirepo
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os

from ucimlrepo import fetch_ucirepo 

student_performance = fetch_ucirepo(id=320) 
  
df = student_performance.variables

# data (as pandas dataframes) 
x = student_performance.data.features 
y = student_performance.data.targets 


df['good student'] = "yes"

# metadata 
print(student_performance.metadata) 
  
# variable information 
print(student_performance.variables) 

feature_names = student_performance.variables[student_performance.variables['role'] == 'Feature']['name'].tolist()