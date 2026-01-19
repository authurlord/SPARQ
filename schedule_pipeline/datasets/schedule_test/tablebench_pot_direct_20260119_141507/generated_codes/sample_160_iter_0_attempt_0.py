import pandas as pd

df = pd.read_csv('table.csv')

# Select only numerical columns for correlation
numeric_cols = ['Shared Titles', 'Runners-Up', 'Total Finals']
outright_titles = df['Outright Titles']

# Compute correlation with 'Outright Titles'
correlations = df[numeric_cols].corrwith(outright_titles)

# Find the column with the highest absolute correlation
max_corr = correlations.abs().idxmax()
max_corr_value = correlations.abs().max()

# If max correlation is below a threshold (e.g., 0.3), no clear impact
if max_corr_value < 0.3:
    print("Final Answer: no clear impact")
else:
    print(f"Final Answer: {max_corr}")