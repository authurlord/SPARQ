import pandas as pd
import numpy as np

df = pd.read_csv('table.csv')

# Select only the relevant columns
columns_to_analyze = ['Location', 'Shared Titles', 'Runners-Up', 'Total Finals', 'Last Title']
target_column = 'Outright Titles'

# Convert to numeric, handling errors (e.g., strings like 'None' or non-numeric)
df_numeric = df.copy()
df_numeric['Shared Titles'] = pd.to_numeric(df_numeric['Shared Titles'], errors='coerce')
df_numeric['Runners-Up'] = pd.to_numeric(df_numeric['Runners-Up'], errors='coerce')
df_numeric['Total Finals'] = pd.to_numeric(df_numeric['Total Finals'], errors='coerce')
df_numeric[target_column] = pd.to_numeric(df_numeric[target_column], errors='coerce')

# Drop rows with NaN in target or factors
df_clean = df_numeric.dropna(subset=[target_column] + [col for col in columns_to_analyze if col != 'Location'])

# Since 'Location' is categorical, we cannot compute direct correlation. We will skip it.
# Compute correlation with other numeric factors
correlations = {}
for col in ['Shared Titles', 'Runners-Up', 'Total Finals']:
    if col in df_clean.columns:
        corr = df_clean[target_column].corr(df_clean[col])
        correlations[col] = corr

# Find the factor with the highest absolute correlation
if correlations:
    max_corr_factor = max(correlations, key=lambda x: abs(correlations[x]))
    max_corr_value = correlations[max_corr_factor]
    if abs(max_corr_value) > 0.1:  # Threshold for meaningful impact
        final_answer = max_corr_factor
    else:
        final_answer = 'no clear impact'
else:
    final_answer = 'no clear impact'

print(f"Final Answer: {final_answer}")