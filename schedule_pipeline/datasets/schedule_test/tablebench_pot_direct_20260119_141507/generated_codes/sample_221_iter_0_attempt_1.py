import pandas as pd
import numpy as np

# Load the data
df = pd.read_csv('table.csv')

# Clean the data: remove non-numeric entries (e.g., 'x', 'O') in the relevant columns
df['Composition'] = pd.to_numeric(df['Composition'], errors='coerce')
df['Drawing'] = pd.to_numeric(df['Drawing'], errors='coerce')
df['Color'] = pd.to_numeric(df['Color'], errors='coerce')
df['Expression'] = pd.to_numeric(df['Expression'], errors='coerce')

# Drop rows with any NaN values after conversion
df_clean = df.dropna()

# Compute correlation between Composition and each of the other variables
correlation_drawing = df_clean['Composition'].corr(df_clean['Drawing'])
correlation_color = df_clean['Composition'].corr(df_clean['Color'])
correlation_expression = df_clean['Composition'].corr(df_clean['Expression'])

# Find the variable with the highest absolute correlation
correlations = {
    'Drawing': correlation_drawing,
    'Color': correlation_color,
    'Expression': correlation_expression
}

max_corr_var = max(correlations, key=abs)
print(f"Final Answer: {max_corr_var}")