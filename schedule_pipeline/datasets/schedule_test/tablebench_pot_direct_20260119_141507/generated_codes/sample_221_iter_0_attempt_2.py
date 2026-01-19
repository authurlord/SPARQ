import pandas as pd
import numpy as np

# Load the data
df = pd.read_csv('table.csv')

# Clean the data: replace 'x' and 'O' with NaN and drop rows with missing values
df_clean = df.replace({'x': np.nan, 'O': np.nan})
df_clean = df_clean.dropna()

# Convert relevant columns to numeric
df_clean['Composition'] = pd.to_numeric(df_clean['Composition'], errors='coerce')
df_clean['Drawing'] = pd.to_numeric(df_clean['Drawing'], errors='coerce')
df_clean['Color'] = pd.to_numeric(df_clean['Color'], errors='coerce')
df_clean['Expression'] = pd.to_numeric(df_clean['Expression'], errors='coerce')

# Compute correlation with Composition
correlation_drawing = df_clean['Composition'].corr(df_clean['Drawing'])
correlation_color = df_clean['Composition'].corr(df_clean['Color'])
correlation_expression = df_clean['Composition'].corr(df_clean['Expression'])

# Find the variable with the highest absolute correlation
max_corr = max(abs(correlation_drawing), abs(correlation_color), abs(correlation_expression))
if max_corr == abs(correlation_drawing):
    result = "Drawing"
elif max_corr == abs(correlation_color):
    result = "Color"
else:
    result = "Expression"

print(f"Final Answer: {result}")