import pandas as pd
import numpy as np

df = pd.read_csv('table.csv')

# Convert 'Expression' and other categorical values like 'O', 'x' to NaN for numerical analysis
df['Color'] = df['Color'].replace({'O': np.nan, 'x': np.nan})
df['Drawing'] = df['Drawing'].replace({'O': np.nan, 'x': np.nan})
df['Expression'] = df['Expression'].replace({'O': np.nan, 'x': np.nan})

# Drop rows with NaN in the relevant columns
df_clean = df.dropna(subset=['Composition', 'Drawing', 'Color', 'Expression'])

# Compute correlation with Composition
correlation_drawing = df_clean['Drawing'].corr(df_clean['Composition'])
correlation_color = df_clean['Color'].corr(df_clean['Composition'])
correlation_expression = df_clean['Expression'].corr(df_clean['Composition'])

# Find the variable with the highest absolute correlation
max_corr = max(abs(correlation_drawing), abs(correlation_color), abs(correlation_expression))
if max_corr == abs(correlation_drawing):
    result = "Drawing"
elif max_corr == abs(correlation_color):
    result = "Color"
else:
    result = "Expression"

print(f"Final Answer: {result}")