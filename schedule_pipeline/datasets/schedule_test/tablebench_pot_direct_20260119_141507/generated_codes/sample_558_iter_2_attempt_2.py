import pandas as pd
import numpy as np

df = pd.read_csv('table.csv')
# Convert Year to numeric, treating '—' as NaN
df['Year'] = pd.to_numeric(df['Year'], errors='coerce')
# Filter rows where Year is between 1935 and 1943 inclusive
filtered_df = df[(df['Year'] >= 1935) & (df['Year'] <= 1943)]
# Calculate the average of "Quantity withdrawn" for the filtered rows
average_withdrawn = filtered_df['Quantity withdrawn'].mean()
print(f"Final Answer: {average_withdrawn:.1f}")