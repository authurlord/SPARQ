import pandas as pd

df = pd.read_csv('table.csv')
# Filter rows from 1935 to 1943
filtered_df = df[(df['Year'] >= '1935') & (df['Year'] <= '1943')]
# Convert 'Quantity withdrawn' to numeric, coercing errors to NaN
filtered_df['Quantity withdrawn'] = pd.to_numeric(filtered_df['Quantity withdrawn'], errors='coerce')
# Calculate the mean of the 'Quantity withdrawn' column
mean_withdrawn = filtered_df['Quantity withdrawn'].mean()
print(f"Final Answer: {mean_withdrawn:.1f}")