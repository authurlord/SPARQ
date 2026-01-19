import pandas as pd

df = pd.read_csv('table.csv')
# Filter rows from 1935 to 1943 inclusive
filtered_df = df[(df['Year'].astype(str).str.startswith('193')) | (df['Year'].astype(str).str.startswith('194')) & (df['Year'].between(1935, 1943))]
# Extract 'Quantity withdrawn' column and calculate mean
withdrawn_avg = filtered_df['Quantity withdrawn'].mean()
print(f"Final Answer: {withdrawn_avg:.1f}")