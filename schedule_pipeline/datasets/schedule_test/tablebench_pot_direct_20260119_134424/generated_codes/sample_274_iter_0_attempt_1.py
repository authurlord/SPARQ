import pandas as pd

df = pd.read_csv('table.csv')
# Convert the first column to numeric for comparison
df['cost'] = pd.to_numeric(df['cost'].str.replace('/ kwp', ''), errors='coerce')

# Filter rows where cost <= 1400
filtered_df = df[df['cost'] <= 1400]

# Select columns where the header >= 2000 (e.g., '2000 kwh / kwp y' and above)
columns_to_select = [col for col in df.columns if col != 'cost' and int(col.split()[0]) >= 2000]
filtered_data = filtered_df[columns_to_select]

# Convert selected data to numeric and sum all values
total_cost = filtered_data.astype(float).sum().sum()
print(f"Final Answer: {total_cost}")