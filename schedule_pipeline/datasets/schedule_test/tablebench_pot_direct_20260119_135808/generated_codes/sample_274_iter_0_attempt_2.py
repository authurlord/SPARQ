import pandas as pd

df = pd.read_csv('table.csv')
# Clean the 'cost' column to numeric
df['cost'] = df['cost'].str.replace('/ kwp', '').astype(float)

# Filter rows where cost <= 1400
filtered_df = df[df['cost'] <= 1400]

# Select columns where the header >= 2000 (from '2000 kwh / kwp y' onwards)
energy_columns = [col for col in df.columns if col != 'cost' and int(col.split()[0]) >= 2000]

# Extract the values from the filtered rows and selected columns
values = filtered_df[energy_columns].astype(float).values

# Sum all values
total_cost = values.sum()
print(f"Final Answer: {total_cost}")