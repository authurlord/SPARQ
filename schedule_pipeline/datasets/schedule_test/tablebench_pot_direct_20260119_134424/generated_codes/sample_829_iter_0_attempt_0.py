import pandas as pd

df = pd.read_csv('table.csv')
# Convert 'bello' column to numeric, errors='coerce' will handle non-numeric values
bello_column = pd.to_numeric(df['bello'], errors='coerce')
# Calculate mean and standard deviation
mean_bello = bello_column.mean()
std_bello = bello_column.std()
print(f"Final Answer: {mean_bello:.1f}, {std_bello:.1f}")