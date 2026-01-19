import pandas as pd

df = pd.read_csv('table.csv')
# Convert 'bello' column to numeric, handling any potential non-numeric values
bello_values = pd.to_numeric(df['bello'], errors='coerce')
# Drop NaN values if any
bello_values = bello_values.dropna()

# Calculate mean and standard deviation
mean_bello = bello_values.mean()
std_bello = bello_values.std()

print(f"Final Answer: {mean_bello:.2f}, {std_bello:.2f}")