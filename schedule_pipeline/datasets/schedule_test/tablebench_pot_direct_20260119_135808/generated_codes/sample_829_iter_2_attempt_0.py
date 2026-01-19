import pandas as pd

# Load the data
df = pd.read_csv('table.csv')

# Convert 'bello' column to numeric (integer)
df['bello'] = pd.to_numeric(df['bello'])

# Calculate mean and standard deviation
mean_bello = df['bello'].mean()
std_bello = df['bello'].std()

# Output the results
print(f"Final Answer: {mean_bello:.2f}, {std_bello:.2f}")