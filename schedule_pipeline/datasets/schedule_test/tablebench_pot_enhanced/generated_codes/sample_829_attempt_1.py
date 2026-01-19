import pandas as pd

df = pd.read_csv('table.csv')
# Convert 'bello' column to numeric
df['bello'] = pd.to_numeric(df['bello'], errors='coerce')
# Calculate mean and standard deviation
mean_bello = df['bello'].mean()
std_bello = df['bello'].std()
print(f"Final Answer: {mean_bello:.2f}, {std_bello:.2f}")