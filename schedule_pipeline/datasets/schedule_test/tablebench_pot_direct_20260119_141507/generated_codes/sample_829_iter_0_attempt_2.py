import pandas as pd

df = pd.read_csv('table.csv')
# Calculate mean and standard deviation of 'bello' column
mean_bello = df['bello'].mean()
std_bello = df['bello'].std()
print(f"Final Answer: {mean_bello:.1f}, {std_bello:.1f}")