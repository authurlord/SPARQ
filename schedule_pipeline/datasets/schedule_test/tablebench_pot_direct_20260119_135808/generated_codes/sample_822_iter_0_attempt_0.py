import pandas as pd

df = pd.read_csv('table.csv')
# Convert 'us viewers (millions)' to float for numerical operations
df['us viewers (millions)'] = pd.to_numeric(df['us viewers (millions)'], errors='coerce')

# Calculate mean and standard deviation
mean_viewers = df['us viewers (millions)'].mean()
std_viewers = df['us viewers (millions)'].std()

print(f"Final Answer: {mean_viewers:.2f}, {std_viewers:.2f}")