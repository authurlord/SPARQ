import pandas as pd

df = pd.read_csv('table.csv')
# Convert 'Build date' to integer
df['Build date'] = pd.to_numeric(df['Build date'])
# Calculate standard deviation
std_dev = df['Build date'].std()
print(f"Final Answer: {std_dev:.2f}")