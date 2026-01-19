import pandas as pd

df = pd.read_csv('table.csv')
# Convert 'average' column to numeric, coercing errors (like 'n/a') to NaN
df['average'] = pd.to_numeric(df['average'], errors='coerce')
# Calculate standard deviation of the average column, excluding NaN values
std_dev = df['average'].std()
print(f"Final Answer: {std_dev:.2f}")