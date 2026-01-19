import pandas as pd

df = pd.read_csv('table.csv')
# Convert 'total' column to integer type to avoid type errors
df['total'] = df['total'].astype(int)
# Calculate the standard deviation of the 'total' column
std_total = df['total'].std()
print(f"Final Answer: {std_total:.1f}")