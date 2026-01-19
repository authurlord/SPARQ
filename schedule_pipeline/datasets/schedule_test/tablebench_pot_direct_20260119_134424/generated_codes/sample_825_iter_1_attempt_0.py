import pandas as pd

df = pd.read_csv('table.csv')
# Convert 'p max (bar)' column to numeric, coercing errors to NaN
df['p max ( bar )'] = pd.to_numeric(df['p max ( bar )'], errors='coerce')
# Drop rows where conversion failed
df.dropna(subset=['p max ( bar )'], inplace=True)
# Calculate mean and standard deviation
mean_pmax = df['p max ( bar )'].mean()
std_pmax = df['p max ( bar )'].std()
print(f"Final Answer: {mean_pmax:.2f}, {std_pmax:.2f}")