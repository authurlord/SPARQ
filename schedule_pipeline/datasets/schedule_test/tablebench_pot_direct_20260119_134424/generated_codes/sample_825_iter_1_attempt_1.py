import pandas as pd

# Load the data
df = pd.read_csv('table.csv')

# Convert 'p max ( bar )' to numeric, coercing errors to NaN
df['p max ( bar )'] = pd.to_numeric(df['p max ( bar )'], errors='coerce')

# Calculate mean and standard deviation
mean_pmax = df['p max ( bar )'].mean()
std_pmax = df['p max ( bar )'].std()

print(f"Final Answer: {mean_pmax:.2f}, {std_pmax:.2f}")