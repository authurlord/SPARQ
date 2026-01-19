import pandas as pd

df = pd.read_csv('table.csv')
# Convert 'p max ( bar )' to numeric, handling any potential parsing issues
df['p max ( bar )'] = pd.to_numeric(df['p max ( bar )'], errors='coerce')

# Calculate mean and standard deviation
mean_p_max = df['p max ( bar )'].mean()
std_p_max = df['p max ( bar )'].std()

print(f"Final Answer: {mean_p_max:.2f}, {std_p_max:.2f}")