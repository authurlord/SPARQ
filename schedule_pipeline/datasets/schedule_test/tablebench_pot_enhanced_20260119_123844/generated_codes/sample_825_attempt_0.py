import pandas as pd

df = pd.read_csv('table.csv')
# Convert 'p max ( bar )' to numeric
p_max = pd.to_numeric(df['p max ( bar )'])
# Calculate mean and standard deviation
mean_pmax = p_max.mean()
std_pmax = p_max.std()
print(f"Final Answer: {mean_pmax:.2f}, {std_pmax:.2f}")