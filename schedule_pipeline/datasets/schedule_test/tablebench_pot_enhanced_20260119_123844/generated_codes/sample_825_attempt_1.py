import pandas as pd

df = pd.read_csv('table.csv')
# Convert 'p max ( bar )' to numeric
p_max = pd.to_numeric(df['p max ( bar )'], errors='coerce')
# Calculate mean and standard deviation
mean_p_max = p_max.mean()
std_p_max = p_max.std()
print(f"Final Answer: {mean_p_max:.2f}, {std_p_max:.2f}")