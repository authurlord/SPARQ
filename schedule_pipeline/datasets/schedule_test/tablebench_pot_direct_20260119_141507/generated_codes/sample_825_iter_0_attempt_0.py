import pandas as pd

df = pd.read_csv('table.csv')
# Calculate mean and standard deviation of 'p max ( bar )'
mean_p_max = df['p max ( bar )'].mean()
std_p_max = df['p max ( bar )'].std()
print(f"Final Answer: {mean_p_max:.2f}, {std_p_max:.2f}")