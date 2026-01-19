import pandas as pd

df = pd.read_csv('table.csv')
# Calculate mean and standard deviation of 'p max ( bar )'
mean_pmax = df['p max ( bar )'].mean()
std_pmax = df['p max ( bar )'].std()

print(f"Final Answer: {mean_pmax:.2f}, {std_pmax:.2f}")