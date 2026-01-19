import pandas as pd

df = pd.read_csv('table.csv')
# Calculate mean and standard deviation of 'Year built'
mean_year = df['Year built'].mean()
std_year = df['Year built'].std()
print(f"Final Answer: {mean_year:.1f}, {std_year:.1f}")