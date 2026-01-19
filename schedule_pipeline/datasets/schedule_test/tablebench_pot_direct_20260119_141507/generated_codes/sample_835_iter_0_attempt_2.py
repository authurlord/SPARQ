import pandas as pd

df = pd.read_csv('table.csv')
# Calculate mean and standard deviation of the 'against' column
mean_against = df['against'].mean()
std_against = df['against'].std()
print(f"Final Answer: {mean_against:.1f}, {std_against:.1f}")