import pandas as pd

df = pd.read_csv('table.csv')
# Convert 'against' column to numeric
df['against'] = pd.to_numeric(df['against'])
# Calculate mean and standard deviation
mean_against = df['against'].mean()
std_against = df['against'].std()
print(f"Final Answer: {mean_against:.1f}, {std_against:.1f}")