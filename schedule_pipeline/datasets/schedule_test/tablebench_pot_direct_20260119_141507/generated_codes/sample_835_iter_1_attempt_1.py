import pandas as pd

df = pd.read_csv('table.csv')
# Convert 'against' column to numeric, coercing errors to NaN if any
df['against'] = pd.to_numeric(df['against'], errors='coerce')

# Calculate mean and standard deviation of the 'against' column
mean_against = df['against'].mean()
std_against = df['against'].std()

print(f"Final Answer: {mean_against:.1f}, {std_against:.1f}")