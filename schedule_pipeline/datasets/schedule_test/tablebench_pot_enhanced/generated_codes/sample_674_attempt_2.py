import pandas as pd

df = pd.read_csv('table.csv')
# Convert 'poles' and 'wins' to numeric, in case they are not already
df['poles'] = pd.to_numeric(df['poles'], errors='coerce')
df['wins'] = pd.to_numeric(df['wins'], errors='coerce')

# Calculate the correlation coefficient between 'poles' and 'wins'
correlation = df['poles'].corr(df['wins'])
print(f"Final Answer: {correlation:.4f}")