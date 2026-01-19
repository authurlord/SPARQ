import pandas as pd

df = pd.read_csv('table.csv')
# Convert the columns to numeric
df['1990 - 95'] = pd.to_numeric(df['1990 - 95'])
df['2006 - 10'] = pd.to_numeric(df['2006 - 10'])

# Calculate correlation coefficient
correlation = df['1990 - 95'].corr(df['2006 - 10'])
print(f"Final Answer: {correlation:.3f}")