import pandas as pd

df = pd.read_csv('table.csv')
# Convert the required columns to float
df['1990 - 95'] = df['1990 - 95'].astype(float)
df['2006 - 10'] = df['2006 - 10'].astype(float)

# Calculate correlation coefficient
correlation = df['1990 - 95'].corr(df['2006 - 10'])
print(f"Final Answer: {correlation:.3f}")