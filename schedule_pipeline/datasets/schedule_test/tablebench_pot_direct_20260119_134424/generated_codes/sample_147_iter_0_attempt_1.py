import pandas as pd

df = pd.read_csv('table.csv')

# Convert 'number of examinees' to integer
df['number of examinees'] = pd.to_numeric(df['number of examinees'])

# Convert 'pass percentage' to float by removing '%' and converting
df['pass percentage'] = df['pass percentage'].str.replace('%', '').astype(float)

# Calculate correlation between number of examinees and pass percentage
correlation = df['number of examinees'].corr(df['pass percentage'])

print(f"Final Answer: {correlation:.2f}")