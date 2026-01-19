import pandas as pd

df = pd.read_csv('table.csv')
# Convert the columns to numeric type to ensure proper calculation
df['km from kingston'] = pd.to_numeric(df['km from kingston'])
df['length meters'] = pd.to_numeric(df['length meters'])

# Calculate correlation coefficient
correlation = df['km from kingston'].corr(df['length meters'])
print(f"Final Answer: {correlation:.3f}")