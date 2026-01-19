import pandas as pd

df = pd.read_csv('table.csv')

# Convert all numeric columns to integers
for col in df.columns[1:]:
    df[col] = pd.to_numeric(df[col], errors='coerce')

# Count municipalities where Spanish population is >= 40,000
spanish_pop = df['spanish']
count = (spanish_pop >= 40000).sum()

print(f"Final Answer: {count}")