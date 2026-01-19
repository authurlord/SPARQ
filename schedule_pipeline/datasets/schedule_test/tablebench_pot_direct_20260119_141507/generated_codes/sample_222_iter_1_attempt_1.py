import pandas as pd

# Load the data
df = pd.read_csv('table.csv')

# Drop the first row (header row) as it's duplicated
df = df.iloc[1:]

# Convert the percentage columns to numeric
df['% learning in latvian'] = pd.to_numeric(df['% learning in latvian'], errors='coerce')
df['total'] = pd.to_numeric(df['total'], errors='coerce')

# Calculate % learning in russian as (100 - % learning in latvian)
df['% learning in russian'] = 100 - df['% learning in latvian']

# Drop rows with NaN values
df = df.dropna()

# Extract total and the two proportions
totals = df['total']
latvian_prop = df['% learning in latvian']
russian_prop = df['% learning in russian']

# Compute correlation between total students and each proportion
corr_latvian = totals.corr(latvian_prop)
corr_russian = totals.corr(russian_prop)

# Determine which one has stronger influence
if abs(corr_latvian) > abs(corr_russian):
    more_influenced = 'latvian'
else:
    more_influenced = 'russian'

print(f"Final Answer: {more_influenced}, {abs(corr_latvian):.3f}, {abs(corr_russian):.3f}")