import pandas as pd

# Load the data
df = pd.read_csv('table.csv')

# Drop the first row (header) since it's duplicated
df = df.iloc[1:]

# Convert relevant columns to numeric
df['% learning in latvian'] = pd.to_numeric(df['% learning in latvian'], errors='coerce')
df['total'] = pd.to_numeric(df['total'], errors='coerce')

# Calculate % learning in Russian
df['% learning in russian'] = 100 - df['% learning in latvian'] - df['others'].astype(float)

# Remove rows where calculations fail
df = df.dropna(subset=['% learning in russian'])

# Compute correlation between total students and the two percentages
correlation_latvian = df['% learning in latvian'].corr(df['total'])
correlation_russian = df['% learning in russian'].corr(df['total'])

print(f"Correlation between total students and % learning in Latvian: {correlation_latvian:.3f}")
print(f"Correlation between total students and % learning in Russian: {correlation_russian:.3f}")

# Determine which is more significantly influenced
if abs(correlation_latvian) > abs(correlation_russian):
    result = "latvian"
else:
    result = "russian"

Final Answer: latvian