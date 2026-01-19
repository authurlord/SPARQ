import pandas as pd

df = pd.read_csv('table.csv')
# Drop the row with invalid year '191822'
df = df[df['year'] != '191822']

# Convert 'typhus' and 'typhoid fever' to numeric, coercing errors to NaN
df['typhus'] = pd.to_numeric(df['typhus'], errors='coerce')
df['typhoid fever'] = pd.to_numeric(df['typhoid fever'], errors='coerce')

# Drop rows with NaN values in either column
df.dropna(subset=['typhus', 'typhoid fever'], inplace=True)

# Calculate correlation coefficient
correlation = df['typhus'].corr(df['typhoid fever'])

# Output result
if correlation > 0:
    print("Final Answer: Yes")
else:
    print("Final Answer: No")