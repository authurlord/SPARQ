import pandas as pd

df = pd.read_csv('table.csv')
# Convert 'typhus' and 'typhoid fever' to numeric, handling non-numeric values like 'avg'
df['typhus'] = pd.to_numeric(df['typhus'], errors='coerce')
df['typhoid fever'] = pd.to_numeric(df['typhoid fever'], errors='coerce')

# Drop rows with missing values after conversion
df.dropna(subset=['typhus', 'typhoid fever'], inplace=True)

# Calculate correlation
correlation = df['typhus'].corr(df['typhoid fever'])

# Check if correlation is positive
if correlation > 0:
    print("Final Answer: Yes")
else:
    print("Final Answer: No")