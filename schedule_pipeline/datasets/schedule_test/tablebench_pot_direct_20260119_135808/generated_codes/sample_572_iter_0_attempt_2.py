import pandas as pd

df = pd.read_csv('table.csv')
# Convert 'typhus' column to numeric
df['typhus'] = pd.to_numeric(df['typhus'], errors='coerce')

# Calculate year-on-year change in typhus cases
df['change'] = df['typhus'].diff()

# Find the year with the maximum decrease (most negative change)
min_change_year = df.loc[df['change'].idxmin(), 'year']

print(f"Final Answer: {min_change_year}")