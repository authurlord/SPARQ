import pandas as pd

df = pd.read_csv('table.csv')
# Convert 'typhus' column to numeric
df['typhus'] = pd.to_numeric(df['typhus'], errors='coerce')

# Calculate the year-over-year change in typhus cases
df['typhus_change'] = df['typhus'].diff()

# Find the year with the maximum decrease (most negative change)
min_decrease_year = df.loc[df['typhus_change'].idxmin(), 'year']

print(f"Final Answer: {min_decrease_year}")