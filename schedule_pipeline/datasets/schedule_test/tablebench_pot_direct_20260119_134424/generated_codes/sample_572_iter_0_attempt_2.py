import pandas as pd

df = pd.read_csv('table.csv')
# Convert 'typhus' column to numeric
df['typhus'] = pd.to_numeric(df['typhus'], errors='coerce')

# Calculate the difference in typhus cases from the previous year
df['diff'] = df['typhus'].diff()

# Find the year with the maximum decrease (most negative difference)
max_decrease_year = df.loc[df['diff'].idxmin(), 'year']

print(f"Final Answer: {max_decrease_year}")