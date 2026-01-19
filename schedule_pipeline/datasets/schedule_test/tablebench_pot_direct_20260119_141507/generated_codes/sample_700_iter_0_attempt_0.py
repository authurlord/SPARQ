import pandas as pd

df = pd.read_csv('table.csv')
# Remove the total row (last row) as it's not a year
df = df[df['year'] != 'total']

# Convert success rate to float by removing the '%' sign
df['success rate'] = df['success rate'].str.rstrip('%').astype(float)

# Find the year with the highest success rate
max_rate_row = df.loc[df['success rate'].idxmax()]
highest_success_year = max_rate_row['year']
print(f"Final Answer: {highest_success_year}")