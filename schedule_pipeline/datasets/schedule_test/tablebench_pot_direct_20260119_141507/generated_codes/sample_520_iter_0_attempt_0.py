import pandas as pd

df = pd.read_csv('table.csv')

# Convert passenger numbers to integers
df['2008'] = pd.to_numeric(df['2008'], errors='coerce')
df['2009'] = pd.to_numeric(df['2009'], errors='coerce')

# Calculate growth rate from 2008 to 2009
df['growth_rate'] = ((df['2009'] - df['2008']) / df['2008']) * 100

# Find the airport with the maximum growth rate
max_growth_row = df.loc[df['growth_rate'].idxmax()]
airport_with_max_growth = max_growth_row['airport']

print(f"Final Answer: {airport_with_max_growth}")