import pandas as pd

df = pd.read_csv('table.csv')
# Convert columns to appropriate types
df['Year'] = pd.to_numeric(df['Year'])
df['Conservative councillors'] = pd.to_numeric(df['Conservative councillors'])

# Calculate the difference in Conservative councillors from the previous year
df['diff'] = df['Conservative councillors'].diff()

# Find the row with the maximum increase
max_increase_row = df.loc[df['diff'].idxmax()]

# Extract the year with the maximum increase
year_with_max_increase = max_increase_row['Year']
print(f"Final Answer: {year_with_max_increase}")