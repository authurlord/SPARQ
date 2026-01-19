import pandas as pd

df = pd.read_csv('table.csv')
# Convert 'Year' to integer
df['Year'] = pd.to_numeric(df['Year'])
# Calculate the difference in Conservative councillors
df['diff'] = df['Conservative councillors'].diff()
# Find the row with the maximum increase
max_increase_row = df.loc[df['diff'].idxmax()]
# Get the year with the maximum increase
final_year = max_increase_row['Year']
print(f"Final Answer: {final_year}")