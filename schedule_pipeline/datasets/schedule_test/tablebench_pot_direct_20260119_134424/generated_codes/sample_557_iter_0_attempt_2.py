import pandas as pd

df = pd.read_csv('table.csv')
# Convert Year to integer for proper sorting
df['Year'] = pd.to_numeric(df['Year'])
# Sort by Year to ensure chronological order
df = df.sort_values('Year')
# Calculate the difference in Conservative councillors from the previous year
df['diff'] = df['Conservative councillors'].diff()
# Find the row with the maximum positive difference
max_increase_year = df.loc[df['diff'].idxmax()]['Year']
print(f"Final Answer: {max_increase_year}")