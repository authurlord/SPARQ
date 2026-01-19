import pandas as pd

df = pd.read_csv('table.csv')
# Convert Year to integer for proper comparison
df['Year'] = df['Year'].astype(int)

# Sort by Year to ensure chronological order
df = df.sort_values('Year').reset_index(drop=True)

# Calculate the difference in Conservative councillors from previous year
df['diff'] = df['Conservative councillors'] - df['Conservative councillors'].shift(1)

# Find the year with the maximum increase
max_increase_year = df[df['diff'] == df['diff'].max()]['Year'].iloc[0]

print(f"Final Answer: {max_increase_year}")