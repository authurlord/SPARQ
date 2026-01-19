import pandas as pd

df = pd.read_csv('table.csv')
# Convert 'rank' and 'viewers (in millions)' to numeric
df['rank'] = pd.to_numeric(df['rank'])
df['viewers (in millions)'] = pd.to_numeric(df['viewers (in millions)'])

# Calculate correlation between rank and viewers
correlation = df['rank'].corr(df['viewers (in millions)'])

# Check if correlation is significant (absolute value > 0.5)
if abs(correlation) > 0.5:
    print("Final Answer: rank")
else:
    print("Final Answer: no clear impact")