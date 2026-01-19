import pandas as pd

df = pd.read_csv('table.csv')
# Convert 'viewers (in millions)' and 'rank' to numeric
df['viewers (in millions)'] = pd.to_numeric(df['viewers (in millions)'])
df['rank'] = pd.to_numeric(df['rank'])

# Calculate correlation between rank and viewers
correlation = df['rank'].corr(df['viewers (in millions)'])

# Check if correlation is significant (absolute value > 0.5)
if abs(correlation) > 0.5:
    print(f"Final Answer: rank")
else:
    print(f"Final Answer: no clear impact")