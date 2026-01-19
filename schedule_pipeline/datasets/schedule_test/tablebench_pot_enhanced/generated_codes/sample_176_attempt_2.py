import pandas as pd

df = pd.read_csv('table.csv')
# Convert 'viewers (in millions)' to float for numerical operations
df['viewers (in millions)'] = pd.to_numeric(df['viewers (in millions)'], errors='coerce')

# Calculate correlation between 'rank' and 'viewers (in millions)'
correlation = df['rank'].astype(float).corr(df['viewers (in millions)'])

# Check if correlation is significant (absolute value > 0.5)
if abs(correlation) > 0.5:
    print("Final Answer: rank")
else:
    print("Final Answer: no clear impact")