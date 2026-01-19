import pandas as pd

df = pd.read_csv('table.csv')
# Convert 'viewers (in millions)' to float
df['viewers (in millions)'] = pd.to_numeric(df['viewers (in millions)'], errors='coerce')

# Check correlation between 'rank' and 'viewers (in millions)'
correlation = df['rank'].astype(float).corr(df['viewers (in millions)'])

# If correlation is close to 0, then no clear impact
if abs(correlation) < 0.3:
    print("Final Answer: no clear impact")
else:
    print("Final Answer: rank")