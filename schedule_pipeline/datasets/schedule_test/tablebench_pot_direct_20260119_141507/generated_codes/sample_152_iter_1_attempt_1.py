import pandas as pd

# Load the data
df = pd.read_csv('table.csv')

# Convert vote percentage to numeric (remove % and convert)
df['vote percentage'] = df['vote percentage'].str.rstrip('%').astype(float)

# Group by result and compute mean of judges, public, and vote percentage
grouped = df.groupby('result')[['judges', 'public', 'vote percentage']].mean()

# Check if there's a noticeable difference in the means
# If all means are similar across groups, then no clear impact
judges_diff = abs(grouped.loc['safe', 'judges'] - grouped.loc['bottom two', 'judges'])
public_diff = abs(grouped.loc['safe', 'public'] - grouped.loc['bottom two', 'public'])
vote_diff = abs(grouped.loc['safe', 'vote percentage'] - grouped.loc['bottom two', 'vote percentage'])

# If any of the differences are significant (say > 2), we consider it to have impact
# For simplicity, if any difference is non-zero, we assume some impact
if judges_diff > 1 or public_diff > 1 or vote_diff > 1:
    # We can say which ones contribute
    contributing_factors = []
    if judges_diff > 1:
        contributing_factors.append('judges')
    if public_diff > 1:
        contributing_factors.append('public')
    if vote_diff > 1:
        contributing_factors.append('vote percentage')
    if contributing_factors:
        Final Answer: , '.join(contributing_factors)
else:
    Final Answer: no clear impact