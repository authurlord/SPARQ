import pandas as pd

df = pd.read_csv('table.csv')

# Convert vote percentage to numeric (remove % sign)
df['vote_percentage_numeric'] = df['vote percentage'].str.replace('%', '').astype(float)

# Group by result and calculate mean for judges, public, and vote_percentage_numeric
grouped = df.groupby('result')[['judges', 'public', 'vote_percentage_numeric']].mean()

# Check if there's a significant difference in any of the factors between safe and bottom two
# We'll compare the means
judges_safe = grouped.loc['safe', 'judges']
judges_bottom = grouped.loc['bottom two', 'judges']

public_safe = grouped.loc['safe', 'public']
public_bottom = grouped.loc['bottom two', 'public']

vote_safe = grouped.loc['safe', 'vote_percentage_numeric']
vote_bottom = grouped.loc['bottom two', 'vote_percentage_numeric']

# Check for noticeable differences
judges_diff = abs(judges_safe - judges_bottom) > 0.5
public_diff = abs(public_safe - public_bottom) > 0.5
vote_diff = abs(vote_safe - vote_bottom) > 0.5

# If any factor shows a meaningful difference, they contribute
if judges_diff or public_diff or vote_diff:
    # List the contributing factors (only include those with difference)
    contributing_factors = []
    if judges_diff:
        contributing_factors.append('judges')
    if public_diff:
        contributing_factors.append('public')
    if vote_diff:
        contributing_factors.append('vote percentage')
    Final Answer: , '.join(contributing_factors)
else:
    Final Answer: no clear impact