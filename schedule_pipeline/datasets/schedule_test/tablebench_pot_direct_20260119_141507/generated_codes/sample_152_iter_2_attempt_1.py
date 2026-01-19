import pandas as pd

# Load the data
df = pd.read_csv('table.csv')

# Convert vote percentage to numeric (remove '%')
df['vote percentage'] = df['vote percentage'].str.replace('%', '').astype(float)

# Group by result and compute means
result_summary = df.groupby('result').agg({
    'judges': 'mean',
    'public': 'mean',
    'vote percentage': 'mean'
}).round(2)

# Check if there's a significant difference in means between 'safe' and 'bottom two'
safe_mean = result_summary.loc['safe']
bottom_two_mean = result_summary.loc['bottom two']

# Check if differences are notable
diff_judges = abs(safe_mean['judges'] - bottom_two_mean['judges'])
diff_public = abs(safe_mean['public'] - bottom_two_mean['public'])
diff_vote = abs(safe_mean['vote percentage'] - bottom_two_mean['vote percentage'])

# If any difference is substantial (say > 2 or so), we say that factor has impact
# Otherwise, no clear impact

if (diff_judges > 1) or (diff_public > 1) or (diff_vote > 5):
    # At least one factor shows a clear difference
    contributing_factors = []
    if diff_judges > 1:
        contributing_factors.append('judges')
    if diff_public > 1:
        contributing_factors.append('public')
    if diff_vote > 5:
        contributing_factors.append('vote percentage')
    Final Answer: , ".join(contributing_factors)
else:
    Final Answer: no clear impact