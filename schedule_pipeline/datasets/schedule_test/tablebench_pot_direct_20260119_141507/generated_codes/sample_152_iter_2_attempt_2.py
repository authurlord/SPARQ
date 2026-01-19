import pandas as pd

df = pd.read_csv('table.csv')

# Convert vote percentage to numeric (remove %)
df['vote percentage'] = df['vote percentage'].str.rstrip('%').astype(float)

# Group by 'result' and compute means
grouped = df.groupby('result')[['judges', 'public', 'vote percentage']].mean()

# Check if the means differ significantly across groups
# If all means are similar, then no clear impact
# Otherwise, we infer impact

# For simplicity, if the difference in means is minimal, we say no clear impact
# We'll compare the mean values across groups

safe_means = grouped.loc['safe']
bottom_two_means = grouped.loc['bottom two']

# Check if the differences are large enough to suggest impact
diff_judges = abs(safe_means['judges'] - bottom_two_means['judges'])
diff_public = abs(safe_means['public'] - bottom_two_means['public'])
diff_vote = abs(safe_means['vote percentage'] - bottom_two_means['vote percentage'])

# If all differences are small (say < 2), then no clear impact
if (diff_judges < 2).all() and (diff_public < 2).all() and (diff_vote < 2).all():
    print("Final Answer: no clear impact")
else:
    print(f"Final Answer: judges, public, vote percentage")