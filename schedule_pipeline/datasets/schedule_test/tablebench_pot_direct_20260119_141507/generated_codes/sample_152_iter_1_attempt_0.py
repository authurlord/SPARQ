import pandas as pd

# Load the data
df = pd.read_csv('table.csv')

# Convert 'vote percentage' to numeric (remove % and convert)
df['vote_percentage_numeric'] = df['vote percentage'].str.replace('%', '').astype(float)

# Group by 'result' and compute mean for judges, public, and vote_percentage_numeric
grouped = df.groupby('result').agg({
    'judges': 'mean',
    'public': 'mean',
    'vote_percentage_numeric': 'mean'
}).round(2)

# Check differences between safe and bottom two
safe_group = grouped.loc['safe']
bottom_group = grouped.loc['bottom two']

# Compute differences
diff_judges = safe_group['judges'] - bottom_group['judges']
diff_public = safe_group['public'] - bottom_group['public']
diff_vote = safe_group['vote_percentage_numeric'] - bottom_group['vote_percentage_numeric']

# Determine which factor has the largest difference
if diff_vote.abs().max() > diff_judges.abs().max() and diff_vote.abs().max() > diff_public.abs().max():
    final_factor = 'vote percentage'
elif diff_judges.abs().max() > diff_vote.abs().max() and diff_judges.abs().max() > diff_public.abs().max():
    final_factor = 'judges'
elif diff_public.abs().max() > diff_vote.abs().max() and diff_public.abs().max() > diff_judges.abs().max():
    final_factor = 'public'
else:
    final_factor = 'no clear impact'

print(f"Final Answer: {final_factor}")