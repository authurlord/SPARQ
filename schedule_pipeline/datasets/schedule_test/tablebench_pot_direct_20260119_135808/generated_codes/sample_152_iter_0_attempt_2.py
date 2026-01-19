import pandas as pd

df = pd.read_csv('table.csv')

# Separate rows by result
safe_couples = df[df['result'] == 'safe']
bottom_two_couples = df[df['result'] == 'bottom two']

# Calculate average values for each factor in both groups
avg_judges_safe = safe_couples['judges'].mean()
avg_judges_bottom = bottom_two_couples['judges'].mean()

avg_public_safe = safe_couples['public'].mean()
avg_public_bottom = bottom_two_couples['public'].mean()

avg_vote_safe = safe_couples['vote percentage'].str.rstrip('%').astype(float).mean()
avg_vote_bottom = bottom_two_couples['vote percentage'].str.rstrip('%').astype(float).mean()

# Compare differences
diff_judges = abs(avg_judges_safe - avg_judges_bottom)
diff_public = abs(avg_public_safe - avg_public_bottom)
diff_vote = abs(avg_vote_safe - avg_vote_bottom)

# Determine which factor has the largest difference
max_diff = max(diff_judges, diff_public, diff_vote)
if max_diff == diff_judges:
    contribution = 'judges'
elif max_diff == diff_public:
    contribution = 'public'
elif max_diff == diff_vote:
    contribution = 'vote percentage'
else:
    contribution = 'no clear impact'

print(f"Final Answer: {contribution}")