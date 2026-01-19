import pandas as pd

df = pd.read_csv('table.csv')

# Convert vote percentage to numeric for analysis
df['vote percentage'] = df['vote percentage'].str.replace('%', '').astype(float)

# Separate data into 'safe' and 'bottom two' groups
safe_group = df[df['result'] == 'safe']
bottom_two_group = df[df['result'] == 'bottom two']

# Calculate average values for each factor
avg_judges_safe = safe_group['judges'].mean()
avg_judges_bottom = bottom_two_group['judges'].mean()

avg_public_safe = safe_group['public'].mean()
avg_public_bottom = bottom_two_group['public'].mean()

avg_vote_safe = safe_group['vote percentage'].mean()
avg_vote_bottom = bottom_two_group['vote percentage'].mean()

# Compare differences
diff_judges = abs(avg_judges_safe - avg_judges_bottom)
diff_public = abs(avg_public_safe - avg_public_bottom)
diff_vote = abs(avg_vote_safe - avg_vote_bottom)

# Identify the factor with the largest difference
max_diff = max(diff_judges, diff_public, diff_vote)
if max_diff == diff_judges:
    result = 'judges'
elif max_diff == diff_public:
    result = 'public'
elif max_diff == diff_vote:
    result = 'vote percentage'
else:
    result = 'no clear impact'

print(f"Final Answer: {result}")