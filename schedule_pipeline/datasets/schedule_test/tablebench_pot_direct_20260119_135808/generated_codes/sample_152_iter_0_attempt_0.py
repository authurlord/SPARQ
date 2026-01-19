import pandas as pd

df = pd.read_csv('table.csv')

# Analyze contribution of 'public', 'judges', and 'vote percentage' to 'result'
# Convert vote percentage to numeric for comparison
df['vote_percentage'] = df['vote percentage'].str.replace('%', '').astype(float)

# Identify the couple in 'bottom two'
bottom_two = df[df['result'] == 'bottom two']

# Check their scores
public_score_bottom = bottom_two['public'].values[0]
vote_percentage_bottom = bottom_two['vote percentage'].values[0]

# Compare with others
max_public = df['public'].max()
min_public = df['public'].min()

max_vote_percentage = df['vote percentage'].max()
min_vote_percentage = df['vote percentage'].min()

# Determine if 'public' or 'vote percentage' are key drivers
if min_public == public_score_bottom and min_vote_percentage == vote_percentage_bottom:
    print("Final Answer: public, vote percentage")
else:
    print("Final Answer: no clear impact")