import pandas as pd

df = pd.read_csv('table.csv')

# Filter rows where event is 'all around'
all_around_rows = df[df['event'] == 'all around']

# Filter only those with competition being 'olympic trials' or 'us championships'
relevant_rows = all_around_rows[all_around_rows['competition'].isin(['olympic trials', 'us championships'])]

# Extract qualifying scores for 'olympic trials' and 'us championships'
# We need to group by the same participant? But no participant ID is given.
# Since each row is a unique event, and only one row per competition, we can just get the two values.

# Get qualifying score for 'olympic trials' and 'us championships'
olympic_qual_score = relevant_rows[relevant_rows['competition'] == 'olympic trials']['qualifying score'].iloc[0]
us_champs_qual_score = relevant_rows[relevant_rows['competition'] == 'us championships']['qualifying score'].iloc[0]

# Compute change (us champs - olympic trials)
change = us_champs_qual_score - olympic_qual_score

# Since only one such pair exists, average is just the change
print(f"Final Answer: {change:.2f}")