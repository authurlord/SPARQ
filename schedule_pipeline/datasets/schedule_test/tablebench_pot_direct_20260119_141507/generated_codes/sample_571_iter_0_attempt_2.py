import pandas as pd

df = pd.read_csv('table.csv')

# Filter rows for 'all around' event in 2008
filtered_df = df[(df['event'] == 'all around') & (df['year'] == '2008')]

# Separate qualifying scores for 'olympic trials' and 'us championships'
olympic_qual_score = filtered_df[filtered_df['competition'] == 'olympic trials']['qualifying score'].values[0]
us_champs_qual_score = filtered_df[filtered_df['competition'] == 'us championships']['qualifying score'].values[0]

# Calculate the change
change = us_champs_qual_score - olympic_qual_score

print(f"Final Answer: {change:.3f}")