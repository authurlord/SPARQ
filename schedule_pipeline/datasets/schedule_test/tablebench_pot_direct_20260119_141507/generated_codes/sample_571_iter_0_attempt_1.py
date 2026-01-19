import pandas as pd

df = pd.read_csv('table.csv')
# Filter rows where event is 'all around' and year is 2008
filtered_df = df[(df['event'] == 'all around') & (df['year'] == '2008')]

# Extract qualifying scores for 'olympic trials' and 'us championships'
olympic_qualifying = filtered_df[filtered_df['competition'] == 'olympic trials']['qualifying score'].iloc[0]
us_championships_qualifying = filtered_df[filtered_df['competition'] == 'us championships']['qualifying score'].iloc[0]

# Calculate the change
change = us_championships_qualifying - olympic_qualifying

# Since there's only one such pair, the average change is just this value
print(f"Final Answer: {change:.3f}")