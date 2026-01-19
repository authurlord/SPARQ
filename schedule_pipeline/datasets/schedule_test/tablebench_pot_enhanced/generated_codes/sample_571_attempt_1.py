import pandas as pd

df = pd.read_csv('table.csv')

# Filter data for 'all around' event in 2008
all_around_data = df[(df['event'] == 'all around') & (df['year'] == '2008')]

# Extract qualifying scores for 'olympic trials' and 'us championships'
olympic_trials_score = all_around_data[all_around_data['competition'] == 'olympic trials']['qualifying score'].values[0]
us_championships_score = all_around_data[all_around_data['competition'] == 'us championships']['qualifying score'].values[0]

# Calculate the change
change = float(us_championships_score) - float(olympic_trials_score)

print(f"Final Answer: {change:.1f}")