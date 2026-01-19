import pandas as pd

df = pd.read_csv('table.csv')

# Filter for 'all around' event in 'olympic trials' and 'us championships' in 2008
olympic_trials = df[(df['event'] == 'all around') & (df['competition'] == 'olympic trials')]
us_championships = df[(df['event'] == 'all around') & (df['competition'] == 'us championships')]

# Extract qualifying scores
qualifying_score_olympic_trials = float(olympic_trials['qualifying score'].values[0])
qualifying_score_us_championships = float(us_championships['qualifying score'].values[0])

# Calculate the change
change = qualifying_score_olympic_trials - qualifying_score_us_championships

print(f"Final Answer: {change:.1f}")