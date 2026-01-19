import pandas as pd

df = pd.read_csv('table.csv')

# Filter rows for 'all around' event in 2008
filtered = df[(df['event'] == 'all around') & (df['year'] == '2008')]

# Extract qualifying scores for 'olympic trials' and 'us championships'
qualifying_trials = filtered[filtered['competition'] == 'olympic trials']['qualifying score'].values[0]
qualifying_champs = filtered[filtered['competition'] == 'us championships']['qualifying score'].values[0]

# Calculate the change
change = qualifying_champs - qualifying_trials

print(f"Final Answer: {change:.3f}")