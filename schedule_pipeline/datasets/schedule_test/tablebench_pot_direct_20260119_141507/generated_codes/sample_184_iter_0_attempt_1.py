import pandas as pd

df = pd.read_csv('table.csv')

# Filter rows where event is 'all around'
all_around_data = df[df['event'] == 'all around']

# Separate final ranks by competition
olympic_trials_rank = all_around_data[all_around_data['competition'] == 'Olympic trials']['final - rank'].dropna()
us_championships_rank = all_around_data[all_around_data['competition'] == 'US championships']['final - rank'].dropna()

# Calculate average final ranks
avg_olympic_trials_rank = olympic_trials_rank.mean()
avg_us_championships_rank = us_championships_rank.mean()

print(f"Final Answer: {avg_olympic_trials_rank:.1f}, {avg_us_championships_rank:.1f}")