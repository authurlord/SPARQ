import pandas as pd

df = pd.read_csv('table.csv')

# Filter rows where event is 'all around'
all_around_data = df[df['event'] == 'all around']

# Separate final ranks by competition
olympic_trials_rank = all_around_data[all_around_data['competition'] == 'olympic trials']['final - rank'].dropna().mean()
us_champs_rank = all_around_data[all_around_data['competition'] == 'us championships']['final - rank'].dropna().mean()

# Compare the ranks
print(f"Final Answer: {olympic_trials_rank}, {us_champs_rank}")