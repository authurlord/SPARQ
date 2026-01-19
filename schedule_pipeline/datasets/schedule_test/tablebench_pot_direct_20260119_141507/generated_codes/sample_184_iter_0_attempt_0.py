import pandas as pd

df = pd.read_csv('table.csv')

# Filter rows where event is 'all around'
all_around_data = df[df['event'] == 'all around']

# Group by competition and calculate mean final rank
rank_by_competition = all_around_data.groupby('competition')['final - rank'].mean()

# Extract the final ranks for Olympic trials and US championships
olympic_trials_rank = rank_by_competition.get('olympic trials', None)
us_championships_rank = rank_by_competition.get('us championships', None)

# If values are not available, handle accordingly
if olympic_trials_rank is not None and us_championships_rank is not None:
    impact = f"Olympic trials: {olympic_trials_rank:.1f}, US Championships: {us_championships_rank:.1f}"
else:
    impact = "Data insufficient for comparison."

print(f"Final Answer: {olympic_trials_rank:.1f}, {us_championships_rank:.1f}")