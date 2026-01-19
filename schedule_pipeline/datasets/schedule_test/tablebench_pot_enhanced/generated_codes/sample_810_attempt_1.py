import pandas as pd

df = pd.read_csv('table.csv')
# Filter rows where alliance is 'star alliance'
star_alliance_df = df[df['alliance / association'] == 'star alliance']
# Convert 'passenger fleet' to numeric
star_alliance_df['passenger fleet'] = pd.to_numeric(star_alliance_df['passenger fleet'])
# Calculate mean and standard deviation
mean_fleet = star_alliance_df['passenger fleet'].mean()
std_fleet = star_alliance_df['passenger fleet'].std()
print(f"Final Answer: {mean_fleet:.1f}, {std_fleet:.1f}")