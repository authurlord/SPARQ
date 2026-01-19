import pandas as pd

df = pd.read_csv('table.csv')
# Filter rows where 'alliance / association' is 'star alliance'
star_alliance_fleet = df[df['alliance / association'] == 'star alliance']['passenger fleet']
# Calculate mean and standard deviation
mean_fleet = star_alliance_fleet.mean()
std_fleet = star_alliance_fleet.std()
print(f"Final Answer: {mean_fleet:.1f}, {std_fleet:.1f}")