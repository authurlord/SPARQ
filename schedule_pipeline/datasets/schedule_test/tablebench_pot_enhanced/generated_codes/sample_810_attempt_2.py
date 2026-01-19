import pandas as pd

df = pd.read_csv('table.csv')
# Filter rows where alliance is 'star alliance'
star_alliance_airlines = df[df['alliance / association'] == 'star alliance']
# Extract passenger fleet column and convert to numeric
passenger_fleet = pd.to_numeric(star_alliance_airlines['passenger fleet'], errors='coerce')
# Calculate mean and standard deviation
mean_fleet = passenger_fleet.mean()
std_fleet = passenger_fleet.std()
print(f"Final Answer: {mean_fleet:.1f}, {std_fleet:.1f}")