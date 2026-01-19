import pandas as pd
import numpy as np

df = pd.read_csv('table.csv')
# Filter airlines that are part of 'star alliance'
star_alliance_fleet = df[df['alliance / association'] == 'star alliance']['passenger fleet'].astype(float)

# Calculate mean and standard deviation
mean_fleet = star_alliance_fleet.mean()
std_fleet = star_alliance_fleet.std()

print(f"Final Answer: {mean_fleet:.1f}, {std_fleet:.1f}")