import pandas as pd
import numpy as np

df = pd.read_csv('table.csv')

# Extract boiling point and critical temperature values
boiling_point = df.loc[df['physical property'] == 'boiling point (degree)', ['helium', 'neon', 'argon', 'krypton', 'xenon']].values[0]
critical_temperature = df.loc[df['physical property'] == 'critical temperature (k)', ['helium', 'neon', 'argon', 'krypton', 'xenon']].values[0]

# Convert to float arrays
boiling_point = [float(x) for x in boiling_point]
critical_temperature = [float(x) for x in critical_temperature]

# Compute correlation coefficient
correlation = np.corrcoef(boiling_point, critical_temperature)[0, 1]
print(f"Final Answer: {correlation:.3f}")