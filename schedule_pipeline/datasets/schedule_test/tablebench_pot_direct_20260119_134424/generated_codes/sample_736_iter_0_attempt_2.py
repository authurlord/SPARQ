import pandas as pd
import numpy as np

df = pd.read_csv('table.csv')

# Filter data for Suzuki in 50cc and 125cc classes
suzuki_50cc = df[(df['team'] == 'suzuki') & (df['class'] == '50cc')]
suzuki_125cc = df[(df['team'] == 'suzuki') & (df['class'] == '125cc')]

# Calculate average points for each class
avg_50cc_points = suzuki_50cc['points'].mean()
avg_125cc_points = suzuki_125cc['points'].mean()

print(f"Final Answer: {avg_50cc_points:.1f}, {avg_125cc_points:.1f}")