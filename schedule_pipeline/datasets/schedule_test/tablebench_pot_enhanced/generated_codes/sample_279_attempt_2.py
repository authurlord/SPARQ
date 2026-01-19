import pandas as pd

df = pd.read_csv('table.csv')

# Find the rows for the two specimens
specimen_1 = df[df['specimen weight / size'] == '0.1 g / 4.08 mm']
specimen_2 = df[df['specimen weight / size'] == '10 g / 1.89 cm']

# Extract estimated exposure values
exposure_1 = float(specimen_1['estimated exposure ( mrem ) / hr'].values[0])
exposure_2 = float(specimen_2['estimated exposure ( mrem ) / hr'].values[0])

# Total exposure
total_exposure = exposure_1 + exposure_2

print(f"Final Answer: {total_exposure}")