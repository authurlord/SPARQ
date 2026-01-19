import pandas as pd

df = pd.read_csv('table.csv')

# Find the rows corresponding to the two specimens
specimen_01 = df[df['specimen weight / size'] == '0.1 g / 4.08 mm']
specimen_10 = df[df['specimen weight / size'] == '10 g / 1.89 cm']

# Extract the estimated exposure for each
exposure_01 = specimen_01['estimated exposure ( mrem ) / hr'].values[0]
exposure_10 = specimen_10['estimated exposure ( mrem ) / hr'].values[0]

# Total exposure
total_exposure = exposure_01 + exposure_10
print(f"Final Answer: {total_exposure}")