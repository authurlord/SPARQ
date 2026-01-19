import pandas as pd

df = pd.read_csv('table.csv')

# Extract data for the two specimens
specimen_01g = df[df['specimen weight / size'] == '0.1 g / 4.08 mm']
specimen_10g = df[df['specimen weight / size'] == '10 g / 1.89 cm']

# Get their estimated exposures
exposure_01g = float(specimen_01g['estimated exposure ( mrem ) / hr'].values[0])
exposure_10g = float(specimen_10g['estimated exposure ( mrem ) / hr'].values[0])

# Since activity is proportional to weight, scale exposure of 0.1g specimen by 100
scaled_exposure_01g = exposure_01g * (10 / 0.1)

# Total estimated exposure
total_exposure = scaled_exposure_01g + exposure_10g

print(f"Final Answer: {total_exposure:.2f}")