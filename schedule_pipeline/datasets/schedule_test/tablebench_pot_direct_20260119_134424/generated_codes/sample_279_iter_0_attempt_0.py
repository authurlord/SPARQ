import pandas as pd

df = pd.read_csv('table.csv')

# Extract data for the two specimens
specimen_0_1g = df[df['specimen weight / size'] == '0.1 g / 4.08 mm']
specimen_10g = df[df['specimen weight / size'] == '10 g / 1.89 cm']

# Get exposure values
exposure_0_1g = float(specimen_0_1g['estimated exposure ( mrem ) / hr'].values[0])
exposure_10g = float(specimen_10g['estimated exposure ( mrem ) / hr'].values[0])

# Calculate exposure per gram for the 10g specimen
exposure_per_gram = exposure_10g / 10

# Calculate expected exposure for 0.1g specimen
expected_exposure_0_1g = exposure_per_gram * 0.1

# Total estimated exposure
total_exposure = exposure_10g + expected_exposure_0_1g

print(f"Final Answer: {total_exposure:.4f}")