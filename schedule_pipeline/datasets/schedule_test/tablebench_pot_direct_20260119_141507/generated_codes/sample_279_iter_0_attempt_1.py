import pandas as pd

df = pd.read_csv('table.csv')
# Find the estimated exposure for the two specimens
exposure_01g = df[df['specimen weight / size'] == '0.1 g / 4.08 mm']['estimated exposure ( mrem ) / hr'].values[0]
exposure_10g = df[df['specimen weight / size'] == '10 g / 1.89 cm']['estimated exposure ( mrem ) / hr'].values[0]

total_exposure = exposure_01g + exposure_10g
print(f"Final Answer: {total_exposure}")