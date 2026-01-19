import pandas as pd

df = pd.read_csv('table.csv')

# Find the row for 0.1 g / 4.08 mm (specimen 1)
row_01g = df[df['specimen weight / size'] == '0.1 g / 4.08 mm']
# Find the row for 10 g / 1.89 cm (specimen 2)
row_10g = df[df['specimen weight / size'] == '10 g / 1.89 cm']

# Extract activities (Bq) and exposures (mrem/hr)
activity_01g = int(row_01g['calculated activity ( bq )'].values[0])
activity_10g = int(row_10g['calculated activity ( bq )'].values[0])
exposure_10g = float(row_10g['estimated exposure ( mrem ) / hr'].values[0])

# Calculate exposure for 0.1g specimen using proportionality
exposure_01g = (activity_01g / activity_10g) * exposure_10g

# Total exposure
total_exposure = exposure_01g + exposure_10g

print(f"Final Answer: {total_exposure:.4f}")