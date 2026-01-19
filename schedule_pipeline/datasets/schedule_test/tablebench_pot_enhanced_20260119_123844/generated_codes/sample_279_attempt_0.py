import pandas as pd

df = pd.read_csv('table.csv')

# Find the row with 0.1 g / 4.08 mm
row_01g = df[df['specimen weight / size'] == '0.1 g / 4.08 mm']
# Find the row with 10 g / 1.89 cm
row_10g = df[df['specimen weight / size'] == '10 g / 1.89 cm']

# Extract activity (Bq) and exposure (mrem/hr)
activity_01g = float(row_01g['calculated activity ( bq )'].values[0])
exposure_01g = float(row_01g['estimated exposure ( mrem ) / hr'].values[0])

activity_10g = float(row_10g['calculated activity ( bq )'].values[0])
exposure_10g = float(row_10g['estimated exposure ( mrem ) / hr'].values[0])

# Use proportionality to compute expected exposure for 0.1g based on activity
# Since activity is proportional to weight, exposure should be proportional to activity
expected_exposure_01g = (activity_01g / activity_10g) * exposure_10g

# Total estimated exposure
total_exposure = expected_exposure_01g + exposure_10g

print(f"Final Answer: {total_exposure:.4f}")