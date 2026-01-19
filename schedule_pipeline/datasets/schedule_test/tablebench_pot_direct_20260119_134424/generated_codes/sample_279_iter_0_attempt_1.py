import pandas as pd

df = pd.read_csv('table.csv')

# Find the row for 0.1 g / 4.08 mm
row_01g = df[df['specimen weight / size'] == '0.1 g / 4.08 mm']
# Find the row for 10 g / 1.89 cm
row_10g = df[df['specimen weight / size'] == '10 g / 1.89 cm']

# Extract activity (Bq) and exposure (mrem/hr)
activity_01g = float(row_01g['calculated activity ( bq )'].values[0])
activity_10g = float(row_10g['calculated activity ( bq )'].values[0])
exposure_10g = float(row_10g['estimated exposure ( mrem ) / hr'].values[0])

# Since activity is proportional to weight, and exposure is proportional to activity,
# we can compute the expected exposure for 0.1g based on 10g
weight_ratio = 0.1 / 10
expected_exposure_01g = weight_ratio * exposure_10g

# Total exposure
total_exposure = expected_exposure_01g + exposure_10g

print(f"Final Answer: {total_exposure:.4f}")