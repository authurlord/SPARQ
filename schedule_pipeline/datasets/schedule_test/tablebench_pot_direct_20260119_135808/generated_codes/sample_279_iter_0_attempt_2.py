import pandas as pd

df = pd.read_csv('table.csv')

# Extract the exposure values for the two specimens
exposure_0_1g = df[df['specimen weight / size'] == '0.1 g / 4.08 mm']['estimated exposure ( mrem ) / hr'].values[0]
exposure_10g = df[df['specimen weight / size'] == '10 g / 1.89 cm']['estimated exposure ( mrem ) / hr'].values[0]

# Since activity is proportional to weight, scale exposure accordingly
# 0.1g specimen scaled to 10g equivalent: (0.1 / 10) * exposure of 10g
expected_exposure_0_1g = (0.1 / 10) * float(exposure_10g)

# Total exposure
total_exposure = expected_exposure_0_1g + float(exposure_10g)

print(f"Final Answer: {total_exposure:.4f}")