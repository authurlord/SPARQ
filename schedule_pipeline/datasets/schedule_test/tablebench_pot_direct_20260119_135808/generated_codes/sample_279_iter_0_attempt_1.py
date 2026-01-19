import pandas as pd

df = pd.read_csv('table.csv')

# Find the exposure for 0.1 g specimen (row 5)
exposure_0_1g = df.loc[4, 'estimated exposure ( mrem ) / hr']  # 0.0

# Find the exposure for 10 g specimen (row 2)
exposure_10g = df.loc[2, 'estimated exposure ( mrem ) / hr']  # 0.03

# Since activity is proportional to weight, scale exposure of 0.1g specimen
weight_ratio = 0.1 / 10
scaled_exposure_0_1g = weight_ratio * exposure_10g

# Total exposure
total_exposure = scaled_exposure_0_1g + exposure_10g

print(f"Final Answer: {total_exposure:.4f}")