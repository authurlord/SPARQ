# Given data
# Specimen 1: 0.1g → activity = 18 Bq → exposure = 0.0 mrem/hr (from table)
# Specimen 2: 10g → activity = 1834 Bq → exposure = 0.03 mrem/hr

# Since activity is proportional to weight, exposure should also scale with weight
# Scaling factor: 10g / 0.1g = 100

# Expected exposure for 0.1g specimen if scaled from 10g specimen:
expected_exposure_0_1g = 0.03 / 100  # 0.0003 mrem/hr

# Total exposure when combined:
total_exposure = expected_exposure_0_1g + 0.03

print(f"Final Answer: {total_exposure:.4f}")