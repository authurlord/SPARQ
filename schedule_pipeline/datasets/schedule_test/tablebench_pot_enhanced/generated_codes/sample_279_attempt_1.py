# Given data
weight_01g = 0.1
exposure_01g = 0.0

weight_10g = 10
exposure_10g = 0.03

# Proportional exposure for 0.1g specimen based on 10g specimen
proportional_exposure_01g = (weight_01g / weight_10g) * exposure_10g

# Total estimated exposure
total_exposure = proportional_exposure_01g + exposure_01g

print(f"Final Answer: {total_exposure:.4f}")