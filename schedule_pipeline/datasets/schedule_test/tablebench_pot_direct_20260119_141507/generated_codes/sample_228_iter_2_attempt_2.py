import pandas as pd

df = pd.read_csv('table.csv')

# Extract relevant columns
s_ppm = df['s ( ppm )']
c_x = df['c_x ( metre )']
c_y = df['c_y (metre)']
c_z = df['c_z (metre)']

# Compute correlation coefficients
corr_cx = s_ppm.corr(c_x)
corr_cy = s_ppm.corr(c_y)
corr_cz = s_ppm.corr(c_z)

print(f"Correlation between s (ppm) and c_x: {corr_cx:.4f}")
print(f"Correlation between s (ppm) and c_y: {corr_cy:.4f}")
print(f"Correlation between s (ppm) and c_z: {corr_cz:.4f}")

# Final Answer: No
Final Answer: No