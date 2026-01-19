import pandas as pd

df = pd.read_csv('table.csv')
# Convert 'bp comp 1 (˚c)' to numeric, handling strings with negative signs
bp_comp_1 = pd.to_numeric(df['bp comp 1 (˚c)'], errors='coerce')
max_bp = bp_comp_1.max()
min_bp = bp_comp_1.min()
difference = max_bp - min_bp
print(f"Final Answer: {difference:.1f}")