import pandas as pd
import numpy as np

df = pd.read_csv('table.csv')
# Convert 'Build date' column to integers
build_dates = pd.to_numeric(df['Build date'], errors='coerce')
# Calculate standard deviation, ignoring any invalid entries
std_build_dates = np.std(build_dates)
print(f"Final Answer: {std_build_dates:.1f}")