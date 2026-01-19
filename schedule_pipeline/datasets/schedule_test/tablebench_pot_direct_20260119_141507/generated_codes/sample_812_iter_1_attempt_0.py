import pandas as pd
import numpy as np

df = pd.read_csv('table.csv')

# Clean 'points for' column: remove any non-numeric parts and convert to float
def clean_points_for(x):
    if isinstance(x, str):
        # Split by space and take the first number
        parts = x.split()
        # If there are multiple numbers, take the first one
        for part in parts:
            if part.isdigit():
                return float(part)
        return np.nan
    else:
        return float(x)

df['points for'] = df['points for'].apply(clean_points_for)

# Calculate standard deviation of the cleaned 'points for' column
std_points_for = df['points for'].std()

print(f"Final Answer: {std_points_for:.1f}")