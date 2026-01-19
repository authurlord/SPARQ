import pandas as pd
import numpy as np

df = pd.read_csv('table.csv')

# Clean 'points for' column: remove spaces and convert to float
def clean_points_for(x):
    if isinstance(x, str):
        # Remove spaces and take the first number
        cleaned = x.replace(' ', '')
        # If cleaned is not empty, convert to float
        try:
            return float(cleaned)
        except ValueError:
            return np.nan
    else:
        return float(x)

df['points for'] = df['points for'].apply(clean_points_for)

# Calculate standard deviation of cleaned 'points for'
std_points_for = df['points for'].std()

print(f"Final Answer: {std_points_for:.1f}")