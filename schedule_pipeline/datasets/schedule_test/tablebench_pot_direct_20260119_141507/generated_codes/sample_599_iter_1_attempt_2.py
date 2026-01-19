import pandas as pd
import numpy as np

df = pd.read_csv('table.csv')

# Filter rows for year 1944
year_1944 = df[df['Year'] == '1944']

# Extract numeric US Chart position by removing text like '(R&B)'
def extract_position(pos):
    # Remove any text in parentheses
    cleaned = pos.split(' (')[0]
    return float(cleaned) if cleaned.isdigit() else np.nan

# Apply the function and compute mean
positions = year_1944['US Chart position'].apply(extract_position)
average_position = positions.mean()

print(f"Final Answer: {average_position:.1f}")