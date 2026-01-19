import pandas as pd
import re

df = pd.read_csv('table.csv')

# Clean the 'cpu clock ( mhz )' column: extract the first number from values like "1000 / 1333 (turbo)" or "0 800"
def clean_clock_speed(x):
    if isinstance(x, str):
        # Remove any text after numbers or inside parentheses
        x = re.sub(r'[^0-9]', '', x)
        if x:
            return int(x)
    return 0

df['cpu clock ( mhz )'] = df['cpu clock ( mhz )'].apply(clean_clock_speed)

# Find the model with the highest CPU clock speed
max_clock_model = df.loc[df['cpu clock ( mhz )'].idxmax(), 'model']
print(f"Final Answer: {max_clock_model}")