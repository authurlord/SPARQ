import pandas as pd

df = pd.read_csv('table.csv')

# Clean the 'cpu clock ( mhz )' column by extracting numeric values from strings
def extract_clock_speed(speed_str):
    if isinstance(speed_str, str):
        # Extract numbers from string, e.g., "1000 / 1333 (turbo)" -> 1333
        import re
        numbers = [int(x) for x in re.findall(r'\d+', speed_str)]
        return max(numbers) if numbers else 0
    return 0

df['cpu clock ( mhz )'] = df['cpu clock ( mhz )'].apply(extract_clock_speed)

# Find the model with the highest CPU clock speed
max_clock_model = df.loc[df['cpu clock ( mhz )'].idxmax(), 'model']
print(f"Final Answer: {max_clock_model}")