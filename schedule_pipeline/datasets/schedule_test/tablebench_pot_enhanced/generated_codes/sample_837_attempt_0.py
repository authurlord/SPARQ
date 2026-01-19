import pandas as pd

df = pd.read_csv('table.csv')

# Clean and convert 'cpu clock ( mhz )' to numeric
def extract_clock(speed):
    if isinstance(speed, str):
        # Split by space and take the first numeric part
        parts = speed.split()
        # Try to extract numbers from the string
        nums = [int(part.replace('(', '').replace(')', '')) for part in parts if part.replace('(', '').replace(')', '').isdigit()]
        if nums:
            return max(nums)
    return 0

df['cpu_clock_clean'] = df['cpu clock ( mhz )'].apply(extract_clock)

# Find the model with the highest CPU clock speed
max_clock_model = df.loc[df['cpu_clock_clean'].idxmax(), 'model']
print(f"Final Answer: {max_clock_model}")