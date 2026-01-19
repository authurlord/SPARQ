import pandas as pd

df = pd.read_csv('table.csv')

# Clean the 'cpu clock ( mhz )' column by extracting numeric values
def extract_clock_speed(clock_str):
    if isinstance(clock_str, str):
        # Handle cases like "1000 / 1333 (turbo)" or "0 800"
        parts = clock_str.split('/')
        if len(parts) > 1:
            return int(parts[0].strip().replace(' ', ''))
        else:
            # Remove any non-digit characters
            cleaned = ''.join(filter(str.isdigit, clock_str))
            return int(cleaned)
    return 0

df['cpu clock ( mhz )'] = df['cpu clock ( mhz )'].apply(extract_clock_speed)

# Find the model with the highest CPU clock speed
max_clock_model = df.loc[df['cpu clock ( mhz )'].idxmax(), 'model']
print(f"Final Answer: {max_clock_model}")