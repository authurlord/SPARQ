import pandas as pd

# Load the dataframe
df = pd.read_csv('table.csv')

# Extract the values of 'l (mh / km)' at frequency 1 Hz and 100k Hz
l_1_hz = df.loc[df['frequency (hz)'] == '1', 'l (mh / km)'].values[0]
l_100k_hz = df.loc[df['frequency (hz)'] == '100k', 'l (mh / km)'].values[0]

# Calculate percentage change
percentage_change = ((l_100k_hz - l_1_hz) / l_1_hz) * 100

print(f"Final Answer: {percentage_change:.2f}")