import pandas as pd

df = pd.read_csv('table.csv')
# Find the value of 'l (mh / km)' at frequency 1 Hz
l_at_1hz = df[df['frequency (hz)'] == '1']['l (mh / km)'].values[0]
# Find the value of 'l (mh / km)' at frequency 100k Hz
l_at_100k = df[df['frequency (hz)'] == '100k']['l (mh / km)'].values[0]

# Calculate percentage change
percentage_change = ((float(l_at_100k) - float(l_at_1hz)) / float(l_at_1hz)) * 100

print(f"Final Answer: {percentage_change:.2f}")