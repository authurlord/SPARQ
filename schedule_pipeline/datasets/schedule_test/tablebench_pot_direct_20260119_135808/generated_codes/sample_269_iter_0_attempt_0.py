import pandas as pd

df = pd.read_csv('table.csv')
# Filter rows for frequency 1 and 100k
l_at_1 = df[df['frequency (hz)'] == '1']['l (mh / km)'].values[0]
l_at_100k = df[df['frequency (hz)'] == '100k']['l (mh / km)'].values[0]

# Convert to float
l_at_1 = float(l_at_1)
l_at_100k = float(l_at_100k)

# Calculate percentage change
percentage_change = ((l_at_100k - l_at_1) / l_at_1) * 100
print(f"Final Answer: {percentage_change:.2f}")