import pandas as pd

df = pd.read_csv('table.csv')
# Find the row with frequency '1' and '100k'
l_1 = df[df['frequency (hz)'] == '1']['l (mh / km)'].values[0]
l_100k = df[df['frequency (hz)'] == '100k']['l (mh / km)'].values[0]

# Calculate percentage change
percentage_change = ((float(l_100k) - float(l_1)) / float(l_1)) * 100
print(f"Final Answer: {percentage_change:.2f}")