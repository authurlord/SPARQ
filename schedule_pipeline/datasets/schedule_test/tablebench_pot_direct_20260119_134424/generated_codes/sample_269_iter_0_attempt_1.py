import pandas as pd

df = pd.read_csv('table.csv')
# Filter rows for frequency 1 and 100k
row_1 = df[df['frequency (hz)'] == '1']
row_100k = df[df['frequency (hz)'] == '100k']

# Extract l values
l_1 = float(row_1['l (mh / km)'].values[0])
l_100k = float(row_100k['l (mh / km)'].values[0])

# Calculate percentage change
percentage_change = ((l_100k - l_1) / l_1) * 100
print(f"Final Answer: {percentage_change:.2f}")