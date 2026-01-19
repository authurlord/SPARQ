import pandas as pd

df = pd.read_csv('table.csv')

# Clean and convert 'number of passed students' to numeric by removing '%'
df['number of passed students'] = df['number of passed students'].str.replace('%', '').astype(float)

# Extract values for 2005 and 2010
passed_2005 = df.loc[df['year'] == '2005', 'number of passed students'].values[0]
passed_2010 = df.loc[df['year'] == '2010', 'number of passed students'].values[0]

# Calculate percentage increase
percentage_increase = ((passed_2010 - passed_2005) / passed_2005) * 100

print(f"Final Answer: {percentage_increase:.2f}")