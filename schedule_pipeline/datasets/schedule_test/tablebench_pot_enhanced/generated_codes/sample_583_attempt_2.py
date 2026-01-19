import pandas as pd

df = pd.read_csv('table.csv')
# Extract passed students for 2005 and 2010
passed_2005 = int(df[df['year'] == '2005']['number of passed students'].iloc[0])
passed_2010 = int(df[df['year'] == '2010']['number of passed students'].iloc[0])

# Calculate percentage increase
percentage_increase = ((passed_2010 - passed_2005) / passed_2005) * 100
print(f"Final Answer: {percentage_increase:.2f}")