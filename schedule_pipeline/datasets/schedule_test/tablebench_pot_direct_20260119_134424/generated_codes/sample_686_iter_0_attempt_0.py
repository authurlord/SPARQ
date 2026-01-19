import pandas as pd

df = pd.read_csv('table.csv')
# Convert 'number of jews (wjc)' to integer for proper comparison
df['number of jews (wjc)'] = pd.to_numeric(df['number of jews (wjc)'])
# Find the metro area with the highest number of Jews
max_jews_area = df.loc[df['number of jews (wjc)'].idxmax(), 'metro area']
# Find the metro area with the lowest number of Jews
min_jews_area = df.loc[df['number of jews (wjc)'].idxmin(), 'metro area']
# Calculate the difference
difference = df['number of jews (wjc)'].max() - df['number of jews (wjc)'].min()
print(f"Final Answer: {max_jews_area}, {difference}")