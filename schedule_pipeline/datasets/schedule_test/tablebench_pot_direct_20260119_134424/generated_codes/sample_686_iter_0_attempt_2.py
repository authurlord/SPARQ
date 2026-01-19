import pandas as pd

df = pd.read_csv('table.csv')
# Convert 'number of jews (wjc)' to integer
df['number of jews (wjc)'] = df['number of jews (wjc)'].astype(int)

# Find the metro area with the highest number of Jews (WJC)
max_jews_area = df.loc[df['number of jews (wjc)'].idxmax(), 'metro area']
min_jews_area = df.loc[df['number of jews (wjc)'].idxmin(), 'metro area']

# Calculate the difference
difference = df['number of jews (wjc)'].max() - df['number of jews (wjc)'].min()

print(f"Final Answer: {max_jews_area}, {difference}")