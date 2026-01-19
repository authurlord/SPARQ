import pandas as pd

df = pd.read_csv('table.csv')
# Convert 'p max' column to numeric
df['p max ( bar )'] = pd.to_numeric(df['p max ( bar )'])
# Find the chambering with the highest p max
max_pressure_chambering = df.loc[df['p max ( bar )'].idxmax(), 'chambering']
print(f"Final Answer: {max_pressure_chambering}")