import pandas as pd

df = pd.read_csv('table.csv')
# Find the chambering with the highest p max (bar)
max_pressure_row = df.loc[df['p max ( bar )'].idxmax()]
highest_pressure_chambering = max_pressure_row['chambering']
print(f"Final Answer: {highest_pressure_chambering}")