import pandas as pd

df = pd.read_csv('table.csv')
# Calculate total lakes and reservoirs per borough
df['total_lakes_reservoirs'] = df['s lake and gnis query link'] + df['s reservoir and gnis query link']
# Find the borough with the highest total
max_borough = df.loc[df['total_lakes_reservoirs'].idxmax(), 'borough or census area']
print(f"Final Answer: {max_borough}")