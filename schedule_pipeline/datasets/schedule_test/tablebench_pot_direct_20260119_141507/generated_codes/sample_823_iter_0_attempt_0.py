import pandas as pd

df = pd.read_csv('table.csv')
# Filter out the 'European Union' row and keep only member states
df_member_states = df[df['member state'] != 'european union']
# Extract the population density column and compute median
median_density = df_member_states['pop density people / km 2'].median()
print(f"Final Answer: {median_density:.1f}")