import pandas as pd

df = pd.read_csv('table.csv')
# Filter out the 'European Union' row to focus on member states
df_member_states = df[df['member state'] != 'european union']
# Extract the population density column and calculate median
median_density = df_member_states['pop density people / km 2'].median()
print(f"Final Answer: {median_density:.1f}")