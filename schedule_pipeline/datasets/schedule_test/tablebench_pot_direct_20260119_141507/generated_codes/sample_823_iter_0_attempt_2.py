import pandas as pd

df = pd.read_csv('table.csv')
# Filter out the 'European Union' row to get only member states
df_member_states = df[df['member state'] != 'european union']
# Extract the 'pop density people / km 2' column
pop_density = df_member_states['pop density people / km 2'].astype(float)
# Calculate the median
median_density = pop_density.median()
print(f"Final Answer: {median_density:.1f}")