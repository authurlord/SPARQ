import pandas as pd

df = pd.read_csv('table.csv')

# Extract student populations for karolinska institutet and swedish university of agricultural sciences
karolinska_pop = df[df['university'] == 'karolinska institutet']['student population ( fte , 2009)'].values[0]
sua_pop = df[df['university'] == 'swedish university of agricultural sciences']['student population ( fte , 2009)'].values[0]

# Calculate total original population
total_pop = karolinska_pop + sua_pop

# Increase by 18%
new_total_pop = total_pop * 1.18

print(f"Final Answer: {new_total_pop:.1f}")