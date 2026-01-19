import pandas as pd

df = pd.read_csv('table.csv')

# Extract student populations for karolinska institutet and swedish university of agricultural sciences
karolinska_population = df[df['university'] == 'karolinska institutet']['student population ( fte , 2009)'].values[0]
swedish_agri_population = df[df['university'] == 'swedish university of agricultural sciences']['student population ( fte , 2009)'].values[0]

# Calculate total original population
total_population = karolinska_population + swedish_agri_population

# Increase by 18%
new_population = total_population * 1.18

print(f"Final Answer: {new_population:.0f}")