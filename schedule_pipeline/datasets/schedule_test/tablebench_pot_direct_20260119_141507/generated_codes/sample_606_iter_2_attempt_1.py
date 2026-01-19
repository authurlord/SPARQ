import pandas as pd

df = pd.read_csv('table.csv')

# Find the student population for karolinska institutet and swedish university of agricultural sciences
karolinska_pop = df[df['university'] == 'karolinska institutet']['student population ( fte , 2009)'].values[0]
swedish_agri_pop = df[df['university'] == 'swedish university of agricultural sciences']['student population ( fte , 2009)'].values[0]

# Increase each by 18%
new_karolinska_pop = karolinska_pop * 1.18
new_swedish_agri_pop = swedish_agri_pop * 1.18

# Calculate the new combined student population
new_combined_population = new_karolinska_pop + new_swedish_agri_pop

print(f"Final Answer: {new_combined_population:.0f}")