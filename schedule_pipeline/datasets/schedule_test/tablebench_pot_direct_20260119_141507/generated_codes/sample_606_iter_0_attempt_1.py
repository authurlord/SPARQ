import pandas as pd

df = pd.read_csv('table.csv')

# Extract student populations for Karolinska Institutet and Swedish University of Agricultural Sciences
karolinska_population = df[df['university'] == 'karolinska institutet']['student population ( fte , 2009)'].values[0]
swedish_agri_population = df[df['university'] == 'swedish university of agricultural sciences']['student population ( fte , 2009)'].values[0]

# Calculate current combined population
combined_population = karolinska_population + swedish_agri_population

# Increase by 18%
new_combined_population = combined_population * 1.18

print(f"Final Answer: {new_combined_population:.0f}")