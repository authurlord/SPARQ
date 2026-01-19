import pandas as pd

df = pd.read_csv('table.csv')
# Extract student populations for karolinska institutet and swedish university of agricultural sciences
karolinska_population = df[df['university'] == 'karolinska institutet']['student population ( fte , 2009)'].values[0]
swedish_uni_population = df[df['university'] == 'swedish university of agricultural sciences']['student population ( fte , 2009)'].values[0]

# Increase each by 18%
new_karolinska = karolinska_population * 1.18
new_swedish = swedish_uni_population * 1.18

# Calculate combined new student population
new_combined_population = new_karolinska + new_swedish
print(f"Final Answer: {new_combined_population:.0f}")