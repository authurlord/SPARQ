import pandas as pd

df = pd.read_csv('table.csv')
# Filter the two universities
karolinska = df[df['university'] == 'karolinska institutet']['student population ( fte , 2009)'].values[0]
swedish_ag = df[df['university'] == 'swedish university of agricultural sciences']['student population ( fte , 2009)'].values[0]

# Convert to integers and calculate combined population
combined_population = int(karolinska) + int(swedish_ag)

# Apply 18% increase
new_combined_population = combined_population * 1.18

print(f"Final Answer: {int(new_combined_population)}")