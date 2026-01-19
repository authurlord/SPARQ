import pandas as pd

df = pd.read_csv('table.csv')
# Filter for 'la haute - côte - nord' region
la_haute_cote_nord = df[df['regional county municipality'] == 'la haute - côte - nord']
# Convert population to numeric
la_haute_cote_nord['population'] = pd.to_numeric(la_haute_cote_nord['population'])
# Find minimum population
min_population = la_haute_cote_nord['population'].min()
# Calculate average population
avg_population = la_haute_cote_nord['population'].mean()
# Calculate the difference
people_needed = avg_population - min_population
print(f"Final Answer: {people_needed:.0f}")