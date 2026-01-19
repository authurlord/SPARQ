import pandas as pd

df = pd.read_csv('table.csv')
# Filter for 'la haute - côte - nord' RCM
la_haute_cote_nord = df[df['regional county municipality'] == 'la haute - côte - nord']
# Convert population to integer
la_haute_cote_nord['population'] = la_haute_cote_nord['population'].astype(int)
# Calculate average population
avg_population = la_haute_cote_nord['population'].mean()
# Find minimum population
min_population = la_haute_cote_nord['population'].min()
# Calculate how many people need to be added
people_needed = avg_population - min_population
print(f"Final Answer: {people_needed:.0f}")