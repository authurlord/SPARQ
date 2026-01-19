import pandas as pd

df = pd.read_csv('table.csv')
# Filter data for 'la haute - côte - nord' regional county municipality
la_haute_cote_nord = df[df['regional county municipality'] == 'la haute - côte - nord']
# Convert population to integer
la_haute_cote_nord['population'] = la_haute_cote_nord['population'].astype(int)
# Calculate average population in the region
avg_population = la_haute_cote_nord['population'].mean()
# Find the smallest population
min_population = la_haute_cote_nord['population'].min()
# Calculate the number of people needed
people_needed = avg_population - min_population
print(f"Final Answer: {int(people_needed)}")