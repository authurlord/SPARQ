import pandas as pd

# Load the data
df = pd.read_csv('table.csv')

# Convert string values to float for numerical analysis
df['bp comp 1 (˚c)'] = pd.to_numeric(df['bp comp 1 (˚c)'], errors='coerce')
df['bp comp 2 (˚c)'] = pd.to_numeric(df['bp comp 2 (˚c)'], errors='coerce')
df['bp azeo (˚c)'] = pd.to_numeric(df['bp azeo (˚c)'], errors='coerce')

# Summary of notable patterns
print("Main components and their properties:")
print("- 'Diethyl ether' (bp ~34.6°C) frequently appears and forms azeotropes with other compounds.")
print("- Azeotropic boiling points are often lower than individual components, indicating low-boiling azeotropes.")
print("- In mixtures like carbon disulfide + diethyl ether, carbon disulfide dominates (up to 99% wt).")
print("- Weight percentages vary widely; high % wt comp 1 indicates dominance of the first component.")
print("- Some mixtures (e.g., chloroform + methyl ethyl ketone) show azeotrope at ~79.9°C, close to both components.")
print("- Components with similar boiling points (e.g., chloroform and methyl ethyl ketone) often form azeotropes.")
print("Notable trend: Mixtures with a significant difference in boiling points tend to form azeotropes near the lower boiling component.")

Final Answer: diethyl ether, azeotrope, boiling point, weight percentage