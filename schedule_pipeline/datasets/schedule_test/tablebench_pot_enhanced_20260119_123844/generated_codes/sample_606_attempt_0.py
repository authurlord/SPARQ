import pandas as pd

df = pd.read_csv('table.csv')
# Filter rows for Karolinska Institutet and Swedish University of Agricultural Sciences
karolinska_pop = df[df['university'] == 'karolinska institutet']['student population ( fte , 2009)'].values[0]
swedish_agricultural_pop = df[df['university'] == 'swedish university of agricultural sciences']['student population ( fte , 2009)'].values[0]

# Convert to integers and calculate combined population with 18% increase
combined_pop = int(karolinska_pop) + int(swedish_agricultural_pop)
new_combined_pop = combined_pop * 1.18

print(f"Final Answer: {int(new_combined_pop)}")