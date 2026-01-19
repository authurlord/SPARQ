import pandas as pd

df = pd.read_csv('table.csv')
# Filter the two universities
karolinska_pop = int(df[df['university'] == 'karolinska institutet']['student population ( fte , 2009)'].values[0])
swedish_pop = int(df[df['university'] == 'swedish university of agricultural sciences']['student population ( fte , 2009)'].values[0])
# Calculate combined population and increase by 18%
combined_pop = (karolinska_pop + swedish_pop) * 1.18
print(f"Final Answer: {combined_pop:.0f}")