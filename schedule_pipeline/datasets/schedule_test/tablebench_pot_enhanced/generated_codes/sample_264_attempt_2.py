import pandas as pd

df = pd.read_csv('table.csv')
# Filter rows where the most spoken language is 'afrikaans'
afrikaans_places = df[df['most spoken language'] == 'afrikaans']
# Calculate total population for these places
total_population = afrikaans_places['population'].sum()
print(f"Final Answer: {total_population}")