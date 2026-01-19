import pandas as pd

df = pd.read_csv('table.csv')
# Filter rows where 'most spoken language' is 'afrikaans'
afrikaans_population = df[df['most spoken language'] == 'afrikaans']['population'].sum()
print(f"Final Answer: {afrikaans_population}")