import pandas as pd

df = pd.read_csv('table.csv')
# Filter schools located in Belfast
belfast_schools = df[df['Location'] == 'Belfast']
# Find the school with the maximum outright titles
max_outright_title_school = belfast_schools.loc[belfast_schools['Outright Titles'].idxmax()]
print(f"Final Answer: {max_outright_title_school['School']}")