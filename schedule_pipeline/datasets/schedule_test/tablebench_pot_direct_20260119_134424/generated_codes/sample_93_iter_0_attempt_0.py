import pandas as pd

df = pd.read_csv('table.csv')
# Filter schools located in Belfast
belfast_schools = df[df['Location'] == 'Belfast']
# Find the school with the maximum outright titles
most_outright_titles_school = belfast_schools.loc[belfast_schools['Outright Titles'].idxmax()]['School']
print(f"Final Answer: {most_outright_titles_school}")