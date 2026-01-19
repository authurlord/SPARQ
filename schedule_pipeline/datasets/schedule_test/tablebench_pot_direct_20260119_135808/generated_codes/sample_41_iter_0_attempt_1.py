import pandas as pd

df = pd.read_csv('table.csv')
# Filter schools located in Belfast
belfast_schools = df[df['Location'] == 'Belfast']
# Sum the 'Outright Titles' for Belfast schools
total_outright_titles_belfast = belfast_schools['Outright Titles'].sum()
print(f"Final Answer: {total_outright_titles_belfast}")