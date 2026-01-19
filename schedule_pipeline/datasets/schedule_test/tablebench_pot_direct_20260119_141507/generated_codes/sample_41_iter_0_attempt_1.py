import pandas as pd

df = pd.read_csv('table.csv')
# Filter schools located in Belfast and sum their Outright Titles
total_outright_titles = df[df['Location'] == 'Belfast']['Outright Titles'].sum()
print(f"Final Answer: {total_outright_titles}")