import pandas as pd

df = pd.read_csv('table.csv')
# Get the number of seats in agricultural panel for Fianna Fáil and Fine Gael
fianna_fail_agri = df.loc[df['party'] == 'fianna fáil', 'agricultural panel'].values[0]
fine_gael_agri = df.loc[df['party'] == 'fine gael', 'agricultural panel'].values[0]
difference = fianna_fail_agri - fine_gael_agri
print(f"Final Answer: {difference}")