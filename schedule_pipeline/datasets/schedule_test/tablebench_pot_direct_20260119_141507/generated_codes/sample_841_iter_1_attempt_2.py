import pandas as pd

df = pd.read_csv('table.csv')
# Extract the agricultural panel values for Fianna Fáil and Fine Gael
fianna_fail_agri = int(df.loc[df['party'] == 'fianna fáil', 'agricultural panel'].values[0])
fine_gael_agri = int(df.loc[df['party'] == 'fine gael', 'agricultural panel'].values[0])
difference = fianna_fail_agri - fine_gael_agri
print(f"Final Answer: {difference}")