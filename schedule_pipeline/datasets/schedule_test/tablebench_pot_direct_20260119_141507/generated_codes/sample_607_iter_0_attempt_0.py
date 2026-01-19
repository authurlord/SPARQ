import pandas as pd

df = pd.read_csv('table.csv')
# Get DC values for Verona and Venice
dc_verona = df.loc[df['Province'] == 'Verona', 'DC'].values[0]
dc_venice = df.loc[df['Province'] == 'Venice', 'DC'].values[0]
difference = dc_verona - dc_venice
print(f"Final Answer: {difference}")