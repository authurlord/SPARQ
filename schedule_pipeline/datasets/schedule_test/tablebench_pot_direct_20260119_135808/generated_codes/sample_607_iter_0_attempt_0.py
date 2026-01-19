import pandas as pd

df = pd.read_csv('table.csv')
# Extract DC values for Verona and Venice
dc_verona = float(df[df['Province'] == 'Verona']['DC'].iloc[0])
dc_venice = float(df[df['Province'] == 'Venice']['DC'].iloc[0])
# Calculate the difference
difference = dc_verona - dc_venice
print(f"Final Answer: {difference:.1f}")