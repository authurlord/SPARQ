import pandas as pd

df = pd.read_csv('table.csv')
# Count mountains where 'location' contains 'austria'
austrian_mountains = df[df['location'].str.contains('austria', case=False, na=False)]
count_austria = len(austrian_mountains)
print(f"Final Answer: {count_austria}")