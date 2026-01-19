import pandas as pd

df = pd.read_csv('table.csv')
# Count the number of mountains located in Austria
austria_mountains_count = df[df['location'].str.contains('austria', case=False, na=False)].shape[0]
print(f"Final Answer: {austria_mountains_count}")