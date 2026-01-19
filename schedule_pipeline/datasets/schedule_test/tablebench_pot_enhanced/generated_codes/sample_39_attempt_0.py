import pandas as pd

df = pd.read_csv('table.csv')
# Sum the 'camp mates' column to get total number of camp mates
total_camp_mates = df['camp mates'].sum()
print(f"Final Answer: {total_camp_mates}")