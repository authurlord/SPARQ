import pandas as pd

df = pd.read_csv('table.csv')

# Filter delegates from Metro Manila (including "manila , metro manila")
df_metro = df[df['hometown'].str.contains('metro manila', case=False, na=False) | 
              df['hometown'].str.contains('manila', case=False, na=False)]

# Further filter for those with winning results
winning_results = ['winner', 'top 5 finalist', 'first runner - up / virreina', 'second runner - up']

# But only count "winner" as a win
df_winning = df_metro[df_metro['result'].str.contains('winner', case=False, na=False)]

count_winning_delegates = len(df_winning)

print(f"Final Answer: {count_winning_delegates}")