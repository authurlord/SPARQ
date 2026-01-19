import pandas as pd

df = pd.read_csv('table.csv')
# Extract values for Poland and Danzig in 1934 and 1939
poland_1934 = float(df.loc[df['-'] == '1934', 'Poland and Danzig'].iloc[0])
poland_1939 = float(df.loc[df['-'] == '1939', 'Poland and Danzig'].iloc[0])
# Calculate increase
increase = poland_1939 - poland_1934
print(f"Final Answer: {increase:.1f}")