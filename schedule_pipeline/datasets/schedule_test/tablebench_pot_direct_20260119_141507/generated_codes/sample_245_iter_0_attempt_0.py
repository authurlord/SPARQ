import pandas as pd

df = pd.read_csv('table.csv')
# Extract the 'Poland and Danzig' column (index 1) and filter years from 1934 to 1939
poland_danzig_values = df.iloc[0:6, 1].astype(float)  # Rows 0 to 5 (1934 to 1939)
increase = poland_danzig_values.iloc[5] - poland_danzig_values.iloc[0]
print(f"Final Answer: {increase:.1f}")