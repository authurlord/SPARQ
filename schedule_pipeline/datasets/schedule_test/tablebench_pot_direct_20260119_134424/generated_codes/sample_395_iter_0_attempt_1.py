import pandas as pd
df = pd.read_csv('table.csv')
df = df.replace('n / a', pd.NA)
comprehension_cols = ['comprehension of danish', 'comprehension of swedish', 'comprehension of norwegian']
df[comprehension_cols] = df[comprehension_cols].astype(float)
avg_comprehension = df[comprehension_cols].mean()
highest_avg_city = df.loc[df['average'].idxmax(), 'city']
lowest_avg_city = df.loc[df['average'].idxmin(), 'city']
print(f"Average comprehension of Danish: {avg_comprehension['comprehension of danish']:.2f}")
print(f"Average comprehension of Swedish: {avg_comprehension['comprehension of swedish']:.2f}")
print(f"Average comprehension of Norwegian: {avg_comprehension['comprehension of norwegian']:.2f}")
print(f"Highest average comprehension: {highest_avg_city}")
print(f"Lowest average comprehension: {lowest_avg_city}")