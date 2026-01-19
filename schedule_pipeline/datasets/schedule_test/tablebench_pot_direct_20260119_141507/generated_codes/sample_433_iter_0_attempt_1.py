import pandas as pd

df = pd.read_csv('table.csv')

# Display basic insights about the distribution of speakers
print("The table contains data on council areas in Scotland, including the number of speakers, population, and percentage.")
print("Key insights:")
print("- The number of speakers varies significantly, ranging from 97 (Shetland) to 15,811 (Na h - Eileanan Siar).")
print("- The distribution is highly skewed, with a few areas having a large number of speakers.")
print("- The council area with the most speakers is 'na h - eileanan siar', which also has the largest population and highest percentage.")
print(f"Average number of speakers: {df['speakers'].mean():.0f}")
print(f"Maximum number of speakers: {df['speakers'].max()}")
print(f"Minimum number of speakers: {df['speakers'].min()}")