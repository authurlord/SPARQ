import pandas as pd

# Load the data
df = pd.read_csv('table.csv')

# Display the first few rows to understand the data
print("First few rows of the table:")
print(df.head())

# Describe the role of each column
print("\nColumn Roles:")
print("- year: The year the coin was issued.")
print("- design: The thematic design featured on the coin (e.g., Niagara Falls).")
print("- issue: The special feature or technique used in the coin's production (e.g., hologram, colorized).")
print("- artist: The artist responsible for the coin's design.")
print("- mintage: The number of coins produced.")
print("- issue price: The retail price at which the coin was issued.")

# Initial observations and trends
print("\nInitial Observations and Trends:")
print(f"- Total number of entries: {len(df)}")
print(f"- Years covered: {df['year'].unique()}")
print(f"- Most frequent design: {df['design'].value_counts().idxmax()} ({df['design'].value_counts().max()} occurrences)")
print(f"- Most frequent issue type: {df['issue'].value_counts().idxmax()} ({df['issue'].value_counts().max()} occurrences)")
print(f"- Most prolific artist: {df['artist'].value_counts().idxmax()} ({df['artist'].value_counts().max()} designs)")
print(f"- Highest mintage: {df['mintage'].max()} (coin: {df.loc[df['mintage'].idxmax(), 'design']})")
print(f"- Lowest mintage: {df['mintage'].min()} (coin: {df.loc[df['mintage'].idxmin(), 'design']})")
print(f"- Price range: ${df['issue price'].min()} - ${df['issue price'].max()}")

# Check if prices are consistent across similar designs or issues
print("\nPrice Analysis:")
print(f"- Average issue price: ${df['issue price'].mean():.2f}")
print(f"- Coins with price $79.95: {len(df[df['issue price'] == '79.95'])}")
print(f"- Coins with price $69.95: {len(df[df['issue price'] == '69.95'])}")

# Final summary
print("\nFinal Answer: year, design, issue, artist, mintage, issue price")