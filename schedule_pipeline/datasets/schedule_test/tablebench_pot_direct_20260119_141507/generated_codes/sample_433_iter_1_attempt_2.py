import pandas as pd

# Load the data
df = pd.read_csv('table.csv')

# Display basic summary of speakers
print("Summary of speakers:")
print(f"Mean speakers: {df['speakers'].mean():.0f}")
print(f"Median speakers: {df['speakers'].median():.0f}")
print(f"Max speakers: {df['speakers'].max()}")
print(f"Min speakers: {df['speakers'].min()}")

# Sort by speakers in descending order to highlight top areas
df_sorted = df.sort_values(by='speakers', ascending=False)

# Show top 5 and bottom 5 areas
print("\nTop 5 council areas by speakers:")
print(df_sorted.head(5)[['council area', 'speakers', 'population']])
print("\nBottom 5 council areas by speakers:")
print(df_sorted.tail(5)[['council area', 'speakers', 'population']])

# Optional: Create a bar chart of speakers per council area
# We'll only plot the top 10 for clarity
top_10 = df_sorted.head(10)
top_10.plot(x='council area', y='speakers', kind='bar', figsize=(12, 6), title='Speakers by Council Area (Top 10)')
plt.xlabel('Council Area')
plt.ylabel('Number of Speakers')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

Final Answer: Mean speakers: 3026, Median speakers: 988, Max speakers: 15811, Min speakers: 97, Top area: na h - eileanan siar, Bottom area: shetland