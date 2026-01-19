# Extract data from the table
data = [
    ['United States', '500', '618', '893', '77', '1,725'],
    ['Canada', '82', '169', '171', '2', '366'],
    ['Australia', '119', '45', '67', '1', '176'],
    ['Ecuador', '9', '9', '253', '0', '253'],
    ['United Kingdom', '14', '13', '125', '13', '132'],
    ['India', '63', '21', '108', '0', '128'],
    ['Mexico', '55', '35', '66', '1', '106'],
    ['China', '37', '27', '83', '1', '101'],
    ['Spain', '11', '14', '93', '0', '95'],
    ['Germany', '23', '11', '67', '2', '71'],
    ['Hong Kong', '32', '9', '29', '1', '56'],
    ['Turkey', '25', '10', '43', '0', '60'],
    ['Argentina', '7', '3', '56', '0', '57'],
    ['Switzerland', '18', '11', '42', '1', '49'],
    ['Indonesia', '32', '14', '29', '0', '48'],
    ['Poland', '6', '8', '40', '0', '45'],
]

# Convert string values to integers, handling commas if necessary
primary_schools = [int(row[1].replace(',', '')) for row in data]

# Get top 3 countries by Primary schools
top_3_primary = sorted(primary_schools, reverse=True)[:3]
total_top_3_primary = sum(top_3_primary)

# Total Career-related schools from "Total Schools Globally" row
career_related_total = int('118')

# Calculate difference
difference = total_top_3_primary - career_related_total

print(f"Final Answer: {difference}")