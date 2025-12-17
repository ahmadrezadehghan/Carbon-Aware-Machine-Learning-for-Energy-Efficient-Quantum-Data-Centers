import os
import csv

# Output file name
output_file = "out.txt"

# Get all CSV files in the current directory
csv_files = [f for f in os.listdir('.') if f.endswith('.csv')]

with open(output_file, 'w', encoding='utf-8') as outfile:
    for csv_file in csv_files:
        outfile.write(f"--- Contents of {csv_file} ---\n")

        with open(csv_file, 'r', encoding='utf-8') as infile:
            reader = csv.reader(infile)
            for row in reader:
                # Join each CSV row with commas and write to output file
                outfile.write(', '.join(row) + '\n')

        outfile.write('\n')  # Blank line between files

print(f"Combined contents of {len(csv_files)} CSV files written to {output_file}")
