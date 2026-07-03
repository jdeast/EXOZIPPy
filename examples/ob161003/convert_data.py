import pandas as pd
from datetime import datetime, timedelta
import math

# Constants for JD conversion
JD_OFFSET = 2450000
JD0 = 2451545 # JD of 2000-01-01

def convert_jd_to_date(jd):
    """Converts a JD to a YYYYMMDD string."""
    # Since the JD is HJD (days since 2450000), we add the offset
    # and convert to a datetime object starting from Jan 1, 2000
    date_obj = datetime(2000, 1, 1, 12) + timedelta(days=jd-JD0)
    return date_obj.strftime("%Y%m%d")

def process_file(input_filename):
    # Load the data - assuming whitespace delimited, adjust as needed
    # Using header=None because the provided format implies columns
    df = pd.read_csv(input_filename, sep='\s+', header=None,skiprows=24,
                     names=['Telescope', 'Filter', 'HJD_offset', 'mag', 'err'],
                     comment='#')

    # Calculate actual HJD
    df['HJD'] = df['HJD_offset'] + JD_OFFSET

    # Group by Telescope and Filter
    grouped = df.groupby(['Telescope', 'Filter'])

    for (telescope, filter_name), group in grouped:
        # Get date from the first data point
        first_hjd = group['HJD'].iloc[0]
        #date_str = convert_jd_to_date(math.floor(first_hjd))
        date_str = convert_jd_to_date(first_hjd)

        # Format filename: nYYYYMMDD.FILTER.TELESCOPE.EVENT.dat
        filename = f"n{date_str}.{filter_name}.{telescope}.OGLE-2016-BLG-1003.dat"

        print((filename, first_hjd, date_str))

        
        # Prepare output data
        output = group[['HJD', 'mag', 'err']].copy()

        # Write to file
        with open(filename, 'w') as f:
            f.write("# HJD mag err\n")
            for _, row in output.iterrows():
                f.write(f"{row['HJD']:.6f} {row['mag']:.3f} {row['err']:.3f}\n")

        print(f"Generated: {filename}")


if __name__ == "__main__":
    process_file('dbf1.txt')
