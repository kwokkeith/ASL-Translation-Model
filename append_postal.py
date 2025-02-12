import requests
import pandas as pd
import extractPostal
import numpy as np
from multiprocessing import Pool, cpu_count
import time

# Define function to get postal code
def get_postal_code(row):
    street_name, block = row["street_name"], row["block"]
    address_query = f"{block} {street_name}"
    try:
        # Call API to get postal data
        res = extractPostal.pcode_to_data(address_query)

        # **Handle Empty API Response**
        if not res or not isinstance(res, list) or 'POSTAL' not in res[0]:
            print(f"Warning: No valid response for {block} {street_name}")
            return pd.NA, pd.NA, pd.NA  # Return NA for missing data

        return res[0].get('POSTAL', pd.NA), res[0].get('LATITUDE', pd.NA), res[0].get('LONGITUDE', pd.NA)

    except requests.exceptions.RequestException as e:
        print(f"Request Error for {block} {street_name}: {e}")
        return pd.NA, pd.NA, pd.NA  # Handle network issues safely

    except requests.exceptions.JSONDecodeError:
        print(f"JSON Decode Error for {block} {street_name} - API response empty or invalid.")
        return pd.NA, pd.NA, pd.NA  # Handle JSON errors safely

# **Multiprocessing Function for Parallel Processing**
def parallel_apply(df_chunk, num_cores=None):
    if num_cores is None:
        num_cores = cpu_count()  # Use all available CPU cores
    print(f"Using {num_cores} cores to process chunk.")

    # Convert DataFrame to a list of dictionaries (avoiding Pandas-specific pickling issues)
    data_records = df_chunk[['street_name', 'block']].to_dict('records')

    # Use multiprocessing Pool to process each row in parallel
    with Pool(num_cores) as pool:
        results = pool.map(get_postal_code, data_records)  # Process rows

    # Convert results back into separate columns
    postal_codes, latitudes, longitudes = zip(*results)

    # Return a DataFrame with computed results (keeping original indices)
    return pd.DataFrame({
        "postal_code": postal_codes,
        "latitude": latitudes,
        "longitude": longitudes
    }, index=df_chunk.index)

# **Helper function to process a chunk of data**
def process_chunk(rows):
    return [get_postal_code(row) for row in rows]

# **Process CSV in Chunks & Append Results at the Correct Position**
def process_csv_in_chunks(csv_path, output_path, chunksize=100):
    first_chunk = False  # Ensures headers are written only for the first chunk
    count = 1

    # Read the full CSV first (to preserve order in appending)
    full_df = pd.read_csv(csv_path)

    # **Ensure the required columns exist** (If missing, initialize with NA)
    for col in ['postal_code', 'latitude', 'longitude']:
        if col not in full_df.columns:
            full_df[col] = pd.NA

    for start_row in range(0, len(full_df), chunksize):
        start_time = time.perf_counter()
        chunk = full_df.iloc[start_row:start_row + chunksize].copy()  # Get chunk by row index
        print(f"Processing chunk {count} (rows {start_row} to {start_row + len(chunk) - 1})")

        # Select only rows where `postal_code`, `latitude`, or `longitude` is missing
        missing_rows = chunk[chunk[['postal_code', 'latitude', 'longitude']].isna().any(axis=1)]

        if not missing_rows.empty:  # Only process if there are missing values
            print(f"Processing {len(missing_rows)} missing rows in chunk {count}.")

            # Apply multiprocessing to get postal code, latitude, and longitude
            processed_data = parallel_apply(missing_rows)

            # Update only the modified rows in the chunk (preserving index position)
            chunk.loc[processed_data.index, ['postal_code', 'latitude', 'longitude']] = processed_data

            # Append only modified rows to the output file
            chunk.to_csv(output_path, mode='a', index=False, header=first_chunk)

            first_chunk = False  # Ensure only the first chunk has headers

        count += 1  # Increment chunk count
        end_time = time.perf_counter()
        diff_time = end_time - start_time
        minutes = int(diff_time // 60)  # Get minutes
        seconds = int(diff_time % 60)   # Get seconds
        milliseconds = int((diff_time % 1) * 1000)  # Convert remainder to ms

        print(f"{minutes} min {seconds} sec {milliseconds} ms")

if __name__ == '__main__':
    # Load CSV
    csv_paths = ['ResaleFlatPrices/Resale-Flat-Prices-1990_1999.csv', 
                'ResaleFlatPrices/Resale-Flat-Prices-2000Feb2012.csv',
                'ResaleFlatPrices/Resale-flat-prices-Jan-2017.csv',
                'ResaleFlatPrices/Resale-Flat-Prices-Jan2015toDec2016.csv',
                'ResaleFlatPrices/Resale-Flat-Prices-Mar_2012_Dec2014.csv']
    for i in range(1, 2):
        csv_path = csv_paths[i]
        df = pd.read_csv(csv_path)
        # df["postal_code"], df["latitude"], df["longitude"] = df.apply(get_postal_code, axis=1)
        output_path = f"ResaleFlatPrices/output{i}.csv"
        # df.to_csv(output_path)
        # df["postal_code"], df["latitude"], df["longitude"] = parallel_apply(df)
        # df.to_csv(f"ResaleFlatPrices/output{i}.csv")
        process_csv_in_chunks(csv_path, output_path, chunksize=10000)