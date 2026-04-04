import pandas as pd

def select_rows_below_threshold(csv_path="label_data.csv", threshold=50, output_path="filtered_label_data.csv"):
    """
    Select rows from the CSV file where the 'YIELD' field is less than the specified threshold
    and save them to a new CSV file.
    
    Args:
        csv_path (str): Path to the input CSV file containing the Yield data.
        threshold (float): Threshold value for filtering YIELD (default is 45).
        output_path (str): Path to save the filtered CSV file.
    """
    try:
        # Read the CSV file
        df = pd.read_csv(csv_path)
        
        # Check if 'YIELD' column exists
        if 'YIELD' in df.columns:
            # Filter rows where YIELD is less than the threshold
            filtered_df = df[df['YIELD'] < threshold]
            
            # Save the filtered dataframe to a new CSV file
            filtered_df.to_csv(output_path, index=False)
            
            print(f"Filtered data saved to {output_path}")
            print(f"Original row count: {len(df)}")
            print(f"Filtered row count: {len(filtered_df)}")
        else:
            print("Error: 'YIELD' column not found in the CSV file.")
            
    except FileNotFoundError:
        print(f"Error: CSV file not found at {csv_path}")
    except Exception as e:
        print(f"Error occurred while processing the CSV: {str(e)}")

if __name__ == "__main__":
    select_rows_below_threshold()
