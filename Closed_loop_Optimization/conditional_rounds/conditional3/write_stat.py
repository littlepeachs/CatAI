import pandas as pd

def calculate_yield_stats(csv_path="filtered_label_data.csv"):
    """
    Calculate the mean and variance of the 'Yield' field in the specified CSV file.
    
    Args:
        csv_path (str): Path to the CSV file containing the Yield data.
        
    Returns:
        tuple: Mean and variance of the Yield values.
    """
    try:
        # Read the CSV file
        df = pd.read_csv(csv_path)
        
        # Check if 'Yield' column exists
        if 'YIELD' in df.columns:
            yield_mean = df['YIELD'].mean()
            yield_variance = df['YIELD'].std()
            
            print(f"Yield Statistics:")
            print(f"Mean: {yield_mean:.4f}")
            print(f"Variance: {yield_variance:.4f}")
            
            return yield_mean, yield_variance
        else:
            print("Error: 'Yield' column not found in the CSV file.")
            return None, None
            
    except FileNotFoundError:
        print(f"Error: CSV file not found at {csv_path}")
        return None, None
    except Exception as e:
        print(f"Error occurred while processing the CSV: {str(e)}")
        return None, None

if __name__ == "__main__":
    calculate_yield_stats()
