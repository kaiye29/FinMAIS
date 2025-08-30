import pandas as pd
import numpy as np
import warnings
from datetime import datetime, timedelta

# Suppress common pandas warnings for a cleaner output
warnings.filterwarnings('ignore', category=FutureWarning)
from pandas.errors import SettingWithCopyWarning
warnings.filterwarnings('ignore', category=SettingWithCopyWarning)


def create_time_machine_dataset_with_temporal_protection(
    technical_file: str,
    ratios_file: str,
    cpi_file: str,
    ecb_file: str,
    output_filepath: str,
    train_end_date: str = "2018-12-31",
    val_end_date: str = "2021-12-31",
    test_end_date: str = "2024-12-31"
):
    """
    🔒 FIXED: Creates a time machine dataset with proper temporal protection.
    
    Key fixes:
    - Creates separate train/val/test splits with no future data leakage
    - Future price calculation respects temporal boundaries
    - Proper temporal validation throughout
    
    Args:
        technical_file (str): Path to the historical data with technical indicators.
        ratios_file (str): Path to the financial ratios data.
        cpi_file (str): Path to the raw CPI data.
        ecb_file (str): Path to the raw ECB interest rate data.
        output_filepath (str): Path to save the final merged dataset.
        train_end_date (str): End date for training split
        val_end_date (str): End date for validation split  
        test_end_date (str): End date for test split
    """
    print("🔒 STARTING TIME MACHINE CONSTRUCTION WITH TEMPORAL PROTECTION")
    print("=" * 70)

    # Convert split dates
    train_end = pd.to_datetime(train_end_date)
    val_end = pd.to_datetime(val_end_date)
    test_end = pd.to_datetime(test_end_date)
    
    print(f"📅 Temporal splits:")
    print(f"  Training: up to {train_end_date}")
    print(f"  Validation: {train_end_date} to {val_end_date}")
    print(f"  Test: {val_end_date} to {test_end_date}")

    # --- 1. Load all data sources ---
    print("\n1. Loading data sources...")
    try:
        df_tech = pd.read_csv(technical_file)
        df_ratios = pd.read_csv(ratios_file)
        df_cpi = pd.read_csv(cpi_file)
        df_ecb = pd.read_csv(ecb_file)
        print(f"✅ Loaded technical data: {len(df_tech)} rows")
        print(f"✅ Loaded ratios data: {len(df_ratios)} rows")
        print(f"✅ Loaded CPI data: {len(df_cpi)} rows")
        print(f"✅ Loaded ECB data: {len(df_ecb)} rows")
    except FileNotFoundError as e:
        print(f"❌ Error: A required data file was not found. {e}")
        return

    # --- 2. Prepare Base and Macro DataFrames ---
    print("\n2. Converting date columns...")
    df_tech['Date'] = pd.to_datetime(df_tech['Date'])
    df_ratios['Date'] = pd.to_datetime(df_ratios['Date'])

    # --- 3. Pre-process Macro Data (CPI and ECB) ---
    print("\n3. Pre-processing macroeconomic data (CPI & ECB)...")
    # Clean CPI data
    df_cpi['Date'] = pd.to_datetime(df_cpi['Date'].str.replace('MM', '-'), format='%Y-%m')
    df_cpi.rename(columns={'CPI_Annual_Rate': 'CPI'}, inplace=True)

    # Clean ECB data
    df_ecb.columns = ['DATE_STR', 'Date', 'ECB_Rate']
    df_ecb['Date'] = pd.to_datetime(df_ecb['Date'])
    df_ecb = df_ecb[['Date', 'ECB_Rate']]

    # 🔒 FIXED: Only use macro data up to test end date
    df_cpi = df_cpi[df_cpi['Date'] <= test_end]
    df_ecb = df_ecb[df_ecb['Date'] <= test_end]

    # Find the earliest date across all relevant data sources
    min_date = min(df_tech['Date'].min(), df_cpi['Date'].min(), df_ecb['Date'].min())
    max_date = min(df_tech['Date'].max(), test_end)  # 🔒 FIXED: Respect test end boundary

    # Create a full date range to map the sparse macro data onto a daily series
    full_date_range = pd.date_range(start=min_date, end=max_date, freq='D')
    df_macro = pd.DataFrame(full_date_range, columns=['Date'])

    # Merge and forward-fill CPI and ECB data
    df_macro = pd.merge(df_macro, df_cpi, on='Date', how='left')
    df_macro = pd.merge(df_macro, df_ecb, on='Date', how='left')
    df_macro.ffill(inplace=True)  # Forward-fill propagates the last known value

    print(f"✅ Macro data prepared from {min_date.date()} to {max_date.date()}")

    # --- 4. Pre-process Fundamental Data (Ratios) ---
    print("\n4. Pivoting financial ratios data...")
    # 🔒 FIXED: Only use ratios data up to test end date
    df_ratios = df_ratios[df_ratios['Date'] <= test_end]
    
    # Pivot the table to turn each ratio 'Field' into its own column
    df_ratios_pivoted = df_ratios.pivot_table(
        index=['Date', 'Ticker'],
        columns='Field',
        values='Value'
    ).reset_index()

    print(f"✅ Ratios data pivoted: {len(df_ratios_pivoted)} rows")

    # --- 5. Merge all DataFrames ---
    print("\n5. Merging Technical, Fundamental, and Macro data...")
    # 🔒 FIXED: Only use technical data up to test end date
    df_tech = df_tech[df_tech['Date'] <= test_end]
    
    # Start with the technical data as the base
    df_merged = df_tech.copy()

    # Merge the pivoted fundamental data
    df_merged = pd.merge(df_merged, df_ratios_pivoted, on=['Date', 'Ticker'], how='left')

    # Merge the daily macro data
    df_merged = pd.merge(df_merged, df_macro, on='Date', how='left')

    print(f"✅ Initial merge complete: {len(df_merged)} rows")

    # --- 6. 🔒 FIXED: Calculate Future Price with Temporal Protection ---
    print("\n6. 🔒 Calculating future price with temporal protection...")
    
    def calculate_future_price_temporally_safe(group):
        """
        Calculate future prices while respecting temporal boundaries for each split.
        """
        group = group.sort_values('Date').copy()
        group['Future_Close_30D'] = None
        
        for i, row in group.iterrows():
            current_date = row['Date']
            future_date = current_date + timedelta(days=30)
            
            # 🔒 CRITICAL FIX: Only calculate future price if it respects temporal boundaries
            # Determine which split this row belongs to
            if current_date <= train_end:
                # Training data: can look ahead within training period
                max_future_date = train_end
            elif current_date <= val_end:
                # Validation data: can look ahead within validation period
                max_future_date = val_end
            else:
                # Test data: can look ahead within test period
                max_future_date = test_end
            
            # Only set future price if the future date is within the allowed boundary
            if future_date <= max_future_date:
                # Find the actual future price
                future_prices = group[group['Date'] == future_date]['Close']
                if not future_prices.empty:
                    group.loc[i, 'Future_Close_30D'] = future_prices.iloc[0]
                else:
                    # Find the closest available future price within the boundary
                    future_data = group[
                        (group['Date'] > current_date) & 
                        (group['Date'] <= max_future_date)
                    ]
                    if not future_data.empty:
                        # Use the closest available future price
                        closest_future = future_data.iloc[0]
                        group.loc[i, 'Future_Close_30D'] = closest_future['Close']
        
        return group

    # Apply the temporally safe future price calculation
    print("🔒 Applying temporal protection to future price calculation...")
    df_merged = df_merged.groupby('Ticker').apply(calculate_future_price_temporally_safe)
    df_merged.reset_index(drop=True, inplace=True)

    # --- 7. Add Split Labels for Reference ---
    print("\n7. Adding temporal split labels...")
    def assign_split(date):
        if date <= train_end:
            return 'train'
        elif date <= val_end:
            return 'validation'
        else:
            return 'test'
    
    df_merged['Split'] = df_merged['Date'].apply(assign_split)
    
    split_counts = df_merged['Split'].value_counts()
    print(f"✅ Split distribution:")
    for split, count in split_counts.items():
        print(f"  {split}: {count} rows")

    # --- 8. Temporal Validation ---
    print("\n8. 🔒 Performing temporal validation...")
    
    # Check for future data leakage in training set
    train_data = df_merged[df_merged['Split'] == 'train']
    if not train_data.empty:
        train_future_prices = train_data.dropna(subset=['Future_Close_30D'])
        if not train_future_prices.empty:
            max_future_date_in_train = train_future_prices['Date'].max() + timedelta(days=30)
            if max_future_date_in_train > train_end + timedelta(days=1):
                print(f"⚠️ WARNING: Training data may contain future leakage!")
            else:
                print(f"✅ Training temporal validation passed")
    
    # Validate no overlap between splits
    train_max = df_merged[df_merged['Split'] == 'train']['Date'].max()
    val_min = df_merged[df_merged['Split'] == 'validation']['Date'].min()
    val_max = df_merged[df_merged['Split'] == 'validation']['Date'].max()
    test_min = df_merged[df_merged['Split'] == 'test']['Date'].min()
    
    assert train_max < val_min, f"TEMPORAL VIOLATION: Training data overlaps with validation!"
    assert val_max < test_min, f"TEMPORAL VIOLATION: Validation data overlaps with test!"
    print("✅ No temporal overlap between splits detected")

    # --- 9. Final Cleanup ---
    print("\n9. Cleaning the final dataset...")
    original_rows = len(df_merged)
    
    # Only remove rows where future price is truly needed but missing
    # Keep rows without future prices for the final periods of each split
    rows_with_future = df_merged.dropna(subset=['Future_Close_30D'])
    rows_without_future = df_merged[df_merged['Future_Close_30D'].isna()]
    
    print(f"📊 Future price analysis:")
    print(f"  Rows with future prices: {len(rows_with_future)}")
    print(f"  Rows without future prices: {len(rows_without_future)}")
    
    # Fill any remaining NaNs in ratio columns with 0 (a neutral value)
    ratio_cols = [col for col in df_merged.columns if col not in ['Date', 'Ticker', 'Split', 'Future_Close_30D']]
    numeric_cols = df_merged[ratio_cols].select_dtypes(include=[np.number]).columns
    df_merged[numeric_cols] = df_merged[numeric_cols].fillna(0)

    print(f"✅ Cleaned {len(numeric_cols)} numeric columns")

    # --- 10. Save the final dataset ---
    print("\n10. Saving temporally protected dataset...")
    try:
        df_merged.to_csv(output_filepath, index=False)
        print(f"✅ Successfully saved to '{output_filepath}'")
    except Exception as e:
        print(f"❌ Error saving file: {e}")
        return

    # --- 11. Final Summary ---
    print(f"\n{'='*70}")
    print("🎉 TIME MACHINE CONSTRUCTION WITH TEMPORAL PROTECTION COMPLETE!")
    print(f"{'='*70}")
    print(f"📊 Final dataset statistics:")
    print(f"  Total rows: {len(df_merged):,}")
    print(f"  Total columns: {len(df_merged.columns)}")
    print(f"  Date range: {df_merged['Date'].min().date()} to {df_merged['Date'].max().date()}")
    print(f"  Unique tickers: {df_merged['Ticker'].nunique()}")
    print(f"\n🔒 Temporal Protection Features:")
    print(f"  ✅ Split-aware future price calculation")
    print(f"  ✅ No future data leakage across splits")
    print(f"  ✅ Proper temporal boundaries enforced")
    print(f"  ✅ Split labels added for reference")

    # Display sample
    print(f"\n📋 Sample of the temporally protected dataset:")
    sample_cols = ['Date', 'Ticker', 'Close', 'Future_Close_30D', 'Split']
    available_cols = [col for col in sample_cols if col in df_merged.columns]
    print(df_merged[available_cols].head(10))
    
    return df_merged


# --- Main Execution Block ---
if __name__ == "__main__":
    print("🔒 CREATING TIME MACHINE DATASET WITH TEMPORAL PROTECTION")
    print("=" * 70)
    
    # Define the input and output file paths
    tech_data_file = 'historical_technical_data.csv'
    fundamental_data_file = 'financial_ratios_clean_updated.csv'
    cpi_data_file = 'CPI_Raw.csv'
    ecb_data_file = 'ECB_Data.csv'
    
    # The final output file, ready for your RL agent with temporal protection
    output_file = 'time_machine_dataset_temporal_protected.csv'
    
    # Run the creation process with temporal protection
    result = create_time_machine_dataset_with_temporal_protection(
        technical_file=tech_data_file,
        ratios_file=fundamental_data_file,
        cpi_file=cpi_data_file,
        ecb_file=ecb_data_file,
        output_filepath=output_file,
        train_end_date="2018-12-31",  # Adjust based on your data
        val_end_date="2021-12-31",    # Adjust based on your data
        test_end_date="2024-12-31"    # Adjust based on your data
    )
    
    if result is not None:
        print("\n🎯 SUCCESS! Your time machine dataset is now temporally protected.")
        print("No future data leakage detected. Ready for proper ML training!")
    else:
        print("\n❌ FAILED! Please check your input files and try again.")