import pandas as pd
from pathlib import Path


def merge_crop_and_rainfall(crop_file, rainfall_file, output_file):
    """
    Merge crop data with seasonal rainfall data.
    
    Args:
        crop_file (str): Path to crop data (Excel/CSV)
        rainfall_file (str): Path to seasonal rainfall data (Excel/CSV)
        output_file (str): Path for merged output file
    """
    print("\n" + "="*60)
    print("Merging Crop and Rainfall Data")
    print("="*60 + "\n")
    
    print(f"Reading crop data: {crop_file}")
    if crop_file.endswith('.xlsx') or crop_file.endswith('.xls'):
        df_crop = pd.read_excel(crop_file)
    else:
        df_crop = pd.read_csv(crop_file)
    print(f"Crop data: {df_crop.shape[0]} rows, {df_crop.shape[1]} columns")
    print(f"Columns: {list(df_crop.columns)}")
    
    print(f"\nReading rainfall data: {rainfall_file}")
    if rainfall_file.endswith('.xlsx') or rainfall_file.endswith('.xls'):
        df_rainfall = pd.read_excel(rainfall_file)
    else:
        df_rainfall = pd.read_csv(rainfall_file)
    print(f"Rainfall data: {df_rainfall.shape[0]} rows, {df_rainfall.shape[1]} columns")
    print(f"Columns: {list(df_rainfall.columns)}")
    
    df_crop['District_Clean'] = df_crop['District'].str.strip().str.title()
    df_rainfall['District_Clean'] = df_rainfall['District'].str.strip().str.title()
    
    df_crop['Year'] = df_crop['Year'].astype(str)
    df_rainfall['Year'] = df_rainfall['Year'].astype(str)
    
    print("\n" + "="*60)
    print("Performing merge on: District, Year, Season")
    print("="*60)
    
    df_merged = pd.merge(
        df_crop,
        df_rainfall,
        left_on=['District_Clean', 'Year', 'Season'],
        right_on=['District_Clean', 'Year', 'Season'],
        how='left',
        suffixes=('', '_rainfall')
    )
    
    if 'District_rainfall' in df_merged.columns:
        df_merged = df_merged.drop(columns=['District_rainfall'])
    
    if 'District_Clean' in df_merged.columns:
        df_merged = df_merged.drop(columns=['District_Clean'])
    
    print(f"\nMerged data: {df_merged.shape[0]} rows, {df_merged.shape[1]} columns")
    print(f"Columns: {list(df_merged.columns)}")
    
    matched = df_merged[df_merged['Total_Rainfall'].notna()].shape[0] if 'Total_Rainfall' in df_merged.columns else 0
    unmatched = df_merged.shape[0] - matched
    print(f"\nMatched records: {matched}")
    print(f"Unmatched records: {unmatched}")
    
    print(f"\nExporting merged data to: {output_file}")
    if output_file.endswith('.xlsx') or output_file.endswith('.xls'):
        df_merged.to_excel(output_file, index=False, engine='openpyxl')
    else:
        df_merged.to_csv(output_file, index=False)
    
    print(f"✓ Export complete! {len(df_merged)} rows written.")
    
    print("\n" + "="*60)
    print("Sample of merged data:")
    print("="*60)
    print(df_merged.head(10))
    
    return df_merged


def main():
    """
    Main function to merge crop and rainfall datasets.
    """
    crop_file = r"D:\AgriSet Project\Data\output\telangana_crop_data_cleaned.xlsx"
    rainfall_file = r"D:\AgriSet Project\Data\output\Rainfall_Seasonal_Aggregated.xlsx"
    output_file = r"D:\AgriSet Project\Data\output\Telangana_Crop_Rainfall_Merged.xlsx"
    output_csv = r"D:\AgriSet Project\Data\output\Telangana_Crop_Rainfall_Merged.csv"
    
    df_merged = merge_crop_and_rainfall(crop_file, rainfall_file, output_file)
    
    print(f"\nAlso saving as CSV: {output_csv}")
    df_merged.to_csv(output_csv, index=False)
    print(f"✓ CSV export complete!")
    
    print("\n" + "="*60)
    print("MERGE COMPLETE!")
    print("="*60)
    print(f"\nOutput files:")
    print(f"  - Excel: {output_file}")
    print(f"  - CSV: {output_csv}")
    print(f"\nYou now have {len(df_merged)} records with both crop and weather data!")


if __name__ == "__main__":
    main()
