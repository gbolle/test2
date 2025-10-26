import pandas as pd
import numpy as np
import re
from pathlib import Path


class CropDataExtractor:
    """
    Extract and transform Telangana crop statistics from horizontal year-vertical crop Excel format
    into clean tabular format.
    """
    
    def __init__(self, input_file):
        """
        Initialize the extractor with input file path.
        
        Args:
            input_file (str): Path to the Excel file
        """
        self.input_file = input_file
        self.df_raw = None
        self.df_transformed = None
        self.year_columns = {}
        
    def read_excel(self):
        """
        Read Excel file with multi-level headers and process them.
        """
        print(f"Reading Excel file: {self.input_file}")
        
        file_extension = Path(self.input_file).suffix.lower()
        
        if file_extension == '.xlsx':
            df_header0 = pd.read_excel(self.input_file, header=0, engine='openpyxl')
        elif file_extension == '.xls':
            df_header0 = pd.read_excel(self.input_file, header=0, engine='xlrd')
        else:
            raise ValueError(f"Unsupported file format: {file_extension}")
        
        year_pattern_simple = re.compile(r'(\d{4})\s*[-–]\s*\d{4}')
        col_mapping = {}
        current_year = None
        metric_counter = 0
        
        for i, col in enumerate(df_header0.columns):
            col_str = str(col)
            
            match = year_pattern_simple.search(col_str)
            if match:
                current_year = match.group(1)
                metric_counter = 0
                col_mapping[i] = f"{current_year}_Area"
                
                if current_year not in self.year_columns:
                    self.year_columns[current_year] = {}
                self.year_columns[current_year]['area'] = f"{current_year}_Area"
            elif 'Unnamed' in col_str and current_year:
                metric_counter += 1
                if metric_counter == 1:
                    col_mapping[i] = f"{current_year}_Production"
                    self.year_columns[current_year]['production'] = f"{current_year}_Production"
                elif metric_counter == 2:
                    col_mapping[i] = f"{current_year}_Yield"
                    self.year_columns[current_year]['yield'] = f"{current_year}_Yield"
                else:
                    col_mapping[i] = col_str
            elif 'State' in col_str or 'Crop' in col_str or 'District' in col_str:
                col_mapping[i] = 'District_Raw'
            elif 'Season' in col_str:
                col_mapping[i] = 'Season'
            else:
                col_mapping[i] = col_str
        
        final_columns = [col_mapping.get(i, str(df_header0.columns[i])) for i in range(len(df_header0.columns))]
        
        engine = 'xlrd' if file_extension == '.xls' else 'openpyxl'
        self.df_raw = pd.read_excel(self.input_file, header=2, engine=engine)
        self.df_raw.columns = final_columns[:len(self.df_raw.columns)]
        
        print(f"Loaded {len(self.df_raw)} rows and {len(self.df_raw.columns)} columns")
        print(f"Found {len(self.year_columns)} year columns: {sorted(self.year_columns.keys())}")
        
        return self.df_raw
    
    def detect_crop_headers(self):
        """
        Detect crop headers by checking for rows with District_Raw but no Season.
        
        Returns:
            pd.Series: Boolean series indicating crop header rows
        """
        print("Detecting crop headers...")
        
        has_district_raw = self.df_raw['District_Raw'].notna()
        no_season = self.df_raw['Season'].isna()
        
        is_crop_header = has_district_raw & no_season
        
        pattern = re.compile(r'^\d+\.\s*[A-Za-z]')
        matches_pattern = self.df_raw['District_Raw'].astype(str).str.match(pattern, na=False)
        
        is_crop_header = is_crop_header & matches_pattern
        
        print(f"Found {is_crop_header.sum()} crop header rows")
        return is_crop_header
    
    def extract_crop_name(self, header_text):
        """
        Extract clean crop name from header text.
        
        Args:
            header_text (str): Raw header text like '1. Arhar/Tur'
            
        Returns:
            str: Clean crop name
        """
        header_text = str(header_text).strip()
        
        match = re.match(r'^\d+\.\s*(.+)$', header_text)
        if match:
            crop_name = match.group(1).strip()
            return crop_name
        
        return header_text
    
    def clean_district_name(self, district_text):
        """
        Clean district name by removing number prefix.
        
        Args:
            district_text (str): Raw district like '1. Adilabad'
            
        Returns:
            str: Clean district name like 'Adilabad'
        """
        if pd.isna(district_text):
            return None
        
        district_text = str(district_text).strip()
        
        match = re.match(r'^\d+\.\s*(.+)$', district_text)
        if match:
            return match.group(1).strip()
        
        return district_text
    
    def process_raw_data(self):
        """
        Process raw data: forward-fill crops and districts, clean names.
        
        Returns:
            pd.DataFrame: Processed data
        """
        print("Processing raw data...")
        
        df_proc = self.df_raw.copy()
        
        is_crop_header = self.detect_crop_headers()
        
        crop_list = []
        current_crop = None
        for idx in df_proc.index:
            if is_crop_header.loc[idx]:
                current_crop = self.extract_crop_name(df_proc.loc[idx, 'District_Raw'])
                crop_list.append(None)
            else:
                crop_list.append(current_crop)
        
        df_proc['Crop'] = crop_list
        
        has_district = df_proc['District_Raw'].notna() & ~is_crop_header
        district_list = []
        current_district = None
        for idx in df_proc.index:
            if has_district.loc[idx]:
                current_district = df_proc.loc[idx, 'District_Raw']
            if is_crop_header.loc[idx]:
                district_list.append(None)
            else:
                district_list.append(current_district)
        
        df_proc['District'] = district_list
        df_proc['District'] = df_proc['District'].apply(self.clean_district_name)
        
        df_proc = df_proc[~is_crop_header]
        
        print(f"Processed {len(df_proc)} data rows")
        return df_proc
    
    def reshape_to_long_format(self, df_proc):
        """
        Reshape data from wide format (years as columns) to long format.
        
        Args:
            df_proc (pd.DataFrame): Processed data
            
        Returns:
            pd.DataFrame: Reshaped data
        """
        print("Reshaping data to long format...")
        
        records = []
        
        for idx, row in df_proc.iterrows():
            district = row['District']
            season = row['Season']
            crop = row['Crop']
            
            for year, cols in self.year_columns.items():
                area = row.get(cols.get('area'), None)
                production = row.get(cols.get('production'), None)
                yield_val = row.get(cols.get('yield'), None)
                
                records.append({
                    'District': district,
                    'Season': season,
                    'Crop': crop,
                    'Year': year,
                    'Area': area,
                    'Production': production,
                    'Yield': yield_val
                })
        
        df_long = pd.DataFrame(records)
        print(f"Reshaped to {len(df_long)} rows")
        
        return df_long
    
    def clean_data(self, df_long):
        """
        Clean the transformed data: remove totals, filter invalid rows.
        
        Args:
            df_long (pd.DataFrame): Long format data
            
        Returns:
            pd.DataFrame: Cleaned data
        """
        print("Cleaning data...")
        
        df_clean = df_long.copy()
        
        df_clean = df_clean[df_clean['Season'].str.lower() != 'total']
        
        df_clean = df_clean[df_clean['District'].notna()]
        df_clean = df_clean[df_clean['Crop'].notna()]
        df_clean = df_clean[df_clean['Season'].notna()]
        
        df_clean['District'] = df_clean['District'].astype(str)
        df_clean = df_clean[df_clean['District'] != 'nan']
        df_clean = df_clean[df_clean['District'] != 'None']
        df_clean = df_clean[~df_clean['District'].str.contains('Telangana', case=False, na=False)]
        
        numeric_cols = ['Area', 'Production', 'Yield']
        for col in numeric_cols:
            df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce')
        
        df_clean = df_clean.dropna(subset=['Area', 'Production', 'Yield'], how='all')
        
        df_clean = df_clean.reset_index(drop=True)
        
        print(f"Cleaned data: {len(df_clean)} rows remaining")
        return df_clean
    
    def add_state_column(self, df_clean, state_name='Telangana'):
        """
        Add state column to the dataframe.
        
        Args:
            df_clean (pd.DataFrame): Cleaned data
            state_name (str): State name (default: 'Telangana')
            
        Returns:
            pd.DataFrame: Data with state column
        """
        df_clean.insert(0, 'State', state_name)
        return df_clean
    
    def transform(self, state_name='Telangana'):
        """
        Execute the complete transformation pipeline.
        
        Args:
            state_name (str): State name (default: 'Telangana')
            
        Returns:
            pd.DataFrame: Transformed data
        """
        print("\n" + "="*60)
        print("Starting data transformation pipeline")
        print("="*60 + "\n")
        
        self.read_excel()
        
        df_proc = self.process_raw_data()
        
        df_long = self.reshape_to_long_format(df_proc)
        
        df_clean = self.clean_data(df_long)
        
        df_final = self.add_state_column(df_clean, state_name)
        
        column_order = ['State', 'District', 'Season', 'Crop', 'Year', 'Area', 'Production', 'Yield']
        df_final = df_final[column_order]
        
        self.df_transformed = df_final
        
        print("\n" + "="*60)
        print("Transformation complete!")
        print("="*60)
        print(f"\nFinal dataset shape: {df_final.shape}")
        print(f"Columns: {list(df_final.columns)}")
        print(f"\nFirst few rows:")
        print(df_final.head(15))
        print(f"\nLast few rows:")
        print(df_final.tail(10))
        
        return df_final
    
    def export_to_excel(self, output_file):
        """
        Export transformed data to Excel file.
        
        Args:
            output_file (str): Path to output Excel file
        """
        if self.df_transformed is None:
            raise ValueError("No transformed data to export. Run transform() first.")
        
        print(f"\nExporting to: {output_file}")
        self.df_transformed.to_excel(output_file, index=False, engine='openpyxl')
        print(f"Export complete! {len(self.df_transformed)} rows written.")
    
    def export_to_csv(self, output_file):
        """
        Export transformed data to CSV file.
        
        Args:
            output_file (str): Path to output CSV file
        """
        if self.df_transformed is None:
            raise ValueError("No transformed data to export. Run transform() first.")
        
        print(f"\nExporting to CSV: {output_file}")
        self.df_transformed.to_csv(output_file, index=False)
        print(f"Export complete! {len(self.df_transformed)} rows written.")
    
    def get_summary_statistics(self):
        """
        Generate summary statistics for the transformed data.
        
        Returns:
            dict: Summary statistics
        """
        if self.df_transformed is None:
            raise ValueError("No transformed data. Run transform() first.")
        
        stats = {
            'total_records': len(self.df_transformed),
            'unique_crops': self.df_transformed['Crop'].nunique(),
            'unique_districts': self.df_transformed['District'].nunique(),
            'unique_years': self.df_transformed['Year'].nunique(),
            'crops': sorted(self.df_transformed['Crop'].dropna().unique().tolist()),
            'districts': sorted(self.df_transformed['District'].dropna().unique().tolist()),
            'years': sorted(self.df_transformed['Year'].dropna().unique().tolist()),
            'seasons': sorted(self.df_transformed['Season'].dropna().unique().tolist())
        }
        
        return stats


def main():
    """
    Main function to demonstrate usage.
    """
    # UPDATE THIS PATH TO YOUR FILE LOCATION
    input_file = r"D:\AgriSet Project\Data\horizontal_year_vertical_crop_report (1).xlsx"
    output_excel = r"D:\AgriSet Project\Data\output\telangana_crop_data_cleaned.xlsx"
    output_csv = r"D:\AgriSet Project\Data\output\telangana_crop_data_cleaned.csv"

    # Create output directory
    Path(r"D:\AgriSet Project\Data\output").mkdir(exist_ok=True)

    extractor = CropDataExtractor(input_file)

    df_transformed = extractor.transform(state_name='Telangana')

    extractor.export_to_excel(output_excel)
    extractor.export_to_csv(output_csv)

    stats = extractor.get_summary_statistics()
    print("\n" + "="*60)
    print("SUMMARY STATISTICS")
    print("="*60)
    print(f"Total records: {stats['total_records']}")
    print(f"Unique crops: {stats['unique_crops']}")
    print(f"Unique districts: {stats['unique_districts']}")
    print(f"Years covered: {', '.join(stats['years'])}")
    print(f"\nCrops in dataset:")
    for i, crop in enumerate(stats['crops'], 1):
        print(f"  {i}. {crop}")
    print(f"\nDistricts ({stats['unique_districts']} total):")
    for i, district in enumerate(stats['districts'][:10], 1):
        print(f"  {i}. {district}")
    if len(stats['districts']) > 10:
        print(f"  ... and {len(stats['districts']) - 10} more")
    print(f"\nSeasons: {', '.join(stats['seasons'])}")

    print(f"\n✓ Output files created:")
    print(f"  - Excel: {output_excel}")
    print(f"  - CSV: {output_csv}")


if __name__ == "__main__":
    main()