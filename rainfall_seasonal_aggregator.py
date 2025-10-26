import pandas as pd
import numpy as np
from pathlib import Path


class RainfallSeasonalAggregator:
    """
    Aggregate daily rainfall and weather data into seasonal summaries.
    """
    
    def __init__(self, input_file):
        """
        Initialize with input CSV file path.
        
        Args:
            input_file (str): Path to the rainfall CSV file
        """
        self.input_file = input_file
        self.df_raw = None
        self.df_seasonal = None
        
    def read_data(self):
        """
        Read the rainfall CSV file.
        """
        print(f"Reading rainfall data: {self.input_file}")
        self.df_raw = pd.read_csv(self.input_file)
        print(f"Loaded {len(self.df_raw)} daily records")
        print(f"Columns: {list(self.df_raw.columns)}")
        return self.df_raw
    
    def parse_dates(self):
        """
        Parse date column and extract year, month information.
        """
        print("Parsing dates...")
        
        self.df_raw['date'] = pd.to_datetime(self.df_raw['date'], errors='coerce')
        
        self.df_raw['Year'] = self.df_raw['date'].dt.year
        self.df_raw['Month'] = self.df_raw['date'].dt.month
        
        print(f"Date range: {self.df_raw['date'].min()} to {self.df_raw['date'].max()}")
        print(f"Years: {sorted(self.df_raw['Year'].unique())}")
        
        return self.df_raw
    
    def assign_season(self):
        """
        Assign season based on month.
        
        Kharif: June (6) to October (10) - Monsoon/Summer season
        Rabi: November (11) to March (3) - Winter season
        """
        print("Assigning seasons...")
        
        def get_season(row):
            month = row['Month']
            year = row['Year']
            
            if 6 <= month <= 10:
                return 'Kharif', year
            elif month >= 11:
                return 'Rabi', year
            elif month <= 3:
                return 'Rabi', year - 1
            else:
                return None, None
        
        self.df_raw[['Season', 'Season_Year']] = self.df_raw.apply(
            get_season, axis=1, result_type='expand'
        )
        
        self.df_raw = self.df_raw.dropna(subset=['Season'])
        
        print(f"Records after season assignment: {len(self.df_raw)}")
        print(f"Seasons: {self.df_raw['Season'].unique()}")
        
        return self.df_raw
    
    def clean_district_names(self):
        """
        Clean and standardize district names.
        """
        print("Cleaning district names...")
        
        if 'district' in self.df_raw.columns:
            self.df_raw['District'] = self.df_raw['district'].str.strip().str.title()
        elif 'District' in self.df_raw.columns:
            self.df_raw['District'] = self.df_raw['District'].str.strip().str.title()
        else:
            print("Warning: No district column found!")
            self.df_raw['District'] = 'Unknown'
        
        return self.df_raw
    
    def convert_to_numeric(self):
        """
        Convert numeric columns to proper numeric types.
        """
        print("Converting columns to numeric types...")
        
        numeric_cols = ['cumm_rainfall', 'temp_min', 'temp_max', 
                       'humidity_min', 'humidity_max', 'wind_speed_min', 'wind_speed_max']
        
        for col in numeric_cols:
            if col in self.df_raw.columns:
                self.df_raw[col] = pd.to_numeric(self.df_raw[col], errors='coerce')
        
        return self.df_raw
    
    def aggregate_by_season(self):
        """
        Aggregate daily data to seasonal level.
        
        Returns:
            pd.DataFrame: Seasonal aggregated data
        """
        print("Aggregating data by season...")
        
        agg_dict = {}
        
        if 'cumm_rainfall' in self.df_raw.columns:
            agg_dict['cumm_rainfall'] = ['sum', 'mean', 'max']
        
        if 'temp_min' in self.df_raw.columns:
            agg_dict['temp_min'] = ['mean', 'min']
        
        if 'temp_max' in self.df_raw.columns:
            agg_dict['temp_max'] = ['mean', 'max']
        
        if 'humidity_min' in self.df_raw.columns:
            agg_dict['humidity_min'] = ['mean', 'min']
        
        if 'humidity_max' in self.df_raw.columns:
            agg_dict['humidity_max'] = ['mean', 'max']
        
        if 'wind_speed_min' in self.df_raw.columns:
            agg_dict['wind_speed_min'] = ['mean', 'min']
        
        if 'wind_speed_max' in self.df_raw.columns:
            agg_dict['wind_speed_max'] = ['mean', 'max']
        
        groupby_cols = ['District', 'Season_Year', 'Season']
        
        df_agg = self.df_raw.groupby(groupby_cols).agg(agg_dict).reset_index()
        
        df_agg.columns = ['_'.join(col).strip('_') if isinstance(col, tuple) else col 
                          for col in df_agg.columns]
        
        column_mapping = {
            'Season_Year': 'Year',
            'cumm_rainfall_sum': 'Total_Rainfall',
            'cumm_rainfall_mean': 'Avg_Daily_Rainfall',
            'cumm_rainfall_max': 'Max_Daily_Rainfall',
            'temp_min_mean': 'Avg_Temp_Min',
            'temp_min_min': 'Min_Temp',
            'temp_max_mean': 'Avg_Temp_Max',
            'temp_max_max': 'Max_Temp',
            'humidity_min_mean': 'Avg_Humidity_Min',
            'humidity_min_min': 'Min_Humidity',
            'humidity_max_mean': 'Avg_Humidity_Max',
            'humidity_max_max': 'Max_Humidity',
            'wind_speed_min_mean': 'Avg_Wind_Speed_Min',
            'wind_speed_min_min': 'Min_Wind_Speed',
            'wind_speed_max_mean': 'Avg_Wind_Speed_Max',
            'wind_speed_max_max': 'Max_Wind_Speed'
        }
        
        df_agg = df_agg.rename(columns=column_mapping)
        
        self.df_seasonal = df_agg
        
        print(f"Seasonal aggregation complete: {len(df_agg)} records")
        print(f"Columns: {list(df_agg.columns)}")
        
        return df_agg
    
    def transform(self):
        """
        Execute the complete transformation pipeline.
        
        Returns:
            pd.DataFrame: Seasonal aggregated data
        """
        print("\n" + "="*60)
        print("Starting Rainfall Seasonal Aggregation")
        print("="*60 + "\n")
        
        self.read_data()
        self.parse_dates()
        self.assign_season()
        self.clean_district_names()
        self.convert_to_numeric()
        df_seasonal = self.aggregate_by_season()
        
        print("\n" + "="*60)
        print("Aggregation Complete!")
        print("="*60)
        print(f"\nFinal shape: {df_seasonal.shape}")
        print(f"\nFirst few rows:")
        print(df_seasonal.head(10))
        
        return df_seasonal
    
    def export_to_csv(self, output_file):
        """
        Export seasonal data to CSV.
        
        Args:
            output_file (str): Path to output CSV file
        """
        if self.df_seasonal is None:
            raise ValueError("No seasonal data to export. Run transform() first.")
        
        print(f"\nExporting to: {output_file}")
        self.df_seasonal.to_csv(output_file, index=False)
        print(f"Export complete! {len(self.df_seasonal)} rows written.")
    
    def export_to_excel(self, output_file):
        """
        Export seasonal data to Excel.
        
        Args:
            output_file (str): Path to output Excel file
        """
        if self.df_seasonal is None:
            raise ValueError("No seasonal data to export. Run transform() first.")
        
        print(f"\nExporting to: {output_file}")
        self.df_seasonal.to_excel(output_file, index=False, engine='openpyxl')
        print(f"Export complete! {len(self.df_seasonal)} rows written.")
    
    def get_summary(self):
        """
        Get summary statistics of the seasonal data.
        
        Returns:
            dict: Summary statistics
        """
        if self.df_seasonal is None:
            raise ValueError("No seasonal data. Run transform() first.")
        
        stats = {
            'total_records': len(self.df_seasonal),
            'unique_districts': self.df_seasonal['District'].nunique(),
            'unique_years': self.df_seasonal['Year'].nunique(),
            'districts': sorted(self.df_seasonal['District'].unique().tolist()),
            'years': sorted(self.df_seasonal['Year'].unique().tolist()),
            'seasons': sorted(self.df_seasonal['Season'].unique().tolist())
        }
        
        return stats


def main():
    """
    Main function to demonstrate usage.
    """
    input_file = r"D:\AgriSet Project\Data\output\Rainfall_2018_Merged.csv"
    output_csv = r"D:\AgriSet Project\Data\output\Rainfall_Seasonal_Aggregated.csv"
    output_excel = r"D:\AgriSet Project\Data\output\Rainfall_Seasonal_Aggregated.xlsx"
    
    aggregator = RainfallSeasonalAggregator(input_file)
    
    df_seasonal = aggregator.transform()
    
    aggregator.export_to_csv(output_csv)
    aggregator.export_to_excel(output_excel)
    
    stats = aggregator.get_summary()
    print("\n" + "="*60)
    print("SUMMARY STATISTICS")
    print("="*60)
    print(f"Total seasonal records: {stats['total_records']}")
    print(f"Unique districts: {stats['unique_districts']}")
    print(f"Unique years: {stats['unique_years']}")
    print(f"Years covered: {', '.join(map(str, stats['years']))}")
    print(f"Seasons: {', '.join(stats['seasons'])}")
    print(f"\nDistricts ({stats['unique_districts']} total):")
    for i, district in enumerate(stats['districts'][:10], 1):
        print(f"  {i}. {district}")
    if len(stats['districts']) > 10:
        print(f"  ... and {len(stats['districts']) - 10} more")
    
    print(f"\n✓ Output files created:")
    print(f"  - CSV: {output_csv}")
    print(f"  - Excel: {output_excel}")


if __name__ == "__main__":
    main()
