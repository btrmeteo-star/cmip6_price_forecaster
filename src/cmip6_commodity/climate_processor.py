"""
处理CMIP6数据，提取农业气候指标
"""
import xarray as xr
import numpy as np
import pandas as pd
from pathlib import Path
import yaml

class ClimateProcessor:
    def __init__(self, config_path="config.yaml"):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
    def load_cmip6_data(self, filepath):
        """加载CMIP6 NetCDF文件"""
        ds = xr.open_dataset(filepath)   # 不給 chunks，直接載入記憶體
        return ds
    
    def calculate_growing_season_indicators(self, temp_ds, precip_ds):
        """计算生长季关键指标"""
        # 提取生长季月份 (5-9月)
        growing_season = temp_ds.sel(time=temp_ds['time.month'].isin([5,6,7,8,9]))
        
        # 计算指标
        indicators = {}
        
        # 1. 生长季平均温度
        indicators['growing_season_mean_temp'] = growing_season.groupby('time.year').mean()
        
        # 2. 生长季总降水量
        indicators['growing_season_total_precip'] = (
            precip_ds.sel(time=precip_ds['time.month'].isin([5,6,7,8,9]))
            .groupby('time.year').sum()
        )
        
        # 3. 极端高温天数（>35°C）
        extreme_heat = (temp_ds > self.config['agriculture']['heat_stress_threshold'])
        indicators['extreme_heat_days'] = extreme_heat.groupby('time.year').sum()
        
        # 4. 标准化降水指数 (SPI) - 简单版本
        indicators['spi_3month'] = self.calculate_spi(precip_ds, scale=3)
        
        return indicators
    
    def calculate_spi(self, precip_ds, scale=3):
        """计算标准化降水指数"""
        # 滚动累计降水
        precip_roll = precip_ds.rolling(time=scale, center=True).sum()
        
        # 按月份计算Gamma分布参数
        spi_values = []
        for month in range(1, 13):
            month_data = precip_roll.sel(time=precip_ds['time.month']==month)
            # 这里简化计算，实际应用需完整Gamma拟合
            spi = (month_data - month_data.mean()) / month_data.std()
            spi_values.append(spi)
        
        # 合并所有月份
        spi_all = xr.concat(spi_values, dim='time').sortby('time')
        return spi_all
    
    def extract_region(self, ds, bbox=None):
        """提取特定区域数据"""
        if bbox is None:
            bbox = self.config['region']['bbox']
        
        lat_min, lon_min, lat_max, lon_max = bbox
        region_ds = ds.sel(
            lat=slice(lat_min, lat_max),
            lon=slice(lon_min, lon_max)
        )
        return region_ds
    
    def save_indicators(self, indicators, output_dir):
        """保存处理后的指标为CSV"""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        for name, data in indicators.items():
            if isinstance(data, xr.DataArray):
                # 转换为DataFrame
                df = data.to_dataframe(name=name).reset_index()
                df.to_csv(output_dir / f"{name}.csv", index=False)
        
        print(f"指标已保存至: {output_dir}")

# 使用示例
if __name__ == "__main__":
    processor = ClimateProcessor()
    
    # 加载数据
    temp_data = processor.load_cmip6_data("data/raw/cmip6_tasmax.nc")
    precip_data = processor.load_cmip6_data("data/raw/cmip6_pr.nc")
    
    # 提取区域
    temp_region = processor.extract_region(temp_data)
    precip_region = processor.extract_region(precip_data)
    
    # 计算指标
    indicators = processor.calculate_growing_season_indicators(
        temp_region, precip_region
    )
    
    # 保存
    processor.save_indicators(indicators, "data/processed/")
