#!/usr/bin/env python3
"""
从ESGF下载CMIP6数据的自动化脚本
"""
import os
import subprocess
import yaml
from pathlib import Path

def download_cmip6_data(config_path="config.yaml"):
    """根据配置下载CMIP6数据"""
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    output_dir = Path("data/raw/cmip6")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # ESGF查询和下载命令模板
    # 注意：实际下载需要ESGF账号和正确配置的wget脚本
    
    base_url = "https://esgf-node.llnl.gov/thredds/fileServer"
    
    for scenario in config['cmip6']['scenarios']:
        for model in config['cmip6']['models']:
            for variable in config['cmip6']['variables']:
                print(f"下载: {model} - {scenario} - {variable}")
                
                # 这里需要根据实际ESGF文件结构构建URL
                # 以下是示例URL结构
                example_url = (
                    f"{base_url}/cmip6/CMIP6/ScenarioMIP/"
                    f"{model}/{scenario}/r1i1p1f1/day/{variable}/"
                    f"gn/v20190710/{variable}_day_{model}_{scenario}_"
                    f"r1i1p1f1_gn_20150101-21001231.nc"
                )
                
                # 实际应用中，您需要：
                # 1. 使用esgf-pyclient进行搜索
                # 2. 获取准确的下载链接
                # 3. 使用wget或aria2下载
                
                print(f"示例URL: {example_url}")
    
    print("\n注意：实际下载需要ESGF账号和认证。")
    print("建议通过 https://esgf-node.llnl.gov/search/cmip6/ 手动下载所需文件。")

if __name__ == "__main__":
    download_cmip6_data()
