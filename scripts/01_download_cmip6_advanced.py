#!/usr/bin/env python3
"""
高级CMIP6数据下载工具 - 使用esgf-pyclient
"""
import os
import sys
from pathlib import Path
import yaml
from esgfpyclient import ESGFClient
import xarray as xr
from tqdm import tqdm
import subprocess

class CMIP6Downloader:
    def __init__(self, config_path="config.yaml"):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # 初始化ESGF客户端
        self.client = ESGFClient()
        
        # 设置下载目录
        self.data_dir = Path("data/raw/cmip6")
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
    def search_datasets(self, variable, model, experiment, frequency='day'):
        """搜索CMIP6数据集"""
        query = {
            'project': 'CMIP6',
            'variable': variable,
            'source_id': model,
            'experiment_id': experiment,
            'frequency': frequency,
            'replica': 'false',
            'latest': 'true'
        }
        
        print(f"搜索: {query}")
        
        try:
            results = self.client.search(**query)
            print(f"找到 {len(results)} 个数据集")
            return results
        except Exception as e:
            print(f"搜索失败: {e}")
            return []
    
    def download_dataset(self, dataset, max_retries=3):
        """下载单个数据集"""
        dataset_id = dataset['id']
        filename = f"{dataset_id.replace('.', '_')}.nc"
        filepath = self.data_dir / filename
        
        if filepath.exists():
            print(f"文件已存在: {filepath}")
            return filepath
        
        print(f"开始下载: {dataset_id}")
        
        # 获取下载URL
        try:
            files = self.client.get_files(dataset_id)
            if not files:
                print(f"未找到下载文件: {dataset_id}")
                return None
            
            # 通常第一个文件是数据文件
            download_url = files[0]['url']
            
            # 使用wget下载
            cmd = [
                'wget', '--no-check-certificate',
                '--progress=bar:force:noscroll',
                '-O', str(filepath),
                download_url
            ]
            
            for attempt in range(max_retries):
                try:
                    print(f"下载尝试 {attempt + 1}/{max_retries}")
                    subprocess.run(cmd, check=True)
                    print(f"下载完成: {filepath}")
                    return filepath
                except subprocess.CalledProcessError as e:
                    print(f"下载失败: {e}")
                    if attempt < max_retries - 1:
                        print("重试...")
                    else:
                        print("达到最大重试次数")
                        return None
                        
        except Exception as e:
            print(f"获取下载URL失败: {e}")
            return None
    
    def download_all(self):
        """下载所有配置的数据"""
        downloaded_files = []
        
        for variable in self.config['cmip6']['variables']:
            for model in self.config['cmip6']['models']:
                for scenario in self.config['cmip6']['scenarios']:
                    print(f"\n{'='*60}")
                    print(f"处理: {variable} | {model} | {scenario}")
                    print(f"{'='*60}")
                    
                    # 搜索数据集
                    datasets = self.search_datasets(
                        variable=variable,
                        model=model,
                        experiment=scenario
                    )
                    
                    if datasets:
                        # 下载第一个匹配的数据集
                        filepath = self.download_dataset(datasets[0])
                        if filepath:
                            downloaded_files.append(filepath)
                    else:
                        print(f"未找到数据: {variable}_{model}_{scenario}")
        
        # 创建数据清单
        self.create_manifest(downloaded_files)
        
        return downloaded_files
    
    def create_manifest(self, filepaths):
        """创建数据清单文件"""
        manifest = []
        
        for filepath in filepaths:
            if filepath:
                try:
                    # 读取文件元数据
                    with xr.open_dataset(filepath) as ds:
                        file_info = {
                            'filename': filepath.name,
                            'variable': ds.attrs.get('variable_id', 'unknown'),
                            'model': ds.attrs.get('source_id', 'unknown'),
                            'experiment': ds.attrs.get('experiment_id', 'unknown'),
                            'time_range': f"{ds.time.values[0]} to {ds.time.values[-1]}",
                            'dimensions': dict(ds.dims),
                            'file_size': os.path.getsize(filepath) / (1024**3)  # GB
                        }
                        manifest.append(file_info)
                except Exception as e:
                    print(f"读取文件元数据失败 {filepath}: {e}")
        
        # 保存为JSON
        import json
        manifest_path = self.data_dir / "manifest.json"
        with open(manifest_path, 'w') as f:
            json.dump(manifest, f, indent=2, default=str)
        
        print(f"\n数据清单已保存至: {manifest_path}")
        
        # 打印摘要
        print("\n下载摘要:")
        print(f"总文件数: {len(manifest)}")
        total_size = sum(item['file_size'] for item in manifest)
        print(f"总大小: {total_size:.2f} GB")
    
    def validate_downloads(self):
        """验证下载的文件"""
        print("\n验证下载的文件...")
        
        nc_files = list(self.data_dir.glob("*.nc"))
        
        for filepath in tqdm(nc_files, desc="验证文件"):
            try:
                # 快速打开检查
                with xr.open_dataset(filepath) as ds:
                    # 检查基本属性
                    required_attrs = ['source_id', 'experiment_id', 'variable_id']
                    for attr in required_attrs:
                        if attr not in ds.attrs:
                            print(f"警告: {filepath} 缺少属性 {attr}")
                    
                    # 检查数据维度
                    if 'time' not in ds.dims:
                        print(f"警告: {filepath} 缺少时间维度")
                    
                print(f"✓ {filepath.name} - 验证通过")
                
            except Exception as e:
                print(f"✗ {filepath.name} - 验证失败: {e}")

def main():
    """主函数"""
    downloader = CMIP6Downloader()
    
    print("CMIP6数据下载工具")
    print("=" * 60)
    
    # 下载数据
    files = downloader.download_all()
    
    # 验证下载
    if files:
        downloader.validate_downloads()
    
    print("\n下载完成！")

if __name__ == "__main__":
    main()
