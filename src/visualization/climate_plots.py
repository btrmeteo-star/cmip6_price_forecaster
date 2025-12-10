"""
气候数据可视化工具
"""
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import xarray as xr
import numpy as np
import pandas as pd

class ClimateVisualizer:
    def __init__(self, style='seaborn'):
        self.style = style
        if style == 'seaborn':
            sns.set_style("whitegrid")
            plt.rcParams['figure.figsize'] = [12, 6]
        self.color_palette = sns.color_palette("husl", 8)
    
    def plot_temperature_trends(self, temp_data, scenarios=None, title="温度趋势"):
        """绘制温度趋势图"""
        if scenarios is None:
            scenarios = temp_data.scenario.unique() if hasattr(temp_data, 'scenario') else ['ssp245']
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        
        # 1. 时间序列图
        for i, scenario in enumerate(scenarios):
            scenario_data = temp_data[temp_data.scenario == scenario] if hasattr(temp_data, 'scenario') else temp_data
            axes[0].plot(scenario_data.year, scenario_data.temperature, 
                        label=scenario, color=self.color_palette[i], linewidth=2)
        
        axes[0].set_xlabel('年份')
        axes[0].set_ylabel('温度 (°C)')
        axes[0].set_title(f'{title} - 时间序列')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # 2. 箱线图（按年代）
        if 'year' in temp_data.columns:
            temp_data['decade'] = (temp_data['year'] // 10) * 10
            decades = sorted(temp_data['decade'].unique())
            
            box_data = []
            decade_labels = []
            for decade in decades:
                decade_data = temp_data[temp_data['decade'] == decade]['temperature']
                if len(decade_data) > 0:
                    box_data.append(decade_data.values)
                    decade_labels.append(f'{decade}s')
            
            axes[1].boxplot(box_data, labels=decade_labels)
            axes[1].set_xlabel('年代')
            axes[1].set_ylabel('温度 (°C)')
            axes[1].set_title(f'{title} - 年代际变化')
            axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    def plot_precipitation_analysis(self, precip_data, scenarios=None):
        """绘制降水分析图"""
        if scenarios is None:
            scenarios = precip_data.scenario.unique() if hasattr(precip_data, 'scenario') else ['ssp245']
        
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('年降水量趋势', '季节降水量变化', '极端降水频率', '干旱事件统计'),
            specs=[[{'type': 'scatter'}, {'type': 'bar'}],
                  [{'type': 'histogram'}, {'type': 'box'}]]
        )
        
        for i, scenario in enumerate(scenarios):
            scenario_data = precip_data[precip_data.scenario == scenario] if hasattr(precip_data, 'scenario') else precip_data
            
            # 1. 年降水量趋势
            fig.add_trace(
                go.Scatter(x=scenario_data.year, y=scenario_data.precipitation,
                          name=scenario, mode='lines', line=dict(width=2)),
                row=1, col=1
            )
            
            # 2. 季节降水量（如果数据包含月份）
            if 'month' in scenario_data.columns:
                seasonal = scenario_data.groupby('month')['precipitation'].mean()
                fig.add_trace(
                    go.Bar(x=seasonal.index, y=seasonal.values, name=scenario,
                          marker_color=px.colors.qualitative.Set1[i]),
                    row=1, col=2
                )
        
        fig.update_layout(height=800, showlegend=True, title_text="降水分析")
        return fig
    
    def plot_extreme_events(self, extremes_data, event_type='heatwave'):
        """绘制极端事件分析"""
        fig = go.Figure()
        
        if event_type == 'heatwave':
            fig.add_trace(go.Histogram(
                x=extremes_data['duration'],
                name='热浪持续时间',
                opacity=0.7,
                nbinsx=20
            ))
            
            fig.add_trace(go.Scatter(
                x=extremes_data['year'],
                y=extremes_data['frequency'],
                name='热浪频率',
                mode='lines+markers',
                yaxis='y2'
            ))
            
            fig.update_layout(
                title='热浪事件分析',
                xaxis_title='年份',
                yaxis_title='持续时间（天）',
                yaxis2=dict(
                    title='频率（次/年）',
                    overlaying='y',
                    side='right'
                )
            )
        
        elif event_type == 'drought':
            # 干旱事件可视化
            fig.add_trace(go.Scatter(
                x=extremes_data['year'],
                y=extremes_data['spi'],
                name='SPI指数',
                mode='lines',
                fill='tozeroy',
                fillcolor='rgba(255, 100, 100, 0.3)'
            ))
            
            # 添加干旱阈值线
            fig.add_hline(y=-1, line_dash="dash", line_color="red",
                         annotation_text="中度干旱阈值")
            fig.add_hline(y=-1.5, line_dash="dash", line_color="darkred",
                         annotation_text="严重干旱阈值")
            
            fig.update_layout(
                title='干旱事件分析 (SPI指数)',
                xaxis_title='年份',
                yaxis_title='标准化降水指数 (SPI)'
            )
        
        return fig
    
    def plot_climate_scenarios_comparison(self, data_dict, variable='temperature'):
        """比较不同气候情景"""
        fig = go.Figure()
        
        scenarios = list(data_dict.keys())
        colors = px.colors.qualitative.Plotly
        
        for i, (scenario, data) in enumerate(data_dict.items()):
            fig.add_trace(go.Scatter(
                x=data['year'],
                y=data[variable],
                name=scenario,
                mode='lines',
                line=dict(width=3, color=colors[i % len(colors)]),
                fill='tonexty' if i > 0 else None
            ))
        
        # 添加不确定性范围（如果数据包含上下界）
        for scenario, data in data_dict.items():
            if f'{variable}_lower' in data.columns and f'{variable}_upper' in data.columns:
                fig.add_trace(go.Scatter(
                    x=list(data['year']) + list(data['year'][::-1]),
                    y=list(data[f'{variable}_upper']) + list(data[f'{variable}_lower'][::-1]),
                    fill='toself',
                    fillcolor=f'rgba{tuple(list(px.colors.hex_to_rgb(colors[scenarios.index(scenario)])) + [0.2])}',
                    line=dict(color='rgba(255,255,255,0)'),
                    name=f'{scenario} 范围',
                    showlegend=True if scenario == scenarios[0] else False
                ))
        
        fig.update_layout(
            title=f'不同SSP情景下的{variable}预测比较',
            xaxis_title='年份',
            yaxis_title=variable,
            hovermode='x unified'
        )
        
        return fig
    
    def create_interactive_map(self, spatial_data, variable, time_slice=None):
        """创建交互式空间分布图"""
        if isinstance(spatial_data, xr.DataArray) or isinstance(spatial_data, xr.Dataset):
            # 选择时间切片
            if time_slice is not None:
                if isinstance(time_slice, int):
                    data_slice = spatial_data.isel(time=time_slice)
                elif isinstance(time_slice, str):
                    data_slice = spatial_data.sel(time=time_slice, method='nearest')
            else:
                data_slice = spatial_data.mean(dim='time')
            
            # 转换为DataFrame用于Plotly
            lons = data_slice.lon.values
            lats = data_slice.lat.values
            values = data_slice.values
            
            # 创建网格
            lon_grid, lat_grid = np.meshgrid(lons, lats)
            
            fig = go.Figure(data=go.Contour(
                z=values,
                x=lons,
                y=lats,
                colorscale='Viridis',
                colorbar=dict(title=variable)
            ))
            
            fig.update_layout(
                title=f'{variable}空间分布',
                xaxis_title='经度',
                yaxis_title='纬度'
            )
            
            return fig
        else:
            print("错误：需要xarray DataArray或Dataset")
            return None
    
    def save_plot(self, fig, filename, format='png', dpi=300):
        """保存图形到文件"""
        if isinstance(fig, go.Figure):
            fig.write_html(f"{filename}.html")
            print(f"交互式图形已保存为: {filename}.html")
        else:
            fig.savefig(f"{filename}.{format}", dpi=dpi, bbox_inches='tight')
            print(f"图形已保存为: {filename}.{format}")
