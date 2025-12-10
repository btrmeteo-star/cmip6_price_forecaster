#!/usr/bin/env python3
"""
生成分析报告
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import yaml
from datetime import datetime
import json
from pathlib import Path

def generate_report(config_path="config.yaml"):
    """生成PDF分析报告"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # 创建输出目录
    output_dir = Path("data/outputs/reports")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = output_dir / f"climate_price_report_{timestamp}.pdf"
    
    # 尝试加载数据
    try:
        # 气候数据
        climate_files = list(Path("data/processed").glob("*climate*.csv"))
        climate_data = {}
        for file in climate_files[:3]:  # 只加载前3个文件
            try:
                df = pd.read_csv(file)
                climate_data[file.stem] = df
            except:
                pass
        
        # 预测结果
        forecast_files = list(Path("data/outputs").glob("*forecast*.csv"))
        forecast_data = {}
        for file in forecast_files:
            try:
                df = pd.read_csv(file)
                forecast_data[file.stem] = df
            except:
                pass
        
        # 模型指标
        metrics_files = list(Path("data/outputs").glob("*metrics*.json"))
        metrics = {}
        for file in metrics_files:
            try:
                with open(file, 'r') as f:
                    metrics[file.stem] = json.load(f)
            except:
                pass
        
    except Exception as e:
        print(f"加载数据失败: {e}")
        climate_data = {}
        forecast_data = {}
        metrics = {}
    
    # 创建PDF报告
    with PdfPages(report_path) as pdf:
        # 第1页：封面
        fig, ax = plt.subplots(figsize=(8.5, 11))
        ax.axis('off')
        
        title = "CMIP6气候经济影响分析报告"
        subtitle = f"商品: {config['project']['commodity']} | 区域: {config['project']['region']}"
        date_str = datetime.now().strftime("%Y年%m月%d日")
        
        ax.text(0.5, 0.7, title, fontsize=24, ha='center', va='center', weight='bold')
        ax.text(0.5, 0.6, subtitle, fontsize=16, ha='center', va='center')
        ax.text(0.5, 0.5, date_str, fontsize=14, ha='center', va='center')
        ax.text(0.5, 0.3, "生成系统: CMIP6商品价格预测模型", fontsize=12, ha='center', va='center')
        
        pdf.savefig(fig, bbox_inches='tight')
        plt.close()
        
        # 第2页：执行摘要
        fig, ax = plt.subplots(figsize=(8.5, 11))
        ax.axis('off')
        
        summary_text = """
        执行摘要
        
        1. 项目概述
        本项目整合CMIP6气候数据与大宗商品价格预测模型，分析气候变化对农产品价格的长期影响。
        
        2. 关键发现
        • 未来气候变化将对商品供应产生显著影响
        • 极端天气事件频率增加可能推高价格波动性
        • 不同SSP情景下价格走势存在显著差异
        
        3. 数据概况
        • 气候模型: {}
        • 情景分析: {}
        • 时间范围: {} 至 {}
        • 分析区域: {}
        
        4. 主要结论
        建议投资者和决策者关注气候风险，将气候变化因素纳入长期投资和风险管理策略。
        """.format(
            ', '.join(config['cmip6']['models'][:3]),
            ', '.join(config['cmip6']['scenarios']),
            config['cmip6']['historical_period'],
            config['cmip6']['future_period'],
            config['project']['region']
        )
        
        ax.text(0.05, 0.95, summary_text, fontsize=10, ha='left', va='top',
               transform=ax.transAxes, wrap=True)
        
        pdf.savefig(fig, bbox_inches='tight')
        plt.close()
        
        # 第3页：气候趋势分析
        if climate_data:
            fig, axes = plt.subplots(2, 2, figsize=(11, 8.5))
            fig.suptitle('气候趋势分析', fontsize=16, weight='bold')
            
            plot_index = 0
            for name, data in list(climate_data.items())[:4]:
                if plot_index < 4:
                    ax = axes.flat[plot_index]
                    if 'year' in data.columns and len(data.columns) > 1:
                        # 绘制第一个数值列
                        numeric_cols = data.select_dtypes(include=[np.number]).columns
                        if len(numeric_cols) > 0:
                            col = numeric_cols[0]
                            ax.plot(data['year'], data[col], 'b-', linewidth=2)
                            ax.set_title(f'{name}: {col}')
                            ax.set_xlabel('年份')
                            ax.set_ylabel(col)
                            ax.grid(True, alpha=0.3)
                    plot_index += 1
            
            # 隐藏多余的子图
            for i in range(plot_index, 4):
                axes.flat[i].axis('off')
            
            plt.tight_layout()
            pdf.savefig(fig, bbox_inches='tight')
            plt.close()
        
        # 第4页：预测结果
        if forecast_data:
            fig, axes = plt.subplots(2, 1, figsize=(11, 8.5))
            fig.suptitle('价格预测结果', fontsize=16, weight='bold')
            
            for idx, (name, data) in enumerate(list(forecast_data.items())[:2]):
                ax = axes[idx]
                
                if 'ds' in data.columns and 'yhat' in data.columns:
                    dates = pd.to_datetime(data['ds'])
                    ax.plot(dates, data['yhat'], 'b-', label='预测值', linewidth=2)
                    
                    if 'yhat_lower' in data.columns and 'yhat_upper' in data.columns:
                        ax.fill_between(dates, data['yhat_lower'], data['yhat_upper'], 
                                       alpha=0.3, label='置信区间')
                    
                    ax.set_title(f'情景: {name}')
                    ax.set_xlabel('日期')
                    ax.set_ylabel('价格')
                    ax.legend()
                    ax.grid(True, alpha=0.3)
            
            plt.tight_layout()
            pdf.savefig(fig, bbox_inches='tight')
            plt.close()
        
        # 第5页：模型性能
        if metrics:
            fig, ax = plt.subplots(figsize=(8.5, 11))
            ax.axis('off')
            
            metrics_text = "模型性能指标\n\n"
            
            for model_name, model_metrics in metrics.items():
                metrics_text += f"{model_name}:\n"
                for key, value in model_metrics.items():
                    if isinstance(value, (int, float)):
                        metrics_text += f"  • {key}: {value:.4f}\n"
                metrics_text += "\n"
            
            if len(metrics) == 0:
                metrics_text += "暂无模型性能数据\n"
            
            ax.text(0.05, 0.95, metrics_text, fontsize=10, ha='left', va='top',
                   transform=ax.transAxes, wrap=True)
            
            pdf.savefig(fig, bbox_inches='tight')
            plt.close()
        
        # 第6页：建议与下一步
        fig, ax = plt.subplots(figsize=(8.5, 11))
        ax.axis('off')
        
        recommendations = """
        建议与下一步计划
        
        1. 风险管理建议
        • 将气候风险纳入投资决策框架
        • 建立气候压力测试机制
        • 考虑购买气候相关保险产品
        
        2. 投资策略调整
        • 关注气候适应性强的商品和地区
        • 分散投资以降低气候风险
        • 长期投资视角考虑气候变化影响
        
        3. 进一步研究
        • 扩展分析更多商品和地区
        • 加入更多经济和社会因素
        • 使用更高分辨率的气候数据
        
        4. 政策建议
        • 建立气候风险早期预警系统
        • 支持气候适应性农业技术
        • 加强国际合作应对气候变化
        
        5. 联系方式
        如需进一步分析或定制报告，请联系研究团队。
        """
        
        ax.text(0.05, 0.95, recommendations, fontsize=10, ha='left', va='top',
               transform=ax.transAxes, wrap=True)
        
        pdf.savefig(fig, bbox_inches='tight')
        plt.close()
    
    print(f"报告已生成: {report_path}")
    return report_path

def create_html_summary():
    """创建HTML格式的交互式摘要"""
    import plotly.express as px
    
    # 尝试加载预测数据
    try:
        forecast_files = list(Path("data/outputs").glob("*forecast*.csv"))
        
        if forecast_files:
            # 创建简单的HTML报告
            html_content = """
            <!DOCTYPE html>
            <html>
            <head>
                <title>CMIP6气候价格预测报告</title>
                <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
                <style>
                    body { font-family: Arial, sans-serif; margin: 40px; }
                    .header { background-color: #f0f0f0; padding: 20px; border-radius: 5px; }
                    .section { margin: 30px 0; }
                    .plot { width: 100%; height: 500px; }
                </style>
            </head>
            <body>
                <div class="header">
                    <h1>CMIP6气候价格预测报告</h1>
                    <p>生成时间: """ + datetime.now().strftime("%Y-%m-%d %H:%M:%S") + """</p>
                </div>
                
                <div class="section">
                    <h2>预测概览</h2>
                    <div id="plot1" class="plot"></div>
                </div>
                
                <div class="section">
                    <h2>情景对比</h2>
                    <div id="plot2" class="plot"></div>
                </div>
            """
            
            # 保存HTML文件
            html_path = Path("data/outputs/reports/interactive_report.html")
            with open(html_path, 'w') as f:
                f.write(html_content)
            
            print(f"交互式报告已生成: {html_path}")
    
    except Exception as e:
        print(f"创建HTML报告失败: {e}")

if __name__ == "__main__":
    print("生成分析报告...")
    pdf_report = generate_report()
    create_html_summary()
    print("报告生成完成！")
