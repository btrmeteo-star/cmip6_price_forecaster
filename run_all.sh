#!/bin/bash
set -e  # 遇错即停

echo "=== 🌾 CMIP6 农产品价格预测流程 ==="

echo "1. 生成气象特征..."
./src/cmip6_commodity/rice_processor.py
./src/cmip6_commodity/corn_processor.py
./src/cmip6_commodity/barley_processor.py

echo "2. 合并特征与价格..."
./src/data_merge.py

echo "3. 训练预测模型..."
./src/train_model.py

echo "✅ 全流程完成！结果位于:"
echo "   - data/final/       # 训练数据集"
echo "   - models/           # 保存的模型 (.pkl)"
