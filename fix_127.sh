#!/usr/bin/env bash
# 确保无 BOM、无零宽空格
set -e
echo "=== 干净重跑 ==="
echo "OK"                    # 最简命令
echo "=== 检查常用命令 ==="
which python
which mlflow
echo "=== 完成，若仍 127 请手打命令 ==="
