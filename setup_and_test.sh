#!/bin/bash
# MolPAL环境安装和测试脚本

echo "=========================================="
echo "MolPAL环境安装和测试"
echo "=========================================="

# 步骤1: 创建conda环境
echo ""
echo "步骤1: 创建conda环境..."
echo "----------------------------------------"
cd /c/Users/34426/Desktop/molpal

# 检查环境是否已存在
if conda env list | grep -q "^molpal "; then
    echo "[INFO] molpal环境已存在，将删除并重新创建"
    conda env remove -n molpal -y
fi

echo "[INFO] 创建molpal环境（CPU版本）..."
conda env create -f environment_cpu.yml

if [ $? -ne 0 ]; then
    echo "[ERROR] 环境创建失败！"
    exit 1
fi

echo "[SUCCESS] 环境创建成功！"

# 步骤2: 激活环境并安装molpal
echo ""
echo "步骤2: 安装MolPAL..."
echo "----------------------------------------"

# 在conda环境中安装molpal
conda run -n molpal pip install -e .

if [ $? -ne 0 ]; then
    echo "[ERROR] MolPAL安装失败！"
    exit 1
fi

echo "[SUCCESS] MolPAL安装成功！"

# 步骤3: 验证安装
echo ""
echo "步骤3: 验证安装..."
echo "----------------------------------------"

# 检查molpal命令
conda run -n molpal molpal --help > /dev/null 2>&1

if [ $? -ne 0 ]; then
    echo "[ERROR] molpal命令不可用！"
    exit 1
fi

echo "[SUCCESS] molpal命令可用！"

# 步骤4: 测试D-MPNN导入
echo ""
echo "步骤4: 测试D-MPNN和MD-EI实现..."
echo "----------------------------------------"

conda run -n molpal python -c "
from molpal.models import model
from molpal.acquirer import metrics

# 测试D-MPNN
print('[TEST] 创建D-MPNN模型...')
dmpn = model('dmpn', test_batch_size=100, ncpu=1)
print(f'  [OK] D-MPNN类型: {dmpn.type_}')
print(f'  [OK] D-MPNN提供: {dmpn.provides}')

# 测试MD-EI
print('[TEST] 检查MD-EI采集函数...')
valid_metrics = metrics.valid_metrics()
print(f'  [OK] 有效指标: {valid_metrics}')
assert 'md_ei' in valid_metrics, 'md_ei不在有效指标中！'
print('  [OK] md_ei在有效指标中')

print('')
print('[SUCCESS] 所有测试通过！')
"

if [ $? -ne 0 ]; then
    echo "[ERROR] 测试失败！"
    exit 1
fi

echo ""
echo "=========================================="
echo "安装和测试完成！"
echo "=========================================="
echo ""
echo "使用方法:"
echo "  1. 激活环境: conda activate molpal"
echo "  2. 运行测试: bash test_enamine50k.sh"
echo ""
