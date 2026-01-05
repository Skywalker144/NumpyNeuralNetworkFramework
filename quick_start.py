"""
快速入门指南
演示如何快速使用神经网络框架
"""

import numpy as np
from model import Sequential, train_test_split, accuracy_score
from layers import Dense, ReLU, Sigmoid
from losses import BinaryCrossEntropy
from optimizers import Adam

# 设置随机种子以获得可重复的结果
np.random.seed(42)

print("="*60)
print("NumPy神经网络框架 - 快速入门")
print("="*60)

# 1. 准备数据
print("\n步骤1: 准备数据")
print("-" * 60)
# 生成简单的二分类数据
X = np.random.randn(1000, 10)
y = (np.sum(X[:, :5], axis=1) > 0).astype(int).reshape(-1, 1)

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
print(f"训练集大小: {X_train.shape}")
print(f"测试集大小: {X_test.shape}")

# 2. 构建模型
print("\n步骤2: 构建模型")
print("-" * 60)
model = Sequential()
model.add(Dense(10, 32))      # 输入层 -> 隐藏层1
model.add(ReLU())             # 激活函数
model.add(Dense(32, 16))      # 隐藏层1 -> 隐藏层2
model.add(ReLU())             # 激活函数
model.add(Dense(16, 1))       # 隐藏层2 -> 输出层
model.add(Sigmoid())          # 输出激活函数

# 打印模型结构
model.summary()

# 3. 编译模型
print("\n步骤3: 编译模型")
print("-" * 60)
model.compile(
    loss=BinaryCrossEntropy(),
    optimizer=Adam(learning_rate=0.001)
)
print("模型已编译")
print("损失函数: BinaryCrossEntropy")
print("优化器: Adam (lr=0.001)")

# 4. 训练模型
print("\n步骤4: 训练模型")
print("-" * 60)
history = model.fit(
    X_train, y_train,
    epochs=30,
    batch_size=32,
    validation_data=(X_test, y_test),
    verbose=True
)

# 5. 评估模型
print("\n步骤5: 评估模型")
print("-" * 60)
test_loss = model.evaluate(X_test, y_test)
y_pred = model.predict(X_test)
test_accuracy = accuracy_score(y_test, y_pred)

print(f"测试集损失: {test_loss:.4f}")
print(f"测试集准确率: {test_accuracy*100:.2f}%")

# 6. 使用模型进行预测
print("\n步骤6: 使用模型进行预测")
print("-" * 60)
sample_data = X_test[:5]
predictions = model.predict(sample_data)

print("前5个样本的预测结果:")
for i, (true_label, pred_prob) in enumerate(zip(y_test[:5], predictions)):
    pred_label = 1 if pred_prob[0] > 0.5 else 0
    print(f"样本{i+1}: 真实标签={true_label[0]}, "
          f"预测概率={pred_prob[0]:.4f}, 预测标签={pred_label}")

# 7. 保存模型
print("\n步骤7: 保存模型权重")
print("-" * 60)
model.save_weights('my_model.npy')
print("✓ 模型权重已保存")

print("\n" + "="*60)
print("快速入门完成！")
print("="*60)
print("\n下一步:")
print("- 查看 examples.py 了解更多高级用例")
print("- 阅读 README.md 了解完整API文档")
print("- 尝试修改网络结构、优化器和超参数")
