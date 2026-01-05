# NumPy神经网络框架

一个使用纯NumPy实现的完整深度学习框架，包含多种层类型、激活函数、损失函数和优化器。

## 特性

### 层类型
- **Dense（全连接层）**: 支持He和Xavier权重初始化
- **激活函数层**: ReLU, LeakyReLU, Sigmoid, Tanh, Softmax
- **Dropout**: 正则化层，防止过拟合
- **BatchNormalization**: 批归一化层，加速训练

### 损失函数
- **MeanSquaredError**: 均方误差（回归任务）
- **MeanAbsoluteError**: 平均绝对误差
- **BinaryCrossEntropy**: 二元交叉熵（二分类）
- **CategoricalCrossEntropy**: 分类交叉熵（多分类）
- **SoftmaxCrossEntropy**: Softmax+交叉熵组合（数值稳定）
- **HuberLoss**: Huber损失（对异常值鲁棒）

### 优化器
- **SGD**: 随机梯度下降（支持动量和Nesterov加速）
- **AdaGrad**: 自适应学习率
- **RMSprop**: RMSprop优化器
- **Adam**: Adam优化器
- **AdamW**: 带权重衰减的Adam
- **Nadam**: Nesterov + Adam

### 学习率调度器
- **StepLR**: 按步长衰减
- **ExponentialLR**: 指数衰减
- **CosineAnnealingLR**: 余弦退火

## 安装

只需要NumPy：
```bash
pip install numpy
```

## 快速开始

### 示例1：回归任务

```python
import numpy as np
from model import Sequential
from layers import Dense, ReLU
from losses import MeanSquaredError
from optimizers import Adam

# 生成数据
X = np.random.randn(1000, 10)
y = np.sum(X, axis=1, keepdims=True)

# 构建模型
model = Sequential()
model.add(Dense(10, 64))
model.add(ReLU())
model.add(Dense(64, 32))
model.add(ReLU())
model.add(Dense(32, 1))

# 编译模型
model.compile(
    loss=MeanSquaredError(),
    optimizer=Adam(learning_rate=0.001)
)

# 训练模型
history = model.fit(X, y, epochs=50, batch_size=32, verbose=True)

# 预测
predictions = model.predict(X[:5])
```

### 示例2：二分类任务

```python
from layers import Sigmoid
from losses import BinaryCrossEntropy

# 生成二分类数据
X = np.random.randn(1000, 20)
y = (np.sum(X[:, :10], axis=1) > 0).astype(int).reshape(-1, 1)

# 构建模型
model = Sequential()
model.add(Dense(20, 32))
model.add(ReLU())
model.add(Dense(32, 1))
model.add(Sigmoid())

# 编译和训练
model.compile(
    loss=BinaryCrossEntropy(),
    optimizer=Adam(learning_rate=0.001)
)
model.fit(X, y, epochs=30, batch_size=32)
```

### 示例3：多分类任务

```python
from losses import SoftmaxCrossEntropy
from layers import Dropout, BatchNormalization

# 生成多分类数据
X = np.random.randn(1000, 30)
y = np.random.randint(0, 5, (1000, 1))

# 构建模型（带正则化）
model = Sequential()
model.add(Dense(30, 128))
model.add(BatchNormalization(128))
model.add(ReLU())
model.add(Dropout(0.3))
model.add(Dense(128, 64))
model.add(ReLU())
model.add(Dense(64, 5))

# SoftmaxCrossEntropy内置了Softmax
model.compile(
    loss=SoftmaxCrossEntropy(),
    optimizer=Adam(learning_rate=0.001)
)
model.fit(X, y, epochs=50, batch_size=32)
```

## 高级功能

### 模型摘要
```python
model.summary()
```

### 模型保存和加载
```python
# 保存权重
model.save_weights('model_weights.npy')

# 加载权重
model.load_weights('model_weights.npy')
```

### 训练/评估模式切换
```python
# 训练模式（启用Dropout和BatchNormalization训练行为）
model.train_mode()

# 评估模式（禁用Dropout，BatchNormalization使用运行时统计）
model.eval_mode()
```

### 工具函数
```python
from model import accuracy_score, train_test_split, normalize, one_hot_encode

# 划分数据集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# 数据标准化
X_normalized, mean, std = normalize(X)

# One-hot编码
y_one_hot = one_hot_encode(y, num_classes=10)

# 计算准确率
accuracy = accuracy_score(y_true, y_pred)
```

## 项目结构

```
NN/
├── layers.py          # 层的实现
├── losses.py          # 损失函数
├── optimizers.py      # 优化器和学习率调度器
├── model.py           # 模型类和工具函数
├── examples.py        # 示例代码
└── README.md          # 文档
```

## 运行示例

运行所有示例代码：
```bash
python examples.py
```

示例包括：
1. 回归任务 - 拟合正弦函数
2. 二分类任务 - 异或问题
3. 多分类任务 - 合成数据集
4. 优化器比较
5. 模型保存和加载

## API参考

### Sequential模型

**方法**:
- `add(layer)`: 添加层
- `compile(loss, optimizer)`: 编译模型
- `fit(X, y, epochs, batch_size, validation_data, verbose)`: 训练模型
- `evaluate(X, y, batch_size)`: 评估模型
- `predict(X, batch_size)`: 预测
- `summary()`: 打印模型摘要
- `save_weights(filepath)`: 保存权重
- `load_weights(filepath)`: 加载权重

### 层类型

**Dense(input_size, output_size, weight_init='he')**
- 全连接层
- weight_init: 'he', 'xavier', 或 'default'

**ReLU(), Sigmoid(), Tanh(), Softmax()**
- 激活函数层

**Dropout(rate=0.5)**
- Dropout正则化
- rate: 丢弃率

**BatchNormalization(input_size, momentum=0.9, epsilon=1e-5)**
- 批归一化层

### 优化器

**Adam(learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8)**
**SGD(learning_rate=0.01, momentum=0.0, nesterov=False)**
**RMSprop(learning_rate=0.001, decay_rate=0.9, epsilon=1e-8)**
**AdaGrad(learning_rate=0.01, epsilon=1e-8)**
**AdamW(learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8, weight_decay=0.01)**
**Nadam(learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8)**

## 特点

- ✅ 纯NumPy实现，无其他依赖
- ✅ 完整的前向和反向传播
- ✅ 支持批量训练
- ✅ 训练/评估模式切换
- ✅ 模型保存和加载
- ✅ 多种优化器和学习率调度器
- ✅ 数值稳定的实现
- ✅ 清晰的代码结构和注释

## 许可证

MIT License

## 作者

神经网络框架示例实现
