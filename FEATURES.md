# 功能特性详解

## 核心组件

### 1. 层（Layers）

#### Dense（全连接层）
```python
Dense(input_size, output_size, weight_init='he')
```
- **He初始化**: 适用于ReLU激活函数
- **Xavier初始化**: 适用于Sigmoid/Tanh激活函数
- 自动梯度计算
- 支持批量处理

#### 激活函数
- **ReLU**: 最常用，计算效率高
- **LeakyReLU**: 解决ReLU死神经元问题
- **Sigmoid**: 二分类输出层
- **Tanh**: 零中心化激活
- **Softmax**: 多分类输出层

#### Dropout
```python
Dropout(rate=0.5)
```
- 训练时随机丢弃神经元
- 测试时自动关闭
- 防止过拟合

#### BatchNormalization
```python
BatchNormalization(input_size, momentum=0.9, epsilon=1e-5)
```
- 加速训练收敛
- 减少内部协变量偏移
- 训练/测试模式自动切换

### 2. 损失函数（Losses）

#### 回归任务
- **MeanSquaredError**: L2损失
- **MeanAbsoluteError**: L1损失
- **HuberLoss**: 结合MSE和MAE，对异常值鲁棒

#### 分类任务
- **BinaryCrossEntropy**: 二分类
- **CategoricalCrossEntropy**: 多分类
- **SoftmaxCrossEntropy**: 数值稳定的Softmax+CE组合

### 3. 优化器（Optimizers）

#### 基础优化器
```python
SGD(learning_rate=0.01, momentum=0.0, nesterov=False)
```
- 支持动量
- 支持Nesterov加速

#### 自适应学习率优化器
```python
Adam(learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8)
AdamW(learning_rate=0.001, weight_decay=0.01)
RMSprop(learning_rate=0.001, decay_rate=0.9)
```
- 自动调整每个参数的学习率
- 适用于各种任务

#### 学习率调度器
```python
StepLR(optimizer, step_size=10, gamma=0.1)
ExponentialLR(optimizer, gamma=0.95)
CosineAnnealingLR(optimizer, T_max=50)
```

### 4. 模型（Model）

#### Sequential模型
```python
model = Sequential()
model.add(Dense(10, 32))
model.add(ReLU())
model.compile(loss=..., optimizer=...)
model.fit(X, y, epochs=50)
```

**核心方法**:
- `add()`: 添加层
- `compile()`: 配置训练
- `fit()`: 训练模型
- `evaluate()`: 评估性能
- `predict()`: 预测
- `summary()`: 打印结构
- `save_weights()` / `load_weights()`: 持久化

## 高级特性

### 1. 数值稳定性
- Softmax使用减去最大值技巧
- 交叉熵使用epsilon防止log(0)
- 梯度裁剪防止梯度爆炸

### 2. 批量训练
- 支持Mini-batch梯度下降
- 自动数据打乱
- 高效的批量矩阵运算

### 3. 训练/评估模式
```python
model.train_mode()  # 启用Dropout和BatchNorm训练行为
model.eval_mode()   # 切换到评估模式
```

### 4. 验证集支持
```python
history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val)
)
```

### 5. 工具函数
```python
# 数据划分
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# 数据标准化
X_norm, mean, std = normalize(X)

# One-hot编码
y_one_hot = one_hot_encode(y, num_classes=10)

# 准确率计算
accuracy = accuracy_score(y_true, y_pred)
```

## 性能优化建议

### 1. 权重初始化
- ReLU系列: 使用He初始化
- Sigmoid/Tanh: 使用Xavier初始化

### 2. 学习率选择
- Adam: 0.001 (默认)
- SGD: 0.01-0.1
- 使用学习率调度器动态调整

### 3. 批量大小
- 小数据集: 16-32
- 大数据集: 64-256
- GPU训练: 尽量大

### 4. 正则化
- Dropout: 0.2-0.5
- 权重衰减: 0.01 (AdamW)
- BatchNormalization

### 5. 网络深度
- 简单任务: 2-3层
- 复杂任务: 5-10层
- 避免过深导致梯度消失

## 使用场景

### 回归任务
```python
model.add(Dense(input_size, hidden_size))
model.add(ReLU())
model.add(Dense(hidden_size, 1))
model.compile(loss=MeanSquaredError(), optimizer=Adam())
```

### 二分类
```python
model.add(Dense(input_size, hidden_size))
model.add(ReLU())
model.add(Dense(hidden_size, 1))
model.add(Sigmoid())
model.compile(loss=BinaryCrossEntropy(), optimizer=Adam())
```

### 多分类
```python
model.add(Dense(input_size, hidden_size))
model.add(ReLU())
model.add(Dense(hidden_size, num_classes))
model.compile(loss=SoftmaxCrossEntropy(), optimizer=Adam())
```

## 扩展建议

可以进一步添加的功能：
1. 卷积层（Conv2D）
2. 池化层（MaxPooling, AvgPooling）
3. 循环层（RNN, LSTM, GRU）
4. 更多正则化方法（L1/L2正则化）
5. 早停（Early Stopping）
6. 模型检查点（Checkpointing）
7. TensorBoard可视化
8. 数据增强
9. 混合精度训练
10. 分布式训练
