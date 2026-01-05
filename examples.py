"""
神经网络框架示例代码
包含回归、二分类和多分类任务的示例
"""

import numpy as np
from model import Sequential, accuracy_score, train_test_split, one_hot_encode, normalize
from layers import Dense, ReLU, Sigmoid, Tanh, Softmax, Dropout, BatchNormalization
from losses import MeanSquaredError, BinaryCrossEntropy, SoftmaxCrossEntropy
from optimizers import SGD, Adam, RMSprop


def example_1_regression():
    """示例1：回归任务 - 拟合正弦函数"""
    print("\n" + "="*60)
    print("示例1：回归任务 - 拟合正弦函数")
    print("="*60)
    
    # 生成数据
    np.random.seed(42)
    X = np.linspace(0, 2*np.pi, 1000).reshape(-1, 1)
    y = np.sin(X) + np.random.normal(0, 0.1, X.shape)
    
    # 划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # 构建模型
    model = Sequential()
    model.add(Dense(1, 32, weight_init='he'))
    model.add(ReLU())
    model.add(Dense(32, 32))
    model.add(ReLU())
    model.add(Dense(32, 1))
    
    # 编译模型
    model.compile(
        loss=MeanSquaredError(),
        optimizer=Adam(learning_rate=0.01)
    )
    
    # 打印模型摘要
    model.summary()
    
    # 训练模型
    history = model.fit(
        X_train, y_train,
        epochs=50,
        batch_size=32,
        validation_data=(X_test, y_test),
        verbose=True
    )
    
    # 评估模型
    test_loss = model.evaluate(X_test, y_test)
    print(f"\n测试集损失: {test_loss:.4f}")
    
    # 预测
    y_pred = model.predict(X_test[:5])
    print("\n前5个预测值:")
    for i in range(5):
        print(f"真实值: {y_test[i][0]:.4f}, 预测值: {y_pred[i][0]:.4f}")


def example_2_binary_classification():
    """示例2：二分类任务 - 异或问题"""
    print("\n" + "="*60)
    print("示例2：二分类任务 - 异或问题")
    print("="*60)
    
    # 生成XOR数据
    np.random.seed(42)
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y = np.array([[0], [1], [1], [0]])
    
    # 扩展数据集（添加噪声）
    X_extended = []
    y_extended = []
    for _ in range(100):
        for i in range(len(X)):
            noise = np.random.normal(0, 0.1, 2)
            X_extended.append(X[i] + noise)
            y_extended.append(y[i])
    
    X = np.array(X_extended)
    y = np.array(y_extended)
    
    # 划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # 构建模型
    model = Sequential()
    model.add(Dense(2, 16))
    model.add(ReLU())
    model.add(Dense(16, 8))
    model.add(ReLU())
    model.add(Dense(8, 1))
    model.add(Sigmoid())
    
    # 编译模型
    model.compile(
        loss=BinaryCrossEntropy(),
        optimizer=Adam(learning_rate=0.01)
    )
    
    # 打印模型摘要
    model.summary()
    
    # 训练模型
    history = model.fit(
        X_train, y_train,
        epochs=100,
        batch_size=16,
        validation_data=(X_test, y_test),
        verbose=False
    )
    
    # 显示训练过程
    print(f"\n训练完成！")
    print(f"最终训练损失: {history['loss'][-1]:.4f}")
    print(f"最终验证损失: {history['val_loss'][-1]:.4f}")
    
    # 评估准确率
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"测试集准确率: {accuracy*100:.2f}%")
    
    # 测试XOR
    print("\nXOR真值表测试:")
    test_inputs = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    predictions = model.predict(test_inputs)
    for i, (inp, pred) in enumerate(zip(test_inputs, predictions)):
        print(f"{inp[0]} XOR {inp[1]} = {pred[0]:.4f} (预测: {int(pred[0] > 0.5)})")


def example_3_multiclass_classification():
    """示例3：多分类任务 - 鸢尾花数据集风格"""
    print("\n" + "="*60)
    print("示例3：多分类任务 - 合成数据集")
    print("="*60)
    
    # 生成合成的多分类数据
    np.random.seed(42)
    n_samples_per_class = 200
    n_features = 4
    n_classes = 3
    
    X = []
    y = []
    
    # 为每个类别生成数据
    for class_id in range(n_classes):
        # 每个类别有不同的均值
        mean = np.random.randn(n_features) * 3
        # 生成该类别的样本
        class_samples = np.random.randn(n_samples_per_class, n_features) + mean
        X.append(class_samples)
        y.append(np.full(n_samples_per_class, class_id))
    
    X = np.vstack(X)
    y = np.concatenate(y).reshape(-1, 1)
    
    # 标准化数据
    X, mean, std = normalize(X)
    
    # 划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # 构建模型（带Dropout和BatchNormalization）
    model = Sequential()
    model.add(Dense(n_features, 64))
    model.add(BatchNormalization(64))
    model.add(ReLU())
    model.add(Dropout(0.3))
    model.add(Dense(64, 32))
    model.add(BatchNormalization(32))
    model.add(ReLU())
    model.add(Dropout(0.3))
    model.add(Dense(32, n_classes))
    
    # 编译模型（使用SoftmaxCrossEntropy，它内置了Softmax）
    model.compile(
        loss=SoftmaxCrossEntropy(),
        optimizer=Adam(learning_rate=0.001)
    )
    
    # 打印模型摘要
    model.summary()
    
    # 训练模型
    history = model.fit(
        X_train, y_train,
        epochs=50,
        batch_size=32,
        validation_data=(X_test, y_test),
        verbose=False
    )
    
    # 显示训练过程
    print(f"\n训练完成！")
    print(f"最终训练损失: {history['loss'][-1]:.4f}")
    print(f"最终验证损失: {history['val_loss'][-1]:.4f}")
    
    # 评估准确率
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"测试集准确率: {accuracy*100:.2f}%")
    
    # 显示混淆矩阵
    print("\n预测分布:")
    predictions = np.argmax(y_pred, axis=1)
    for class_id in range(n_classes):
        count = np.sum(predictions == class_id)
        print(f"类别 {class_id}: {count} 个样本")


def example_4_optimizer_comparison():
    """示例4：不同优化器的比较"""
    print("\n" + "="*60)
    print("示例4：优化器比较")
    print("="*60)
    
    # 生成数据
    np.random.seed(42)
    X = np.random.randn(500, 10)
    y = (np.sum(X[:, :5], axis=1) > 0).astype(int).reshape(-1, 1)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    optimizers = {
        'SGD': SGD(learning_rate=0.01),
        'SGD+Momentum': SGD(learning_rate=0.01, momentum=0.9),
        'RMSprop': RMSprop(learning_rate=0.001),
        'Adam': Adam(learning_rate=0.001)
    }
    
    results = {}
    
    for opt_name, optimizer in optimizers.items():
        print(f"\n训练使用 {opt_name}...")
        
        # 构建模型
        model = Sequential()
        model.add(Dense(10, 32))
        model.add(ReLU())
        model.add(Dense(32, 16))
        model.add(ReLU())
        model.add(Dense(16, 1))
        model.add(Sigmoid())
        
        # 编译模型
        model.compile(
            loss=BinaryCrossEntropy(),
            optimizer=optimizer
        )
        
        # 训练模型
        history = model.fit(
            X_train, y_train,
            epochs=30,
            batch_size=32,
            validation_data=(X_test, y_test),
            verbose=False
        )
        
        results[opt_name] = {
            'final_loss': history['loss'][-1],
            'final_val_loss': history['val_loss'][-1],
            'history': history
        }
    
    # 显示结果
    print("\n" + "="*60)
    print("优化器比较结果:")
    print("="*60)
    for opt_name, result in results.items():
        print(f"{opt_name:15} - 训练损失: {result['final_loss']:.4f}, "
              f"验证损失: {result['final_val_loss']:.4f}")


def example_5_model_save_load():
    """示例5：模型保存和加载"""
    print("\n" + "="*60)
    print("示例5：模型保存和加载")
    print("="*60)
    
    # 生成简单数据
    np.random.seed(42)
    X = np.random.randn(100, 5)
    y = (np.sum(X, axis=1) > 0).astype(int).reshape(-1, 1)
    
    # 构建并训练模型
    print("\n训练原始模型...")
    model = Sequential()
    model.add(Dense(5, 10))
    model.add(ReLU())
    model.add(Dense(10, 1))
    model.add(Sigmoid())
    
    model.compile(
        loss=BinaryCrossEntropy(),
        optimizer=Adam(learning_rate=0.01)
    )
    
    model.fit(X, y, epochs=20, batch_size=16, verbose=False)
    
    # 保存模型权重
    model.save_weights('model_weights.npy')
    
    # 获取原始预测
    original_pred = model.predict(X[:5])
    print("\n原始模型预测:")
    print(original_pred)
    
    # 创建新模型并加载权重
    print("\n创建新模型并加载权重...")
    new_model = Sequential()
    new_model.add(Dense(5, 10))
    new_model.add(ReLU())
    new_model.add(Dense(10, 1))
    new_model.add(Sigmoid())
    
    new_model.load_weights('model_weights.npy')
    
    # 获取新模型预测
    new_pred = new_model.predict(X[:5])
    print("\n加载权重后的预测:")
    print(new_pred)
    
    # 验证预测是否一致
    print(f"\n预测差异: {np.max(np.abs(original_pred - new_pred)):.10f}")
    print("模型保存和加载测试通过！" if np.allclose(original_pred, new_pred) else "模型保存和加载测试失败！")


if __name__ == '__main__':
    """运行所有示例"""
    print("\n" + "="*60)
    print("神经网络框架示例演示")
    print("="*60)
    
    # 运行示例
    example_1_regression()
    example_2_binary_classification()
    example_3_multiclass_classification()
    example_4_optimizer_comparison()
    example_5_model_save_load()
    
    print("\n" + "="*60)
    print("所有示例运行完成！")
    print("="*60)
