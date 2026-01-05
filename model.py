import numpy as np
from layers import Layer

# ============ 神经网络模型 ============
class Sequential:
    """序列模型：按顺序堆叠层"""
    def __init__(self):
        self.layers = []
        self.loss_function = None
        self.optimizer = None
        self.training = True
    
    def add(self, layer):
        """添加层到模型"""
        self.layers.append(layer)
    
    def compile(self, loss, optimizer):
        """编译模型：设置损失函数和优化器"""
        self.loss_function = loss
        self.optimizer = optimizer
        
        # 为每个可训练层设置优化器
        for layer in self.layers:
            if hasattr(layer, 'trainable') and layer.trainable:
                if hasattr(layer, 'weights'):
                    layer.weights_optimizer = optimizer
                    layer.biases_optimizer = optimizer
                if hasattr(layer, 'gamma'):
                    layer.gamma_optimizer = optimizer
                    layer.beta_optimizer = optimizer
    
    def forward(self, X):
        """前向传播"""
        output = X
        for layer in self.layers:
            output = layer.forward(output)
        return output
    
    def backward(self, loss_gradient):
        """反向传播"""
        gradient = loss_gradient
        for layer in reversed(self.layers):
            gradient = layer.backward(gradient)
        return gradient
    
    def train_mode(self):
        """设置为训练模式"""
        self.training = True
        for layer in self.layers:
            if hasattr(layer, 'set_training'):
                layer.set_training(True)
    
    def eval_mode(self):
        """设置为评估模式"""
        self.training = False
        for layer in self.layers:
            if hasattr(layer, 'set_training'):
                layer.set_training(False)
    
    def fit(self, X_train, y_train, epochs, batch_size=32, validation_data=None, verbose=True):
        """训练模型"""
        history = {
            'loss': [],
            'val_loss': [] if validation_data else None
        }
        
        n_samples = X_train.shape[0]
        
        for epoch in range(epochs):
            # 设置为训练模式
            self.train_mode()
            
            # 打乱训练数据
            indices = np.random.permutation(n_samples)
            X_shuffled = X_train[indices]
            y_shuffled = y_train[indices]
            
            epoch_loss = 0
            n_batches = 0
            
            # 批次训练
            for i in range(0, n_samples, batch_size):
                X_batch = X_shuffled[i:i+batch_size]
                y_batch = y_shuffled[i:i+batch_size]
                
                # 前向传播
                y_pred = self.forward(X_batch)
                
                # 计算损失
                loss = self.loss_function.forward(y_pred, y_batch)
                epoch_loss += loss
                n_batches += 1
                
                # 反向传播
                loss_gradient = self.loss_function.backward()
                self.backward(loss_gradient)
                
                # 更新参数
                if self.optimizer:
                    for layer in self.layers:
                        self.optimizer.update(layer)
            
            # 平均损失
            avg_loss = epoch_loss / n_batches
            history['loss'].append(avg_loss)
            
            # 验证集评估
            val_loss = None
            if validation_data:
                X_val, y_val = validation_data
                val_loss = self.evaluate(X_val, y_val, batch_size)
                history['val_loss'].append(val_loss)
            
            # 打印训练信息
            if verbose:
                if validation_data:
                    print(f'Epoch {epoch+1}/{epochs} - loss: {avg_loss:.4f} - val_loss: {val_loss:.4f}')
                else:
                    print(f'Epoch {epoch+1}/{epochs} - loss: {avg_loss:.4f}')
        
        return history
    
    def evaluate(self, X, y, batch_size=32):
        """评估模型"""
        self.eval_mode()
        
        n_samples = X.shape[0]
        total_loss = 0
        n_batches = 0
        
        for i in range(0, n_samples, batch_size):
            X_batch = X[i:i+batch_size]
            y_batch = y[i:i+batch_size]
            
            y_pred = self.forward(X_batch)
            loss = self.loss_function.forward(y_pred, y_batch)
            total_loss += loss
            n_batches += 1
        
        return total_loss / n_batches
    
    def predict(self, X, batch_size=32):
        """预测"""
        self.eval_mode()
        
        n_samples = X.shape[0]
        predictions = []
        
        for i in range(0, n_samples, batch_size):
            X_batch = X[i:i+batch_size]
            y_pred = self.forward(X_batch)
            predictions.append(y_pred)
        
        return np.vstack(predictions)
    
    def summary(self):
        """打印模型摘要"""
        print("=" * 65)
        print(f"{'Layer (type)':<25} {'Output Shape':<20} {'Param #':<15}")
        print("=" * 65)
        
        total_params = 0
        trainable_params = 0
        
        for i, layer in enumerate(self.layers):
            layer_name = f"{layer.__class__.__name__}_{i+1}"
            
            # 获取输出形状
            if hasattr(layer, 'output_size'):
                output_shape = f"(None, {layer.output_size})"
            elif hasattr(layer, 'input_size'):
                output_shape = f"(None, {layer.input_size})"
            else:
                output_shape = "-"
            
            # 计算参数数量
            params = 0
            if hasattr(layer, 'weights'):
                params += layer.weights.size
            if hasattr(layer, 'biases'):
                params += layer.biases.size
            if hasattr(layer, 'gamma'):
                params += layer.gamma.size
            if hasattr(layer, 'beta'):
                params += layer.beta.size
            
            total_params += params
            if hasattr(layer, 'trainable') and layer.trainable:
                trainable_params += params
            
            print(f"{layer_name:<25} {output_shape:<20} {params:<15,}")
        
        print("=" * 65)
        print(f"Total params: {total_params:,}")
        print(f"Trainable params: {trainable_params:,}")
        print(f"Non-trainable params: {total_params - trainable_params:,}")
        print("=" * 65)
    
    def save_weights(self, filepath):
        """保存模型权重"""
        weights_data = []
        for layer in self.layers:
            layer_weights = {}
            if hasattr(layer, 'weights'):
                layer_weights['weights'] = layer.weights
            if hasattr(layer, 'biases'):
                layer_weights['biases'] = layer.biases
            if hasattr(layer, 'gamma'):
                layer_weights['gamma'] = layer.gamma
            if hasattr(layer, 'beta'):
                layer_weights['beta'] = layer.beta
            if hasattr(layer, 'running_mean'):
                layer_weights['running_mean'] = layer.running_mean
            if hasattr(layer, 'running_var'):
                layer_weights['running_var'] = layer.running_var
            weights_data.append(layer_weights)
        
        np.save(filepath, weights_data, allow_pickle=True)
        print(f"模型权重已保存到 {filepath}")
    
    def load_weights(self, filepath):
        """加载模型权重"""
        weights_data = np.load(filepath, allow_pickle=True)
        
        for i, layer in enumerate(self.layers):
            layer_weights = weights_data[i]
            if 'weights' in layer_weights and hasattr(layer, 'weights'):
                layer.weights = layer_weights['weights']
            if 'biases' in layer_weights and hasattr(layer, 'biases'):
                layer.biases = layer_weights['biases']
            if 'gamma' in layer_weights and hasattr(layer, 'gamma'):
                layer.gamma = layer_weights['gamma']
            if 'beta' in layer_weights and hasattr(layer, 'beta'):
                layer.beta = layer_weights['beta']
            if 'running_mean' in layer_weights and hasattr(layer, 'running_mean'):
                layer.running_mean = layer_weights['running_mean']
            if 'running_var' in layer_weights and hasattr(layer, 'running_var'):
                layer.running_var = layer_weights['running_var']
        
        print(f"模型权重已从 {filepath} 加载")


# ============ 工具函数 ============
def accuracy_score(y_true, y_pred):
    """计算准确率"""
    if y_pred.shape[1] > 1:
        # 多分类：获取最大概率的类别
        predictions = np.argmax(y_pred, axis=1)
        if y_true.ndim > 1:
            targets = np.argmax(y_true, axis=1)
        else:
            targets = y_true.flatten()
    else:
        # 二分类
        predictions = (y_pred > 0.5).astype(int).flatten()
        targets = y_true.flatten()
    
    return np.mean(predictions == targets)


def train_test_split(X, y, test_size=0.2, random_state=None):
    """划分训练集和测试集"""
    if random_state is not None:
        np.random.seed(random_state)
    
    n_samples = X.shape[0]
    indices = np.random.permutation(n_samples)
    test_size = int(n_samples * test_size)
    
    test_indices = indices[:test_size]
    train_indices = indices[test_size:]
    
    return X[train_indices], X[test_indices], y[train_indices], y[test_indices]


def one_hot_encode(y, num_classes=None):
    """将标签转换为one-hot编码"""
    if num_classes is None:
        num_classes = int(np.max(y)) + 1
    
    n_samples = y.shape[0]
    one_hot = np.zeros((n_samples, num_classes))
    one_hot[np.arange(n_samples), y.flatten().astype(int)] = 1
    
    return one_hot


def normalize(X, mean=None, std=None):
    """标准化数据"""
    if mean is None:
        mean = np.mean(X, axis=0)
    if std is None:
        std = np.std(X, axis=0)
        std[std == 0] = 1  # 避免除以0
    
    return (X - mean) / std, mean, std
