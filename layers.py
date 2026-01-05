import numpy as np

# ============ 基础层类 ============
class Layer:
    """所有层的基类"""
    def __init__(self):
        self.input = None
        self.output = None
        self.trainable = False
    
    def forward(self, input_data):
        """前向传播"""
        raise NotImplementedError
    
    def backward(self, output_gradient, learning_rate=None):
        """反向传播"""
        raise NotImplementedError


# ============ 全连接层 ============
class Dense(Layer):
    """全连接层（线性层）"""
    def __init__(self, input_size, output_size, weight_init='he'):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.trainable = True
        
        # 权重初始化
        if weight_init == 'he':
            self.weights = np.random.randn(input_size, output_size) * np.sqrt(2.0 / input_size)
        elif weight_init == 'xavier':
            self.weights = np.random.randn(input_size, output_size) * np.sqrt(1.0 / input_size)
        else:
            self.weights = np.random.randn(input_size, output_size) * 0.01
        
        self.biases = np.zeros((1, output_size))
        
        # 用于优化器的参数
        self.weights_optimizer = None
        self.biases_optimizer = None

    def forward(self, input_data):
        self.input = input_data
        self.output = np.dot(input_data, self.weights) + self.biases
        return self.output

    def backward(self, output_gradient, learning_rate=None):
        # 计算梯度
        self.weights_gradient = np.dot(self.input.T, output_gradient)
        self.biases_gradient = np.sum(output_gradient, axis=0, keepdims=True)
        input_gradient = np.dot(output_gradient, self.weights.T)
        
        # 如果没有使用优化器，直接用学习率更新
        if learning_rate is not None and self.weights_optimizer is None:
            self.weights -= learning_rate * self.weights_gradient
            self.biases -= learning_rate * self.biases_gradient
        
        return input_gradient
    
    def get_parameters(self):
        """返回可训练参数"""
        return [self.weights, self.biases]
    
    def get_gradients(self):
        """返回梯度"""
        return [self.weights_gradient, self.biases_gradient]


# ============ 激活函数层 ============
class Activation(Layer):
    """激活函数基类"""
    def __init__(self):
        super().__init__()


class ReLU(Activation):
    """ReLU激活函数"""
    def forward(self, input_data):
        self.input = input_data
        self.output = np.maximum(0, input_data)
        return self.output
    
    def backward(self, output_gradient, learning_rate=None):
        return output_gradient * (self.input > 0)


class LeakyReLU(Activation):
    """Leaky ReLU激活函数"""
    def __init__(self, alpha=0.01):
        super().__init__()
        self.alpha = alpha
    
    def forward(self, input_data):
        self.input = input_data
        self.output = np.where(input_data > 0, input_data, input_data * self.alpha)
        return self.output
    
    def backward(self, output_gradient, learning_rate=None):
        return output_gradient * np.where(self.input > 0, 1, self.alpha)


class Sigmoid(Activation):
    """Sigmoid激活函数"""
    def forward(self, input_data):
        self.input = input_data
        self.output = 1 / (1 + np.exp(-np.clip(input_data, -500, 500)))
        return self.output
    
    def backward(self, output_gradient, learning_rate=None):
        sigmoid_grad = self.output * (1 - self.output)
        return output_gradient * sigmoid_grad


class Tanh(Activation):
    """Tanh激活函数"""
    def forward(self, input_data):
        self.input = input_data
        self.output = np.tanh(input_data)
        return self.output
    
    def backward(self, output_gradient, learning_rate=None):
        return output_gradient * (1 - self.output ** 2)


class Softmax(Activation):
    """Softmax激活函数"""
    def forward(self, input_data):
        self.input = input_data
        # 数值稳定性处理
        exp_values = np.exp(input_data - np.max(input_data, axis=1, keepdims=True))
        self.output = exp_values / np.sum(exp_values, axis=1, keepdims=True)
        return self.output
    
    def backward(self, output_gradient, learning_rate=None):
        # Softmax的反向传播通常与损失函数结合
        return output_gradient


# ============ Dropout层 ============
class Dropout(Layer):
    """Dropout正则化层"""
    def __init__(self, rate=0.5):
        super().__init__()
        self.rate = rate
        self.mask = None
        self.training = True
    
    def forward(self, input_data):
        self.input = input_data
        if self.training:
            self.mask = np.random.binomial(1, 1 - self.rate, size=input_data.shape) / (1 - self.rate)
            self.output = input_data * self.mask
        else:
            self.output = input_data
        return self.output
    
    def backward(self, output_gradient, learning_rate=None):
        return output_gradient * self.mask if self.training else output_gradient
    
    def set_training(self, training):
        self.training = training


# ============ 批归一化层 ============
class BatchNormalization(Layer):
    """批归一化层"""
    def __init__(self, input_size, momentum=0.9, epsilon=1e-5):
        super().__init__()
        self.input_size = input_size
        self.momentum = momentum
        self.epsilon = epsilon
        self.trainable = True
        
        # 可学习参数
        self.gamma = np.ones((1, input_size))
        self.beta = np.zeros((1, input_size))
        
        # 运行时统计量
        self.running_mean = np.zeros((1, input_size))
        self.running_var = np.ones((1, input_size))
        
        self.training = True
        
        # 用于优化器
        self.gamma_optimizer = None
        self.beta_optimizer = None
    
    def forward(self, input_data):
        self.input = input_data
        
        if self.training:
            # 训练模式：使用当前批次统计量
            self.batch_mean = np.mean(input_data, axis=0, keepdims=True)
            self.batch_var = np.var(input_data, axis=0, keepdims=True)
            
            # 更新运行时统计量
            self.running_mean = self.momentum * self.running_mean + (1 - self.momentum) * self.batch_mean
            self.running_var = self.momentum * self.running_var + (1 - self.momentum) * self.batch_var
            
            # 归一化
            self.x_normalized = (input_data - self.batch_mean) / np.sqrt(self.batch_var + self.epsilon)
        else:
            # 推理模式：使用运行时统计量
            self.x_normalized = (input_data - self.running_mean) / np.sqrt(self.running_var + self.epsilon)
        
        self.output = self.gamma * self.x_normalized + self.beta
        return self.output
    
    def backward(self, output_gradient, learning_rate=None):
        batch_size = output_gradient.shape[0]
        
        # 计算gamma和beta的梯度
        self.gamma_gradient = np.sum(output_gradient * self.x_normalized, axis=0, keepdims=True)
        self.beta_gradient = np.sum(output_gradient, axis=0, keepdims=True)
        
        # 计算输入梯度
        dx_normalized = output_gradient * self.gamma
        dvar = np.sum(dx_normalized * (self.input - self.batch_mean) * -0.5 * 
                     (self.batch_var + self.epsilon) ** (-1.5), axis=0, keepdims=True)
        dmean = np.sum(dx_normalized * -1 / np.sqrt(self.batch_var + self.epsilon), axis=0, keepdims=True) + \
                dvar * np.sum(-2 * (self.input - self.batch_mean), axis=0, keepdims=True) / batch_size
        
        input_gradient = dx_normalized / np.sqrt(self.batch_var + self.epsilon) + \
                        dvar * 2 * (self.input - self.batch_mean) / batch_size + \
                        dmean / batch_size
        
        # 如果没有使用优化器，直接用学习率更新
        if learning_rate is not None and self.gamma_optimizer is None:
            self.gamma -= learning_rate * self.gamma_gradient
            self.beta -= learning_rate * self.beta_gradient
        
        return input_gradient
    
    def set_training(self, training):
        self.training = training
    
    def get_parameters(self):
        return [self.gamma, self.beta]
    
    def get_gradients(self):
        return [self.gamma_gradient, self.beta_gradient]
