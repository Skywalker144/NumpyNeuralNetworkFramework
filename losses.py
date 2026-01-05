import numpy as np

# ============ 损失函数基类 ============
class Loss:
    """损失函数基类"""
    def __init__(self):
        self.output = None
    
    def forward(self, y_pred, y_true):
        """计算损失值"""
        raise NotImplementedError
    
    def backward(self):
        """计算损失函数的梯度"""
        raise NotImplementedError


# ============ 均方误差损失 ============
class MeanSquaredError(Loss):
    """均方误差损失函数（用于回归问题）"""
    def __init__(self):
        super().__init__()
        self.y_pred = None
        self.y_true = None
    
    def forward(self, y_pred, y_true):
        self.y_pred = y_pred
        self.y_true = y_true
        self.output = np.mean(np.square(y_pred - y_true))
        return self.output
    
    def backward(self):
        batch_size = self.y_pred.shape[0]
        return 2 * (self.y_pred - self.y_true) / batch_size


# ============ 平均绝对误差损失 ============
class MeanAbsoluteError(Loss):
    """平均绝对误差损失函数"""
    def __init__(self):
        super().__init__()
        self.y_pred = None
        self.y_true = None
    
    def forward(self, y_pred, y_true):
        self.y_pred = y_pred
        self.y_true = y_true
        self.output = np.mean(np.abs(y_pred - y_true))
        return self.output
    
    def backward(self):
        batch_size = self.y_pred.shape[0]
        return np.sign(self.y_pred - self.y_true) / batch_size


# ============ 二元交叉熵损失 ============
class BinaryCrossEntropy(Loss):
    """二元交叉熵损失函数（用于二分类问题）"""
    def __init__(self):
        super().__init__()
        self.y_pred = None
        self.y_true = None
    
    def forward(self, y_pred, y_true):
        # 数值稳定性处理
        epsilon = 1e-15
        y_pred_clipped = np.clip(y_pred, epsilon, 1 - epsilon)
        
        self.y_pred = y_pred_clipped
        self.y_true = y_true
        
        # 计算交叉熵
        self.output = -np.mean(y_true * np.log(y_pred_clipped) + 
                              (1 - y_true) * np.log(1 - y_pred_clipped))
        return self.output
    
    def backward(self):
        batch_size = self.y_pred.shape[0]
        epsilon = 1e-15
        return (-(self.y_true / (self.y_pred + epsilon)) + 
                ((1 - self.y_true) / (1 - self.y_pred + epsilon))) / batch_size


# ============ 分类交叉熵损失 ============
class CategoricalCrossEntropy(Loss):
    """分类交叉熵损失函数（用于多分类问题）"""
    def __init__(self):
        super().__init__()
        self.y_pred = None
        self.y_true = None
    
    def forward(self, y_pred, y_true):
        # 数值稳定性处理
        epsilon = 1e-15
        y_pred_clipped = np.clip(y_pred, epsilon, 1 - epsilon)
        
        self.y_pred = y_pred_clipped
        self.y_true = y_true
        
        # 计算交叉熵
        if y_true.ndim == 1 or y_true.shape[1] == 1:
            # 如果y_true是类别索引
            correct_confidences = y_pred_clipped[range(len(y_pred)), y_true.flatten()]
            self.output = -np.mean(np.log(correct_confidences))
        else:
            # 如果y_true是one-hot编码
            self.output = -np.mean(np.sum(y_true * np.log(y_pred_clipped), axis=1))
        
        return self.output
    
    def backward(self):
        batch_size = self.y_pred.shape[0]
        
        if self.y_true.ndim == 1 or self.y_true.shape[1] == 1:
            # 如果y_true是类别索引，转换为one-hot
            y_true_one_hot = np.zeros_like(self.y_pred)
            y_true_one_hot[range(batch_size), self.y_true.flatten()] = 1
            gradient = -y_true_one_hot / self.y_pred
        else:
            # 如果y_true已经是one-hot编码
            gradient = -self.y_true / self.y_pred
        
        return gradient / batch_size


# ============ Softmax交叉熵损失（组合优化） ============
class SoftmaxCrossEntropy(Loss):
    """Softmax + 交叉熵损失（数值稳定的组合实现）"""
    def __init__(self):
        super().__init__()
        self.y_pred = None
        self.y_true = None
        self.softmax_output = None
    
    def forward(self, y_pred, y_true):
        self.y_pred = y_pred
        self.y_true = y_true
        
        # Softmax计算（数值稳定版本）
        exp_values = np.exp(y_pred - np.max(y_pred, axis=1, keepdims=True))
        self.softmax_output = exp_values / np.sum(exp_values, axis=1, keepdims=True)
        
        # 交叉熵损失
        epsilon = 1e-15
        softmax_clipped = np.clip(self.softmax_output, epsilon, 1 - epsilon)
        
        if y_true.ndim == 1 or y_true.shape[1] == 1:
            # 类别索引格式
            correct_confidences = softmax_clipped[range(len(softmax_clipped)), y_true.flatten()]
            self.output = -np.mean(np.log(correct_confidences))
        else:
            # one-hot格式
            self.output = -np.mean(np.sum(y_true * np.log(softmax_clipped), axis=1))
        
        return self.output
    
    def backward(self):
        batch_size = self.y_pred.shape[0]
        
        # Softmax + CrossEntropy的组合梯度非常简洁
        gradient = self.softmax_output.copy()
        
        if self.y_true.ndim == 1 or self.y_true.shape[1] == 1:
            # 类别索引格式
            gradient[range(batch_size), self.y_true.flatten()] -= 1
        else:
            # one-hot格式
            gradient -= self.y_true
        
        return gradient / batch_size


# ============ Huber损失 ============
class HuberLoss(Loss):
    """Huber损失函数（对异常值更鲁棒）"""
    def __init__(self, delta=1.0):
        super().__init__()
        self.delta = delta
        self.y_pred = None
        self.y_true = None
    
    def forward(self, y_pred, y_true):
        self.y_pred = y_pred
        self.y_true = y_true
        
        error = y_pred - y_true
        abs_error = np.abs(error)
        
        # Huber损失的定义
        quadratic = np.minimum(abs_error, self.delta)
        linear = abs_error - quadratic
        self.output = np.mean(0.5 * quadratic**2 + self.delta * linear)
        
        return self.output
    
    def backward(self):
        batch_size = self.y_pred.shape[0]
        error = self.y_pred - self.y_true
        
        # 根据误差大小选择不同的梯度
        gradient = np.where(
            np.abs(error) <= self.delta,
            error,
            self.delta * np.sign(error)
        )
        
        return gradient / batch_size
