import numpy as np

# ============ 优化器基类 ============
class Optimizer:
    """优化器基类"""
    def __init__(self, learning_rate=0.01):
        self.learning_rate = learning_rate
    
    def update(self, layer):
        """更新层的参数"""
        raise NotImplementedError


# ============ 随机梯度下降 ============
class SGD(Optimizer):
    """随机梯度下降优化器"""
    def __init__(self, learning_rate=0.01, momentum=0.0, nesterov=False):
        super().__init__(learning_rate)
        self.momentum = momentum
        self.nesterov = nesterov
        self.velocities = {}
    
    def update(self, layer):
        if not hasattr(layer, 'get_parameters'):
            return
        
        parameters = layer.get_parameters()
        gradients = layer.get_gradients()
        
        # 为每个参数初始化速度
        if id(layer) not in self.velocities:
            self.velocities[id(layer)] = [np.zeros_like(param) for param in parameters]
        
        velocities = self.velocities[id(layer)]
        
        for i, (param, grad) in enumerate(zip(parameters, gradients)):
            if self.momentum > 0:
                # 动量更新
                velocities[i] = self.momentum * velocities[i] - self.learning_rate * grad
                
                if self.nesterov:
                    # Nesterov加速梯度
                    param += self.momentum * velocities[i] - self.learning_rate * grad
                else:
                    # 标准动量
                    param += velocities[i]
            else:
                # 标准SGD
                param -= self.learning_rate * grad


# ============ AdaGrad优化器 ============
class AdaGrad(Optimizer):
    """AdaGrad优化器（自适应学习率）"""
    def __init__(self, learning_rate=0.01, epsilon=1e-8):
        super().__init__(learning_rate)
        self.epsilon = epsilon
        self.cache = {}
    
    def update(self, layer):
        if not hasattr(layer, 'get_parameters'):
            return
        
        parameters = layer.get_parameters()
        gradients = layer.get_gradients()
        
        # 为每个参数初始化缓存
        if id(layer) not in self.cache:
            self.cache[id(layer)] = [np.zeros_like(param) for param in parameters]
        
        caches = self.cache[id(layer)]
        
        for i, (param, grad) in enumerate(zip(parameters, gradients)):
            # 累积梯度平方
            caches[i] += grad ** 2
            # 更新参数
            param -= self.learning_rate * grad / (np.sqrt(caches[i]) + self.epsilon)


# ============ RMSprop优化器 ============
class RMSprop(Optimizer):
    """RMSprop优化器"""
    def __init__(self, learning_rate=0.001, decay_rate=0.9, epsilon=1e-8):
        super().__init__(learning_rate)
        self.decay_rate = decay_rate
        self.epsilon = epsilon
        self.cache = {}
    
    def update(self, layer):
        if not hasattr(layer, 'get_parameters'):
            return
        
        parameters = layer.get_parameters()
        gradients = layer.get_gradients()
        
        # 为每个参数初始化缓存
        if id(layer) not in self.cache:
            self.cache[id(layer)] = [np.zeros_like(param) for param in parameters]
        
        caches = self.cache[id(layer)]
        
        for i, (param, grad) in enumerate(zip(parameters, gradients)):
            # 梯度平方的移动平均
            caches[i] = self.decay_rate * caches[i] + (1 - self.decay_rate) * (grad ** 2)
            # 更新参数
            param -= self.learning_rate * grad / (np.sqrt(caches[i]) + self.epsilon)


# ============ Adam优化器 ============
class Adam(Optimizer):
    """Adam优化器（结合动量和RMSprop）"""
    def __init__(self, learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8):
        super().__init__(learning_rate)
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.m = {}  # 一阶矩估计（动量）
        self.v = {}  # 二阶矩估计（RMSprop）
        self.t = {}  # 时间步
    
    def update(self, layer):
        if not hasattr(layer, 'get_parameters'):
            return
        
        parameters = layer.get_parameters()
        gradients = layer.get_gradients()
        
        layer_id = id(layer)
        
        # 初始化
        if layer_id not in self.m:
            self.m[layer_id] = [np.zeros_like(param) for param in parameters]
            self.v[layer_id] = [np.zeros_like(param) for param in parameters]
            self.t[layer_id] = 0
        
        # 更新时间步
        self.t[layer_id] += 1
        t = self.t[layer_id]
        
        m_list = self.m[layer_id]
        v_list = self.v[layer_id]
        
        for i, (param, grad) in enumerate(zip(parameters, gradients)):
            # 更新一阶矩估计
            m_list[i] = self.beta1 * m_list[i] + (1 - self.beta1) * grad
            # 更新二阶矩估计
            v_list[i] = self.beta2 * v_list[i] + (1 - self.beta2) * (grad ** 2)
            
            # 偏差修正
            m_corrected = m_list[i] / (1 - self.beta1 ** t)
            v_corrected = v_list[i] / (1 - self.beta2 ** t)
            
            # 更新参数
            param -= self.learning_rate * m_corrected / (np.sqrt(v_corrected) + self.epsilon)


# ============ AdamW优化器 ============
class AdamW(Optimizer):
    """AdamW优化器（带权重衰减的Adam）"""
    def __init__(self, learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8, weight_decay=0.01):
        super().__init__(learning_rate)
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.weight_decay = weight_decay
        self.m = {}
        self.v = {}
        self.t = {}
    
    def update(self, layer):
        if not hasattr(layer, 'get_parameters'):
            return
        
        parameters = layer.get_parameters()
        gradients = layer.get_gradients()
        
        layer_id = id(layer)
        
        # 初始化
        if layer_id not in self.m:
            self.m[layer_id] = [np.zeros_like(param) for param in parameters]
            self.v[layer_id] = [np.zeros_like(param) for param in parameters]
            self.t[layer_id] = 0
        
        # 更新时间步
        self.t[layer_id] += 1
        t = self.t[layer_id]
        
        m_list = self.m[layer_id]
        v_list = self.v[layer_id]
        
        for i, (param, grad) in enumerate(zip(parameters, gradients)):
            # 更新一阶矩估计
            m_list[i] = self.beta1 * m_list[i] + (1 - self.beta1) * grad
            # 更新二阶矩估计
            v_list[i] = self.beta2 * v_list[i] + (1 - self.beta2) * (grad ** 2)
            
            # 偏差修正
            m_corrected = m_list[i] / (1 - self.beta1 ** t)
            v_corrected = v_list[i] / (1 - self.beta2 ** t)
            
            # Adam更新
            param -= self.learning_rate * m_corrected / (np.sqrt(v_corrected) + self.epsilon)
            
            # 权重衰减（解耦的L2正则化）
            param -= self.learning_rate * self.weight_decay * param


# ============ Nadam优化器 ============
class Nadam(Optimizer):
    """Nadam优化器（Nesterov + Adam）"""
    def __init__(self, learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8):
        super().__init__(learning_rate)
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.m = {}
        self.v = {}
        self.t = {}
    
    def update(self, layer):
        if not hasattr(layer, 'get_parameters'):
            return
        
        parameters = layer.get_parameters()
        gradients = layer.get_gradients()
        
        layer_id = id(layer)
        
        # 初始化
        if layer_id not in self.m:
            self.m[layer_id] = [np.zeros_like(param) for param in parameters]
            self.v[layer_id] = [np.zeros_like(param) for param in parameters]
            self.t[layer_id] = 0
        
        # 更新时间步
        self.t[layer_id] += 1
        t = self.t[layer_id]
        
        m_list = self.m[layer_id]
        v_list = self.v[layer_id]
        
        for i, (param, grad) in enumerate(zip(parameters, gradients)):
            # 更新一阶矩估计
            m_list[i] = self.beta1 * m_list[i] + (1 - self.beta1) * grad
            # 更新二阶矩估计
            v_list[i] = self.beta2 * v_list[i] + (1 - self.beta2) * (grad ** 2)
            
            # 偏差修正
            m_corrected = m_list[i] / (1 - self.beta1 ** t)
            v_corrected = v_list[i] / (1 - self.beta2 ** t)
            
            # Nesterov动量更新
            m_nesterov = self.beta1 * m_corrected + (1 - self.beta1) * grad / (1 - self.beta1 ** t)
            
            # 更新参数
            param -= self.learning_rate * m_nesterov / (np.sqrt(v_corrected) + self.epsilon)


# ============ 学习率调度器 ============
class LearningRateScheduler:
    """学习率调度器基类"""
    def __init__(self, optimizer):
        self.optimizer = optimizer
        self.initial_lr = optimizer.learning_rate
    
    def step(self, epoch=None):
        """更新学习率"""
        raise NotImplementedError


class StepLR(LearningRateScheduler):
    """按步长衰减学习率"""
    def __init__(self, optimizer, step_size, gamma=0.1):
        super().__init__(optimizer)
        self.step_size = step_size
        self.gamma = gamma
        self.last_epoch = 0
    
    def step(self, epoch=None):
        if epoch is None:
            epoch = self.last_epoch + 1
        self.last_epoch = epoch
        
        if epoch % self.step_size == 0 and epoch > 0:
            self.optimizer.learning_rate *= self.gamma


class ExponentialLR(LearningRateScheduler):
    """指数衰减学习率"""
    def __init__(self, optimizer, gamma=0.95):
        super().__init__(optimizer)
        self.gamma = gamma
        self.last_epoch = 0
    
    def step(self, epoch=None):
        if epoch is None:
            epoch = self.last_epoch + 1
        self.last_epoch = epoch
        
        self.optimizer.learning_rate = self.initial_lr * (self.gamma ** epoch)


class CosineAnnealingLR(LearningRateScheduler):
    """余弦退火学习率"""
    def __init__(self, optimizer, T_max, eta_min=0):
        super().__init__(optimizer)
        self.T_max = T_max
        self.eta_min = eta_min
        self.last_epoch = 0
    
    def step(self, epoch=None):
        if epoch is None:
            epoch = self.last_epoch + 1
        self.last_epoch = epoch
        
        self.optimizer.learning_rate = self.eta_min + (self.initial_lr - self.eta_min) * \
            (1 + np.cos(np.pi * epoch / self.T_max)) / 2
