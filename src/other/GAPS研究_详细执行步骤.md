# GAPS粒子识别研究 - 详细执行步骤

**基于**: 確認した計画書.txt  
**论文标题**：
- 日語：機械学習を用いたGAPS実験における宇宙線粒子識別の研究
- 英語：Machine Learning Approach to Cosmic-Ray Identification

---

## 阶段1：数据获取与验证（1～2周）

### Step 1.1: 确认GEANT4数据

**任务**：定位和验证数据

**具体操作**：
```bash
# 询问老师或在4090搜索
find / -name "*geant*" -o -name "*GAPS*" 2>/dev/null | head -50

# 找到数据后，检查：
ls -lh /path/to/geant4_data/
file /path/to/geant4_data/*.root  # 检查文件格式
```

**输出**：
- 数据路径确认
- 文件格式确认（ROOT、HDF5等）
- 反陽子数据量
- 反重陽子数据量
- 验证类不平衡比例 ≈ 10⁴:1

**检查点**：
```python
# test_data_info.py
import uproot  # 如果是ROOT格式
import h5py    # 如果是HDF5格式

# 读取第一个文件，检查结构
# 输出：事件数、通道数、标签分布
```

---

### Step 1.2: 创建数据加载脚本

**文件**：`src/data_loader.py`

**功能**：将GEANT4原始数据转换为Python可用格式

**具体代码框架**：
```python
# src/data_loader.py

import numpy as np
import h5py
import uproot  # 如果需要ROOT格式

class GEANTDataLoader:
    def __init__(self, data_path):
        self.data_path = data_path
    
    def load_single_event(self, event_id):
        """
        加载单个事件的GEANT4数据
        
        返回:
            hit_data: [num_hits, features]
                - features: [x, y, z, energy, time, channel_id, ...]
            label: 0 (反陽子) or 1 (反重陽子)
        """
        # 根据实际数据格式编写
        # 提取Si(Li)检出器hit点
        # 提取TOF计数器信息
        # 组合成hit_data
        pass
    
    def load_batch(self, event_ids):
        """
        加载一批事件
        """
        pass
    
    def get_all_event_ids(self):
        """
        获取所有有效事件ID
        """
        pass
    
    def get_class_distribution(self):
        """
        统计反陽子和反重陽子的数量
        返回: (num_antiproton, num_antideuteron, ratio)
        """
        pass
```

**验证**：
```bash
# test_data_loader.py
python << 'EOF'
from src.data_loader import GEANTDataLoader

loader = GEANTDataLoader("/path/to/geant4_data")

# 加载第一个事件
event = loader.load_single_event(0)
print(f"Event shape: {event['hit_data'].shape}")
print(f"Label: {event['label']}")

# 检查类分布
num_ap, num_ad, ratio = loader.get_class_distribution()
print(f"反陽子: {num_ap}, 反重陽子: {num_ad}, 比例: {ratio:.2e}")

# 验证：比例应接近10⁴:1
assert abs(ratio - 1e4) / 1e4 < 0.1, "类不平衡比例异常"
print("✅ 数据加载验证通过")
EOF
```

**输出**：
- `src/data_loader.py` 完整实现
- 验证报告：数据统计、类分布、样本验证

---

### Step 1.3: 分割训练/验证/测试数据集

**文件**：`src/data_splitter.py`

**任务**：
```python
class DataSplitter:
    def __init__(self, loader, random_seed=42):
        self.loader = loader
        self.random_seed = random_seed
    
    def split(self, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15):
        """
        分割数据集，保持各集合内的类平衡比例
        
        返回:
            train_ids, val_ids, test_ids
        """
        # 重要：保持反陽子:反重陽子 ≈ 10⁴:1 在各集合内
        pass
```

**输出**：
```
Train: 70% (约 X 反陽子 + Y 反重陽子)
Val:   15% (约 X' 反陽子 + Y' 反重陽子)
Test:  15% (约 X'' 反陽子 + Y'' 反重陽子)
```

---

## 阶段2：图构造与数据处理（2～3周）

### Step 2.1: 实现图构造

**文件**：`src/graph_builder.py`

**任务**：hit点 → 图结构 (Graph = Nodes + Edges)

**具体步骤**：

#### 2.1a：节点特征提取
```python
class GraphBuilder:
    def __init__(self, k_neighbors=10):
        """
        k_neighbors: k-nearest neighbors中的k值
        """
        self.k_neighbors = k_neighbors
    
    def extract_node_features(self, hit_data):
        """
        从GEANT4数据提取节点特征
        
        输入:
            hit_data: [num_hits, raw_features]
        
        输出:
            node_features: [num_hits, feature_dim]
                - feature_dim 至少包括: x, y, z, energy, time
                - 可能包括: channel_id, charge, ...
        """
        # 步骤1: 规范化坐标到[-1, 1]范围
        # 步骤2: 归一化能量值
        # 步骤3: 归一化时间
        pass
```

**验证代码**：
```python
# 取一个样本事件验证
sample_event = loader.load_single_event(0)
node_features = builder.extract_node_features(sample_event['hit_data'])

# 检查
print(f"节点数: {len(node_features)}")
print(f"特征维度: {node_features.shape[1]}")
print(f"特征范围: {node_features.min():.2f} ~ {node_features.max():.2f}")
assert node_features.shape[1] >= 5, "特征维度太少"
```

#### 2.1b：边构造（k-NN）
```python
def construct_edges_knn(node_features, k=10):
    """
    使用k-nearest neighbors构造图边
    
    输入:
        node_features: [num_nodes, feature_dim]
        k: 每个节点连接的最近邻数量
    
    输出:
        edge_index: [2, num_edges]
            - edge_index[0]: 源节点索引
            - edge_index[1]: 目标节点索引
        edge_attr (可选): [num_edges, edge_feature_dim]
            - 距离、时间差等
    
    """
    from sklearn.neighbors import NearestNeighbors
    
    # 计算距离
    nbrs = NearestNeighbors(n_neighbors=k+1).fit(node_features)
    distances, indices = nbrs.kneighbors(node_features)
    
    # 构造边（排除自环 indices[:, 0]）
    edge_list = []
    edge_weights = []
    for i in range(len(node_features)):
        for j in indices[i, 1:]:  # 跳过第一个（自己）
            edge_list.append([i, j])
            edge_weights.append(distances[i, np.where(indices[i] == j)[0][0]])
    
    return np.array(edge_list).T, np.array(edge_weights)
```

**验证**：
```python
node_feats = np.random.randn(50, 5)  # 50个节点
edge_index, edge_weights = construct_edges_knn(node_feats, k=5)

print(f"节点数: {node_feats.shape[0]}")
print(f"边数: {edge_index.shape[1]}")
# 期望: 约 50 * 5 = 250 条边（无向图则双向）
assert edge_index.shape[0] == 2
assert len(edge_weights) == edge_index.shape[1]
print("✅ 边构造验证通过")
```

#### 2.1c：转换为PyTorch Geometric格式
```python
from torch_geometric.data import Data
import torch

def create_graph_data(event, label):
    """
    将一个事件转换为PyTorch Geometric Data对象
    
    输入:
        event: 单个GEANT4事件
        label: 粒子类型标签
    
    输出:
        data: torch_geometric.data.Data 对象
    """
    # 提取节点特征
    node_features = builder.extract_node_features(event['hit_data'])
    
    # 构造边
    edge_index, edge_attr = construct_edges_knn(node_features, k=10)
    
    # 转换为torch张量
    x = torch.FloatTensor(node_features)
    edge_index = torch.LongTensor(edge_index)
    edge_attr = torch.FloatTensor(edge_attr).unsqueeze(1)  # [num_edges, 1]
    y = torch.LongTensor([label])  # [1]
    
    # 创建Data对象
    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y)
    
    return data
```

---

### Step 2.2: 创建完整数据集类

**文件**：`src/dataset.py`

```python
from torch_geometric.data import Dataset
import os

class GAPSDataset(Dataset):
    def __init__(self, loader, event_ids, root='data/processed', transform=None):
        """
        PyTorch Geometric 数据集类
        """
        self.loader = loader
        self.event_ids = event_ids
        super().__init__(root, transform)
    
    @property
    def raw_file_names(self):
        return []
    
    @property
    def processed_file_names(self):
        return [f'event_{eid}.pt' for eid in self.event_ids]
    
    def download(self):
        pass  # 数据已有
    
    def process(self):
        """
        处理数据：逐个事件转换为图数据并保存
        """
        builder = GraphBuilder(k_neighbors=10)
        
        for idx, event_id in enumerate(self.event_ids):
            event = self.loader.load_single_event(event_id)
            label = event['label']
            
            data = create_graph_data(event, label)
            
            # 保存
            torch.save(data, os.path.join(self.processed_dir, f'event_{event_id}.pt'))
            
            if (idx + 1) % 100 == 0:
                print(f"Processed {idx+1}/{len(self.event_ids)} events")
    
    def len(self):
        return len(self.event_ids)
    
    def get(self, idx):
        event_id = self.event_ids[idx]
        data = torch.load(os.path.join(self.processed_dir, f'event_{event_id}.pt'))
        return data
```

**执行**：
```bash
# create_dataset.py
python << 'EOF'
from src.data_loader import GEANTDataLoader
from src.data_splitter import DataSplitter
from src.dataset import GAPSDataset

# 加载数据
loader = GEANTDataLoader("/path/to/geant4_data")

# 分割
splitter = DataSplitter(loader)
train_ids, val_ids, test_ids = splitter.split()

# 创建数据集（会处理数据并保存）
train_dataset = GAPSDataset(loader, train_ids, root='data/processed')
val_dataset = GAPSDataset(loader, val_ids, root='data/processed')
test_dataset = GAPSDataset(loader, test_ids, root='data/processed')

print(f"✅ 数据集创建完成: Train={len(train_dataset)}, Val={len(val_dataset)}, Test={len(test_dataset)}")
EOF
```

**输出**：
- `data/processed/` 目录：所有图数据（PyTorch格式）
- 验证日志：事件数、图的统计信息

---

### Step 2.3: 可视化验证

**文件**：`src/utils/visualization.py`

```python
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def visualize_event(event, label, title=""):
    """
    可视化单个事件：3D散点图显示hit点
    """
    hit_data = event['hit_data']
    x, y, z = hit_data[:, 0], hit_data[:, 1], hit_data[:, 2]
    energy = hit_data[:, 3]
    
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    scatter = ax.scatter(x, y, z, c=energy, cmap='viridis', s=50)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(f"{title}\nLabel: {['反陽子', '反重陽子'][label]}")
    
    plt.colorbar(scatter, ax=ax, label='Energy')
    plt.show()

def visualize_graph(data, title=""):
    """
    可视化图结构：节点位置+边连接
    """
    # 使用前3个特征作为3D坐标
    pos = data.x[:, :3].numpy()
    
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # 绘制边
    edge_index = data.edge_index
    for i in range(edge_index.shape[1]):
        src, dst = edge_index[0, i], edge_index[1, i]
        ax.plot([pos[src, 0], pos[dst, 0]], 
               [pos[src, 1], pos[dst, 1]], 
               [pos[src, 2], pos[dst, 2]], 'b-', alpha=0.1)
    
    # 绘制节点
    ax.scatter(pos[:, 0], pos[:, 1], pos[:, 2], c='r', s=20)
    ax.set_title(f"{title}\n(Nodes={data.x.shape[0]}, Edges={data.edge_index.shape[1]})")
    plt.show()
```

**执行验证**：
```bash
python << 'EOF'
# 可视化几个样本
for i in range(3):
    event = loader.load_single_event(train_ids[i])
    visualize_event(event, event['label'], f"Event {i}")
    
    graph_data = train_dataset[i]
    visualize_graph(graph_data, f"Graph {i}")
EOF
```

---

## 阶段3：GNN模型实现（3～4周）

### Step 3.1: 实现基础GNN框架

**文件**：`src/models/gnn_base.py`

```python
import torch
import torch.nn as nn
from torch_geometric.nn import global_mean_pool

class GNNBase(nn.Module):
    """
    基础GNN分类器
    
    输入：图数据 (nodes, edges)
    输出：2分类概率 [p_antiproton, p_antideuteron]
    """
    
    def __init__(self, input_dim=5, hidden_dim=64, output_dim=2):
        super(GNNBase, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        
        # 子类应该在这里定义 self.conv layers
        # 示例：
        # self.conv1 = GNNConv(input_dim, hidden_dim)
        # self.conv2 = GNNConv(hidden_dim, hidden_dim)
        
        # 全局池化 + 分类头
        self.fc1 = nn.Linear(hidden_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
    
    def forward(self, data):
        """
        输入: torch_geometric.data.Data
        输出: logits [batch_size, 2]
        """
        x, edge_index, batch = data.x, data.edge_index, data.batch
        
        # GNN层（由子类实现）
        x = self.gnn_forward(x, edge_index)
        
        # 全局池化：每个图的所有节点特征求均值
        x = global_mean_pool(x, batch)
        
        # 分类头
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x
    
    def gnn_forward(self, x, edge_index):
        """
        由子类实现的GNN前向传播
        """
        raise NotImplementedError
```

---

### Step 3.2: 实现GravNet

**文件**：`src/models/gravnet.py`

```python
import torch.nn as nn
from torch_scatter import scatter_mean
from .gnn_base import GNNBase

class GravNet(GNNBase):
    """
    GravNet: 动态图神经网络
    特点：动态学习图连接结构
    
    参考: Qasim et al. 2019
    """
    
    def __init__(self, input_dim=5, hidden_dim=64, output_dim=2, 
                 num_layers=3, k_neighbors=10):
        super(GravNet, self).__init__(input_dim, hidden_dim, output_dim)
        
        self.num_layers = num_layers
        self.k_neighbors = k_neighbors
        
        # GravNet层
        self.gravnet_layers = nn.ModuleList([
            self._build_gravnet_layer(input_dim if i == 0 else hidden_dim, 
                                      hidden_dim)
            for i in range(num_layers)
        ])
    
    def _build_gravnet_layer(self, in_channels, out_channels):
        """
        构建单个GravNet层：
        1. 学习变换后的坐标空间
        2. 在该空间中计算k-NN
        3. 聚合邻域信息
        """
        return GravNetConv(in_channels, out_channels, k=self.k_neighbors)
    
    def gnn_forward(self, x, edge_index):
        for layer in self.gravnet_layers:
            x = layer(x, edge_index)
            x = nn.functional.relu(x)
        return x


class GravNetConv(nn.Module):
    """
    GravNet卷积层
    
    步骤：
    1. 学习一个变换矩阵，将x映射到s维空间
    2. 在s维空间中计算每个点的k-nearest neighbors
    3. 聚合邻域特征
    """
    
    def __init__(self, in_channels, out_channels, k=10, dropout=0.0):
        super(GravNetConv, self).__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.k = k
        
        # 学习变换矩阵：x -> s维空间
        self.mlp_transform = nn.Sequential(
            nn.Linear(in_channels, 64),
            nn.ReLU(),
            nn.Linear(64, out_channels)
        )
        
        # 聚合MLP
        self.mlp_aggregate = nn.Sequential(
            nn.Linear(2 * in_channels, 64),
            nn.ReLU(),
            nn.Linear(64, out_channels),
            nn.Dropout(dropout)
        )
    
    def forward(self, x, edge_index):
        """
        x: [num_nodes, in_channels]
        edge_index: [2, num_edges] 或 None（动态构造）
        """
        # 1. 变换到s维空间
        s = self.mlp_transform(x)  # [num_nodes, out_channels]
        
        # 2. 计算k-NN（基于s空间）
        from sklearn.neighbors import NearestNeighbors
        nbrs = NearestNeighbors(n_neighbors=self.k+1).fit(s.detach().cpu().numpy())
        _, indices = nbrs.kneighbors(s.detach().cpu().numpy())
        
        # 3. 聚合邻域特征
        # 对每个节点，聚合其k-NN的特征
        out = []
        for i in range(x.shape[0]):
            neighbor_indices = indices[i, 1:]  # 排除自己
            neighbor_feats = x[neighbor_indices]  # [k, in_channels]
            
            # 拼接：自身特征 + 邻域特征（各种聚合方式）
            center_feat = x[i:i+1].repeat(self.k, 1)  # [k, in_channels]
            
            # 方式1：求均值
            neighbor_mean = neighbor_feats.mean(dim=0, keepdim=True)
            combined = torch.cat([x[i:i+1], neighbor_mean], dim=1)
            
            # 通过MLP
            out_feat = self.mlp_aggregate(combined)
            out.append(out_feat)
        
        out = torch.cat(out, dim=0)  # [num_nodes, out_channels]
        return out
```

---

### Step 3.3: 实现DGCNN

**文件**：`src/models/dgcnn.py`

```python
from torch_geometric.nn import DynamicEdgeConv, global_mean_pool
from .gnn_base import GNNBase

class DGCNN(GNNBase):
    """
    Dynamic Graph Convolutional Neural Network
    特点：Edge Convolution + 动态图更新
    
    参考: Wang et al. 2019
    """
    
    def __init__(self, input_dim=5, hidden_dim=64, output_dim=2, 
                 num_layers=3, k_neighbors=10):
        super(DGCNN, self).__init__(input_dim, hidden_dim, output_dim)
        
        self.num_layers = num_layers
        self.k_neighbors = k_neighbors
        
        # DynamicEdgeConv层
        self.edge_convs = nn.ModuleList([
            DynamicEdgeConv(
                self._build_edge_mlp(input_dim if i == 0 else hidden_dim),
                k=k_neighbors
            )
            for i in range(num_layers)
        ])
    
    def _build_edge_mlp(self, in_channels):
        """
        边卷积的MLP：处理源节点和目标节点的特征拼接
        """
        return nn.Sequential(
            nn.Linear(2 * in_channels, 64),
            nn.ReLU(),
            nn.Linear(64, self.hidden_dim),
            nn.ReLU()
        )
    
    def gnn_forward(self, x, edge_index):
        xs = []
        for conv in self.edge_convs:
            # DynamicEdgeConv 动态重新计算k-NN
            x = conv(x)
            x = nn.functional.relu(x)
            xs.append(x)
        
        # 多尺度特征拼接
        x = torch.cat(xs, dim=1)
        return x
```

---

### Step 3.4: 实现Focal Loss

**文件**：`src/losses.py`

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class FocalLoss(nn.Module):
    """
    Focal Loss：处理极端类不平衡问题
    
    公式：FL = -α * (1-p_t)^γ * log(p_t)
    
    其中：
    - p_t：模型对正确类的预测概率
    - γ：焦点参数（γ=0则为交叉熵，γ越大越关注难分类样本）
    - α：类权重参数（给少数类更高权重）
    """
    
    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean'):
        """
        参数：
        - alpha: 少数类权重（推荐0.25）
        - gamma: 焦点参数（推荐2.0）
        - reduction: 'mean' 或 'sum'
        """
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    
    def forward(self, inputs, targets):
        """
        输入：
        - inputs: [batch_size, num_classes] logits
        - targets: [batch_size] 类标签 (0 or 1)
        
        输出：
        - loss: 标量
        """
        # 计算交叉熵
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        
        # 计算预测概率
        p = torch.exp(-ce_loss)
        
        # 计算Focal Loss
        focal_loss = self.alpha * (1 - p) ** self.gamma * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


# 使用示例：
# loss_fn = FocalLoss(alpha=0.25, gamma=2.0)
# loss = loss_fn(model_output, labels)
```

---

## 阶段4：训练循环（第2个月）

### Step 4.1: 实现训练脚本

**文件**：`src/train.py`

```python
import torch
import torch.optim as optim
from torch_geometric.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

def train_epoch(model, train_loader, optimizer, loss_fn, device):
    """
    训练一个epoch
    """
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    for batch in train_loader:
        batch = batch.to(device)
        
        # 前向传播
        out = model(batch)
        loss = loss_fn(out, batch.y)
        
        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # 统计
        total_loss += loss.item()
        pred = out.argmax(dim=1)
        correct += (pred == batch.y).sum().item()
        total += batch.y.size(0)
    
    return total_loss / len(train_loader), correct / total


def validate(model, val_loader, loss_fn, device):
    """
    验证
    """
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for batch in val_loader:
            batch = batch.to(device)
            out = model(batch)
            loss = loss_fn(out, batch.y)
            
            total_loss += loss.item()
            pred = out.argmax(dim=1)
            correct += (pred == batch.y).sum().item()
            total += batch.y.size(0)
    
    return total_loss / len(val_loader), correct / total


def train_model(model_name='gravnet', num_epochs=100, batch_size=64, lr=0.001):
    """
    完整训练流程
    
    参数：
    - model_name: 'gravnet' 或 'dgcnn'
    - num_epochs: 训练轮数
    - batch_size: 批大小
    - lr: 初始学习率
    """
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 1. 加载数据
    from src.dataset import GAPSDataset
    from src.data_loader import GEANTDataLoader
    from src.data_splitter import DataSplitter
    
    loader = GEANTDataLoader("/path/to/geant4_data")
    splitter = DataSplitter(loader)
    train_ids, val_ids, test_ids = splitter.split()
    
    train_dataset = GAPSDataset(loader, train_ids)
    val_dataset = GAPSDataset(loader, val_ids)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    
    # 2. 创建模型
    if model_name == 'gravnet':
        from src.models.gravnet import GravNet
        model = GravNet(input_dim=5, hidden_dim=64, output_dim=2)
    elif model_name == 'dgcnn':
        from src.models.dgcnn import DGCNN
        model = DGCNN(input_dim=5, hidden_dim=64, output_dim=2)
    
    model = model.to(device)
    
    # 3. 优化器和损失
    optimizer = optim.Adam(model.parameters(), lr=lr)
    loss_fn = FocalLoss(alpha=0.25, gamma=2.0)
    
    # 4. 学习率衰减
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
    
    # 5. TensorBoard日志
    writer = SummaryWriter(f'logs/{model_name}')
    
    # 6. 训练循环
    for epoch in range(num_epochs):
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, loss_fn, device)
        val_loss, val_acc = validate(model, val_loader, loss_fn, device)
        
        scheduler.step()
        
        # 日志
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{num_epochs}")
            print(f"  Train Loss: {train_loss:.4f}, Acc: {train_acc:.4f}")
            print(f"  Val Loss:   {val_loss:.4f}, Acc:   {val_acc:.4f}")
        
        writer.add_scalar('Train/Loss', train_loss, epoch)
        writer.add_scalar('Train/Acc', train_acc, epoch)
        writer.add_scalar('Val/Loss', val_loss, epoch)
        writer.add_scalar('Val/Acc', val_acc, epoch)
    
    # 7. 保存模型
    torch.save(model.state_dict(), f'results/model_{model_name}.pth')
    writer.close()
    
    return model

# 执行：
# python << 'EOF'
# model = train_model('gravnet', num_epochs=100)
# EOF
```

---

## 阶段5：性能评估（第2个月末）

### Step 5.1: 生成Rejection Curve

**文件**：`src/utils/metrics.py`

```python
import numpy as np
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt

def compute_rejection_curve(y_true, y_pred_proba):
    """
    计算Rejection Curve（拒绝曲线）
    
    输入：
    - y_true: [N] 真实标签 (0=反陽子, 1=反重陽子)
    - y_pred_proba: [N, 2] 预测概率
    
    输出：
    - signal_eff: 信号效率 (反重陽子识别率)
    - rejection_pow: 拒绝力
    - thresholds: 用于的阈值
    """
    
    # 提取反重陽子的预测概率
    y_score = y_pred_proba[:, 1]
    
    # ROC曲线：按阈值扫描
    fpr, tpr, thresholds = roc_curve(y_true, y_score)
    
    # signal_eff = tpr（True Positive Rate）
    # rejection_pow = 1 / fpr（当fpr > 0）
    
    signal_eff = tpr
    
    # 避免除以0
    rejection_pow = np.where(fpr > 0, 1 / fpr, np.inf)
    
    # 限制最大值
    rejection_pow = np.minimum(rejection_pow, 1e6)
    
    return signal_eff, rejection_pow, thresholds


def plot_rejection_curve(signal_eff, rejection_pow, save_path='rejection_curve.png'):
    """
    绘制Rejection Curve
    """
    plt.figure(figsize=(10, 8))
    
    plt.loglog(signal_eff, rejection_pow, 'b-', linewidth=2, label='GNN')
    plt.axhline(y=1e5, color='r', linestyle='--', label='Target (10⁵)')
    
    plt.xlabel('Signal Efficiency (反重陽子识别率)', fontsize=12)
    plt.ylabel('Rejection Power (拒绝力)', fontsize=12)
    plt.title('Rejection Curve: GNN Particle Identification', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=11)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    print(f"✅ Rejection Curve saved to {save_path}")
    plt.show()


# 使用：
# signal_eff, rejection_pow, _ = compute_rejection_curve(y_test, y_pred_proba)
# plot_rejection_curve(signal_eff, rejection_pow)
```

### Step 5.2: 与CNN+DNN基线对比

**文件**：`src/evaluate.py`

```python
def evaluate_and_compare(gravnet_model, dgcnn_model, test_loader, device):
    """
    评估两个模型并对标CNN+DNN基线
    """
    
    all_preds_gravnet = []
    all_preds_dgcnn = []
    all_targets = []
    
    gravnet_model.eval()
    dgcnn_model.eval()
    
    with torch.no_grad():
        for batch in test_loader:
            batch = batch.to(device)
            
            # GravNet预测
            out_gravnet = gravnet_model(batch)
            proba_gravnet = torch.softmax(out_gravnet, dim=1)
            
            # DGCNN预测
            out_dgcnn = dgcnn_model(batch)
            proba_dgcnn = torch.softmax(out_dgcnn, dim=1)
            
            all_preds_gravnet.append(proba_gravnet.cpu().numpy())
            all_preds_dgcnn.append(proba_dgcnn.cpu().numpy())
            all_targets.append(batch.y.cpu().numpy())
    
    y_pred_gravnet = np.concatenate(all_preds_gravnet)
    y_pred_dgcnn = np.concatenate(all_preds_dgcnn)
    y_true = np.concatenate(all_targets)
    
    # 生成Rejection Curve
    eff_gravnet, rej_gravnet, _ = compute_rejection_curve(y_true, y_pred_gravnet)
    eff_dgcnn, rej_dgcnn, _ = compute_rejection_curve(y_true, y_pred_dgcnn)
    
    # 对标CNN+DNN基线
    # （需要从先行研究获得基线数据）
    
    # 绘制对比图
    plt.figure(figsize=(12, 8))
    
    plt.loglog(eff_gravnet, rej_gravnet, 'b-', linewidth=2, label='GravNet')
    plt.loglog(eff_dgcnn, rej_dgcnn, 'g-', linewidth=2, label='DGCNN')
    # plt.loglog(eff_baseline, rej_baseline, 'r--', linewidth=2, label='CNN+DNN (Baseline)')
    
    plt.axhline(y=1e5, color='k', linestyle='--', alpha=0.5, label='Target (10⁵)')
    
    plt.xlabel('Signal Efficiency', fontsize=12)
    plt.ylabel('Rejection Power', fontsize=12)
    plt.title('Model Comparison: GravNet vs DGCNN', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=11)
    
    plt.tight_layout()
    plt.savefig('results/model_comparison.png', dpi=300)
    plt.show()
    
    print("✅ 性能评估完成")
    return {
        'gravnet': {'eff': eff_gravnet, 'rej': rej_gravnet},
        'dgcnn': {'eff': eff_dgcnn, 'rej': rej_dgcnn}
    }
```

---

## 阶段6：超参数优化（第2个月中旬～末）

### Step 6.1: 网格搜索超参数

**文件**：`scripts/hyperparam_search.py`

```bash
#!/bin/bash

# k-NN参数扫描
for K in 5 10 15 20; do
    echo "Testing k=$K"
    python << EOF
from src.train import train_model
model = train_model('gravnet', num_epochs=50, batch_size=64)
# 记录结果...
EOF
done

# 学习率扫描
for LR in 0.0001 0.001 0.01; do
    echo "Testing lr=$LR"
    python << EOF
from src.train import train_model
model = train_model('gravnet', num_epochs=50, lr=$LR)
EOF
done

# Focal Loss参数扫描
for GAMMA in 1.0 2.0 3.0; do
    for ALPHA in 0.1 0.25 0.5; do
        echo "Testing gamma=$GAMMA, alpha=$ALPHA"
        # 修改loss_fn，训练模型...
    done
done
```

---

## 阶段7：论文撰写（第3个月）

### Step 7.1: 整理图表和结果

**输出文件**：
```
results/
├── model_comparison.png          # 模型对比曲线
├── rejection_curve_gravnet.png   # GravNet性能曲线
├── rejection_curve_dgcnn.png     # DGCNN性能曲线
├── hyperparams_search.pdf        # 超参数扫描结果
├── final_metrics.txt             # 最终性能指标
└── model_gravnet.pth             # 最优模型权重
```

### Step 7.2: 论文结构

```markdown
# 機械学習を用いたGAPS実験における宇宙線粒子識別の研究

## 1. 研究背景と目的
- GAPS実験の概要
- 科学目標（暗黒物質探索）
- 粒子識別の課題（10⁴:1不平衡）
- GNNの適用性

## 2. 研究方法

### 2.1 データ処理とグラフ構築
- GEANT4シミュレーション
- Hit点の抽出
- ノード特徴量の定義
- k-NN辺の構築
- グラフデータへの変換

### 2.2 GNNモデルの実装
- GravNetの説明
- DGCNNの説明
- Focal Lossによるクラス不均衡対策
- 訓練パラメータ

### 2.3 性能評価方法
- Rejection Curve（効率 vs 拒否力）
- ベースラインとの比較
- 評価指標

## 3. 結果

### 3.1 GravNetの性能
- グラフ: Rejection Curve
- 数値: Rejection Power @ 95% Signal Efficiency

### 3.2 DGCNNの性能
- グラフ: Rejection Curve
- 数値: 最優秀性能

### 3.3 モデル比較
- 対比表: GravNet vs DGCNN
- 超パラメータ最適化結果

## 4. 考察

### 4.1 GNNの優位性
- CNN+DNNとの比較
- グラフ構造情報の利用効果
- 性能向上のメカニズム

### 4.2 Focal Lossの効果
- クラス不均衡問題への対策
- 少数派クラス（反重陽子）の識別向上

### 4.3 物理的解釈
- hit点の空間構造
- グラフ辺の物理的意味

## 5. 結論とFuture Work

### 5.1 主な成果
- GNNによる高精度粒子識別の実現
- Rejection Power ≥ 10⁵の達成状況

### 5.2 今後の方向
- 物理制約（Bethe-Bloch、TOF）の融合
- 飛迹再構成への拡張
- 実験データでの検証

## 参考文献
- GNN関連論文
- GAPS実験論文
```

---

## 📋 速查表：每周工作清单

### 第1～2周（5月1～14日）
- [ ] Step 1.1: GEANT4数据确认
- [ ] Step 1.2: 数据加载脚本完成
- [ ] Step 1.3: 数据分割
- [ ] Step 2.1: 图构造实现完成

### 第3～4周（5月15～28日）
- [ ] Step 2.2: 完整数据集类
- [ ] Step 2.3: 可视化验证
- [ ] Step 3.1～3.4: GNN模型和Focal Loss实现
- [ ] Step 4.1: 基本训练循环

### 第5～8周（6月1～28日）
- [ ] Step 4.1: 完整训练脚本
- [ ] Step 5.1～5.2: 性能评估和对比
- [ ] Step 6.1: 超参数优化
- [ ] 消融实验

### 第9～12周（7月1～24日）
- [ ] 最终模型验证
- [ ] 论文撰写（Step 7.2）
- [ ] 论文提交

---

**关键提醒**：
- ✅ 每完成一个Step就提交代码到git
- ✅ 每周生成至少一张evaluation图表
- ✅ 保持日志记录（loss曲线、超参数、结果）
- ✅ 定期与老师同步进度

---

## 项目文件结构

```
/workspace/gaps_project/
├── data/
│   ├── raw/                    # GEANT4原始数据（符号链接）
│   └── processed/              # 转换后的图数据
├── src/
│   ├── config.py               # 配置文件
│   ├── data_loader.py          # GEANT4数据加载
│   ├── graph_builder.py        # hit点→图结构
│   ├── utils/
│   │   ├── metrics.py          # Rejection Curve计算
│   │   └── visualization.py    # 绘图工具
│   ├── models/
│   │   ├── gnn_base.py         # 基类
│   │   ├── gravnet.py          # GravNet实现
│   │   └── dgcnn.py            # DGCNN实现
│   ├── train.py                # 训练脚本
│   └── evaluate.py             # 评估脚本
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   └── 02_results_analysis.ipynb
├── scripts/
│   ├── train.sh
│   └── evaluate.sh
├── results/                    # 性能结果、图表
├── logs/                       # 训练日志
├── config.yaml                 # 超参数配置
└── requirements.txt            # 环境依赖
```
