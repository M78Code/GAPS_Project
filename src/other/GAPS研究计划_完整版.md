# GAPS粒子识别研究计划 - 完整版

**论文题目**：
- 日语：機械学習を用いたGAPS実験における宇宙線粒子識別の研究
- 英语：Machine Learning-based Cosmic-Ray Particle Identification in the GAPS Experiment

**完成日期**：2026年4月24日  
**论文提交截止**：2026年7月24日  
**研究时间**：约3个月

---

## 一、研究背景与意义

### GAPS实验概述
- **全称**：General Anti-Particle Spectrometer（反粒子通用谱仪）
- **性质**：日美意国际共同实验计划
- **载体**：南极周回气球
- **科学目标**：通过探测宇宙线反重陽子，间接探索暗黑物质

### 核心科学问题
- **检出器系统**：
  - Si(Li)半导体检出器阵列（~1000个检出器素子）
  - Time-of-Flight (TOF)计数器（测速β）
  
- **极端类不平衡问题**：
  - 反陽子流量 : 反重陽子流量 ≈ **10⁴:1**
  - 高精度识别两者是反重陽子探索的关键

### 为什么选择GNN？

**先行研究局限**：
- CNN+DNN混合方法：将稀疏hit数据转换成规则3D grid
- **缺点**：转换过程丢失图结构信息，精度有瓶颈

**GNN优势**：
- 直接处理稀疏、不规则的hit点云数据
- 保留物理结构信息（邻域关系、距离）
- 特别适合极端类不平衡问题

---

## 二、研究方法

### （1）数据处理与图构造

**数据来源**：
- 研究组已有大量GEANT4蒙特卡洛模拟数据
- GEANT4模拟粒子通过检出器时的物理过程（能量损失、电离等）
- 数据包含真实标签（反陽子/反重陽子）

**数据转换流程**：
```
GEANT4原始数据 
  ↓
提取Si(Li)检出器hit信息
  ↓
图节点：hit点特征 [x, y, z, Energy, Time, ...]
  ↓
图边：k-nearest neighbors + 时间相关性
  ↓
图数据对象 (Graph, Label)
```

**节点特征**：
- 3D空间坐标（x, y, z）
- 能量损失（Energy Loss）
- 击中时间（Hit Time）
- 其他物理量

**边构造方法**：
- k-nearest neighbors（k = 5～15，待优化）
- 时间相关性（同时击中的hit点之间更强连接）
- 边权重：编码空间距离和时间信息

### （2）GNN模型实现

**模型架构**：
选择 **GravNet** 或 **DGCNN** 之一（两者对比）

| 架构 | 优势 | 特点 |
|------|------|------|
| **GravNet** | 动态学习图连接 | 适合不规则几何 |
| **DGCNN** | 多尺度特征提取 | Edge Convolution效果好 |

**分类任务**：
- 2分类：反陽子（y=0）vs 反重陽子（y=1）
- 输出层：2个神经元 + softmax

**训练配置**：
- **优化器**：Adam
- **初始学习率**：0.001（带衰减）
- **批量大小**：64
- **损失函数**：**Focal Loss**（处理10⁴:1不平衡）

**Focal Loss原理**：
- 普通交叉熵：$CE = -\log(p_t)$
- 焦点损失：$FL = -\alpha(1-p_t)^\gamma \log(p_t)$
- 作用：自动给少数类样本分配更高权重

### （3）性能评估

**主要指标**：**Rejection Curve**（拒绝曲线）

```
Rejection Power（拒绝力）
      ↑
      │
 10⁵  │ ←---- 目标
      │
 10³  │
      │
      │
  1   └──────────────→ Signal Efficiency（信号效率）
      0              1.0
```

**评估方法**：
- 横轴：信号效率（Signal Efficiency）= $\frac{TP}{TP+FN}$
- 纵轴：拒绝力（Rejection Power）= $\frac{1}{FPR}$，其中FPR为假正例率
- **目标**：Rejection Power ≥ $10^5$ @ 95%信号效率

**对标**：与先行研究的CNN+DNN基线对比

### （4）发展性研究（Future Work）

如果时间允许，融合物理约束：
- **Bethe-Bloch公式**：能量损失与粒子性质的物理关系
- **飞行时间一致性**：TOF测量的物理约束
- **目标**：改善对实验数据的泛化性能

---

## 三、时间表（3个月）

### 第1个月（5月）- 数据处理与基础模型

| 周次 | 具体任务 | 输出/检查点 |
|------|--------|-----------|
| 5.1-5.7 | 环境搭建、数据验证 | ✅ 可正常加载10个样本 |
| | | ✅ 确认标签比例（10⁴:1） |
| 5.8-5.14 | 图构造实现、可视化 | ✅ 能可视化hit点→图转换 |
| | | ✅ 验证节点/边数合理 |
| 5.15-5.21 | 基础模型框架、FocalLoss | ✅ 能跑通完整训练流程 |
| | | ✅ Loss正常下降 |
| 5.22-5.28 | GravNet实现、基线对标 | ✅ 生成Rejection Curve |
| | | ✅ 与CNN+DNN对比 |

**第1个月目标**：
- ✅ 完整的数据处理管道
- ✅ GravNet模型正常运行
- ✅ 有baseline性能指标

### 第2个月（6月）- 参数调优与DGCNN

| 周次 | 具体任务 | 输出/检查点 |
|------|--------|-----------|
| 6.1-6.7 | DGCNN实现、对比实验 | ✅ GravNet vs DGCNN性能对比 |
| 6.8-6.14 | 超参数优化 | ✅ k值（图构造）扫描结果 |
| | （k值、学习率、batch size等） | ✅ 学习率调度优化结果 |
| 6.15-6.21 | Focal Loss参数调优 | ✅ γ, α参数扫描 |
| | 不同dropout/regularization | ✅ 最佳配置确定 |
| 6.22-6.28 | 模型评估、消融实验 | ✅ 各模块贡献度分析 |

**第2个月目标**：
- ✅ Rejection Power显著提升（对标CNN+DNN）
- ✅ 最优超参数确定
- ✅ 模块级性能分析报告

### 第3个月（7月）- 验证与论文撰写

| 期间 | 具体任务 | 输出/检查点 |
|------|--------|-----------|
| 7.1-7.15 | 最终性能验证 | ✅ 在独立测试集验证Rejection Power |
| | | ✅ 物理约束探索（可选） |
| | | ✅ 模型鲁棒性测试 |
| 7.16-7.24 | 论文撰写、提交 | ✅ 完整论文PDF |
| | | ✅ 所有图表、结果 |

**第3个月目标**：
- ✅ 论文完成并提交（7月24日截止）
- ✅ 有充分的性能验证数据
- ✅ 清晰的研究贡献说明

---

## 四、当前技术环境

### Docker容器环境
```
4090 Ubuntu 主机
├── ai-train2 (pytorch:2.1.2-cuda11.8)
│   ├── ai-train3 (conda base环境)
│   └── gaps-env (GAPS项目虚拟环境) ← 当前使用
└── GPU: RTX 4090 (24GB显存)
```

### 已安装的关键库
| 库 | 版本 | 用途 |
|----|------|------|
| torch | 2.2.2 | 深度学习框架 |
| numpy | 1.26.2 | 数值计算 |
| pandas | 2.3.3 | 数据处理 |
| scipy | 1.15.3 | 科学计算 |
| matplotlib | 3.10.8 | 绘图 |
| tensorboard | 2.20.0 | 训练监控 |
| h5py | 3.15.1 | HDF5数据读写 |

### 待安装（GNN专用）
```bash
pip install torch-geometric torch-scatter torch-sparse torch-cluster
pip install scikit-learn seaborn pytorch-lightning wandb
```

---

## 五、项目文件结构

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

---

## 六、关键技术决策

### 1️⃣ 架构选择：GravNet vs DGCNN
- **方案**：两者都实现，通过对比确定最优
- **优先级**：先做GravNet（更适合不规则几何），再做DGCNN
- **评估**：基于Rejection Curve和性能对标

### 2️⃣ 图构造方法
- **节点**：Si(Li) hit点 + TOF信息
- **边**：k-nearest neighbors（k: 5/10/15扫描）+ 时间相关性
- **权重**：距离和时间的加权组合

### 3️⃣ 类不平衡处理
- **方案**：Focal Loss（α=0.25, γ=2.0，可调）
- **替代方案**：加权交叉熵、过采样（如需要）

### 4️⃣ 数据来源
- **使用现有GEANT4数据**（不重新生成）
- **节省时间**：避免2-3周的数据生成时间
- **后续扩展**：7月后可考虑改进detector geometry

---

## 七、预期成果与贡献

### 主要贡献
1. **技术贡献**：
   - 首次在GAPS实验中应用GNN进行粒子识别
   - 验证GNN相比CNN+DNN的优势
   - 定量分析极端类不平衡问题的解决方案

2. **性能指标**：
   - Rejection Power ≥ $10^5$（目标）
   - 相比先行研究的性能提升量化

3. **论文产出**：
   - 完整的学位论文（日语/中文/英文）
   - 技术细节完整、结果清晰、分析深入

### 后续可能的发展
- 物理约束融合（Bethe-Bloch、TOF一致性）
- 飞迹重建任务（如时间允许）
- 实验数据的模型应用（后续工作）

---

## 八、每日工作要点

### 第1个月核心任务顺序
```
周1  ← 环境 + 数据验证
周2  ← 图构造实现
周3  ← 基础模型框架
周4  ← GravNet + baseline对标
```

### 关键里程碑
- ✅ **5月7日前**：环境搭建完成，数据可加载
- ✅ **5月14日前**：图构造代码完成
- ✅ **5月21日前**：完整训练流程可运行
- ✅ **5月28日前**：GravNet性能有结果，与baseline对比完成

---

## 九、参考资源

### 核心论文
- **GravNet**: Qasim et al. (2019) - Distance-weighted graph networks
- **DGCNN**: Wang et al. (2019) - Dynamic Graph CNN
- **Focal Loss**: Lin et al. (2017) - Addressing Class Imbalance
- **GNN综述**: Thais et al. (2022) - GNN in Particle Physics

### 实现参考
- PyTorch Geometric官方文档
- 先行研究论文01（GAPS+ML）、02（反重陽子灵敏度）

### 项目工具
- **容器**：Docker (ai-train2 + gaps-env)
- **GPU**：RTX 4090 (24GB)
- **框架**：PyTorch + PyTorch Geometric
- **跟踪**：TensorBoard / Weights & Biases

---

## 十、常见问题Q&A

**Q: 为什么不重新生成GEANT4数据？**  
A: 省时间。研究组已有现成数据，生成新数据需要2-3周。三个月期限内，应该专注于模型研究。

**Q: GravNet和DGCNN先做哪个？**  
A: GravNet优先。它更适合不规则几何，论文也推荐。DGCNN作为对比。

**Q: 如何处理10⁴:1的不平衡？**  
A: Focal Loss。它在训练时自动给少数类更高权重，特别适合极端不平衡。

**Q: Rejection Power ≥ 10⁵可能达到吗？**  
A: 合理可行。GNN相比CNN+DNN的优势正是在处理不规则数据上，而GAPS恰好是不规则的hit pattern。先行论文也验证了类似的性能。

**Q: 时间不够怎么办？**  
A: 优先级：(1)数据处理→(2)GravNet→(3)性能评估→(4)DGCNN对比→(5)物理约束。前三个是必须的。

---

**更新日期**：2026年4月24日  
**状态**：环境搭建中，待GEANT4数据位置确认后启动第1阶段
