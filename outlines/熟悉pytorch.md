# 主题：熟悉PyTorch

---

## 章节1：PyTorch基础与生态

### 小节1.1：PyTorch简介与框架对比

#### 子小节1.1.1：PyTorch发展历程与特点

**PyTorch的核心设计理念**
- PyTorch相比其他深度学习框架有哪些独特的设计哲学？
- 动态计算图和静态计算图各有什么优缺点？
- PyTorch的"Python-first"设计理念体现在哪些方面？
- PyTorch 2.0引入了哪些重大改进？

**PyTorch的生态系统**
- PyTorch生态中有哪些重要的子项目（如torchvision、torchaudio等）？
- PyTorch Hub的作用是什么？如何使用？
- PyTorch Lightning和原生PyTorch的关系是什么？各自的使用场景？
- Hugging Face Transformers库为什么选择PyTorch作为主要后端？

#### 子小节1.1.2：PyTorch与其他框架对比

**PyTorch vs TensorFlow**
- PyTorch和TensorFlow 2.x在编程范式上有什么区别？
- Eager Execution模式下，两者的性能差异如何？
- 在生产部署方面，两者各有什么优势？
- 如何在PyTorch和TensorFlow之间迁移模型？

**PyTorch vs JAX**
- JAX的函数式编程范式与PyTorch的面向对象范式有何不同？
- JAX的自动向量化(vmap)和PyTorch的实现方式有什么区别？
- 在科学计算场景下，两者各有什么优势？

**PyTorch vs MXNet/PaddlePaddle**
- MXNet的混合编程模式是什么？与PyTorch有何异同？
- PaddlePaddle在中文NLP任务上有什么优势？

### 小节1.2：环境配置与版本管理

#### 子小节1.2.1：安装与配置

**PyTorch安装方式**
- 如何根据CUDA版本选择合适的PyTorch安装命令？
- conda安装和pip安装PyTorch有什么区别？
- 如何在同一环境中管理多个PyTorch版本？
- 源码编译PyTorch需要注意哪些事项？

**CUDA与cuDNN配置**
- PyTorch如何检测和使用GPU？
- torch.cuda.is_available()返回False可能是哪些原因？
- CUDA版本与PyTorch版本不兼容会导致什么问题？
- 如何在多GPU环境中指定使用特定GPU？

#### 子小节1.2.2：版本兼容性

**版本依赖管理**
- PyTorch的版本号命名规则是什么？
- 如何查看当前PyTorch版本及其依赖的CUDA版本？
- PyTorch模型在不同版本间的兼容性如何保证？
- 如何处理PyTorch升级后的API breaking changes？

**硬件兼容性**
- PyTorch支持哪些类型的加速器（GPU、TPU、NPU等）？
- 在ARM架构上运行PyTorch需要注意什么？
- PyTorch对AMD GPU的支持情况如何？

---

## 章节2：张量(Tensor)操作

### 小节2.1：张量基础

#### 子小节2.1.1：张量创建与属性

**张量创建方法**
- 列举至少5种创建张量的方法（torch.tensor、torch.zeros等）
- torch.tensor()和torch.Tensor()有什么区别？
- 如何从NumPy数组创建张量？创建后是否共享内存？
- torch.empty()创建的张量内容是什么？为什么要使用它？

**张量的基本属性**
- 张量的shape、dtype、device、layout分别表示什么？
- 如何查看张量是否需要梯度（requires_grad）？
- 张量的stride是什么？如何理解内存布局？
- 什么是contiguous tensor？为什么需要关注连续性？

**张量数据类型**
- PyTorch支持哪些数据类型？默认的浮点类型是什么？
- 如何进行张量的数据类型转换？
- 半精度（fp16）和单精度（fp32）的使用场景和转换方法？
- bfloat16与float16有什么区别？

#### 子小节2.1.2：张量的视图与克隆

**视图操作**
- view()和reshape()的区别是什么？
- 什么情况下view()会失败？如何解决？
- transpose()和permute()的区别和使用场景？
- squeeze()和unsqueeze()的作用是什么？

**张量复制**
- clone()和detach()的区别是什么？
- 浅拷贝和深拷贝在PyTorch中如何实现？
- detach()操作对计算图有什么影响？
- 如何既分离计算图又复制数据？

### 小节2.2：张量运算

#### 子小节2.2.1：数学运算

**基本数学操作**
- PyTorch中的广播机制(Broadcasting)是如何工作的？
- 逐元素运算(element-wise)和矩阵运算有什么区别？
- torch.mm()、torch.matmul()、@运算符的区别是什么？
- einsum()函数的使用场景和优势是什么？

**归约运算**
- sum()、mean()、max()等归约操作如何指定维度？
- keepdim参数的作用是什么？
- argmax()和max()的返回值有什么不同？
- 如何计算张量的范数（norm）？

**高级数学运算**
- torch.where()的三种使用方式分别是什么？
- torch.gather()和torch.scatter()的作用和使用场景？
- 如何使用masked_select()和masked_fill()？
- topk()函数的返回值是什么？

#### 子小节2.2.2：索引与切片

**基本索引**
- PyTorch的索引方式与NumPy有什么异同？
- 如何使用布尔索引选择满足条件的元素？
- 高级索引和基本索引的区别是什么？
- index_select()与直接索引有什么区别？

**切片操作**
- 切片操作是否会创建新的张量？
- 如何实现跨维度的复杂切片？
- ...（省略号）在索引中的作用是什么？
- 负索引的使用规则是什么？

#### 子小节2.2.3：张量变形

**形状操作**
- flatten()和ravel()的区别是什么？
- 如何理解-1在reshape中的作用？
- repeat()和expand()的区别和性能差异？
- tile()函数的使用场景是什么？

**维度操作**
- 如何交换张量的两个维度？
- torch.cat()和torch.stack()的区别？
- chunk()和split()的区别和使用场景？
- unbind()函数的作用是什么？

### 小节2.3：设备管理

#### 子小节2.3.1：CPU与GPU操作

**设备转换**
- 如何将张量在CPU和GPU之间转移？
- .cuda()、.cpu()、.to(device)三种方式的区别？
- 跨设备操作（如CPU张量和GPU张量相加）会发生什么？
- 如何批量将模型和数据移动到GPU？

**多GPU管理**
- 如何查看当前系统中有多少个可用GPU？
- torch.cuda.device()上下文管理器的作用是什么？
- 如何在不同GPU上创建张量？
- CUDA_VISIBLE_DEVICES环境变量如何使用？

#### 子小节2.3.2：设备同步与通信

**同步操作**
- torch.cuda.synchronize()的作用是什么？何时需要调用？
- 为什么GPU计算是异步的？这对性能有什么影响？
- 如何正确测量GPU代码的执行时间？
- Stream的概念是什么？如何使用CUDA Streams？

**设备间通信**
- 如何在不同GPU之间复制张量？
- 点对点通信(P2P)是什么？如何启用？
- NCCL在PyTorch中的作用是什么？
- GPU Direct和NVLink如何提升多GPU通信效率？

### 小节2.4：内存管理

#### 子小节2.4.1：内存分配与释放

**内存分配机制**
- PyTorch的内存分配器(caching allocator)是如何工作的？
- torch.cuda.empty_cache()的作用是什么？何时使用？
- 为什么GPU显存占用与实际使用不一致？
- 如何监控和分析PyTorch的内存使用？

**内存泄漏**
- PyTorch中常见的内存泄漏场景有哪些？
- 如何避免在循环中累积计算图导致的内存泄漏？
- 使用.item()和不使用有什么区别？
- 如何检测和定位内存泄漏问题？

#### 子小节2.4.2：内存优化

**显存优化技巧**
- 梯度累积(Gradient Accumulation)如何减少显存占用？
- 混合精度训练如何节省显存？
- Checkpoint技术的原理和使用方法是什么？
- inplace操作如何节省内存？有哪些注意事项？

**张量共享与复用**
- 多个张量何时会共享底层存储？
- storage()方法的作用是什么？
- as_strided()函数如何实现零拷贝视图？
- 如何判断两个张量是否共享内存？

---

## 章节3：自动微分机制(Autograd)

### 小节3.1：计算图原理

#### 子小节3.1.1：动态计算图

**计算图构建**
- PyTorch的动态计算图是如何构建的？
- 前向传播时计算图是如何记录操作的？
- 什么是Function对象？它在计算图中的作用是什么？
- 叶子节点(leaf node)和非叶子节点的区别是什么？

**计算图的生命周期**
- 计算图在什么时候被创建和销毁？
- 为什么默认情况下backward()后计算图会被释放？
- retain_graph=True的使用场景是什么？
- 如何多次对同一个计算图求导？

#### 子小节3.1.2：自动微分原理

**反向传播机制**
- autograd是如何实现自动求导的？
- 链式法则在autograd中如何体现？
- 雅可比矩阵和向量积(Jacobian-vector product)是什么？
- 为什么PyTorch使用反向模式自动微分而不是正向模式？

**梯度计算细节**
- grad_fn属性的含义是什么？
- 如何查看一个张量的梯度函数链？
- 标量和非标量张量backward()的区别是什么？
- backward()中的grad_tensors参数有什么作用？

### 小节3.2：梯度计算与管理

#### 子小节3.2.1：梯度的计算与访问

**requires_grad属性**
- requires_grad的作用和设置方法？
- 如何冻结部分网络层的参数？
- with torch.no_grad()和@torch.no_grad()的作用？
- torch.set_grad_enabled()的使用场景？

**梯度访问与操作**
- 如何访问张量的梯度？
- 为什么非叶子节点的梯度会被自动释放？
- 如何保留中间变量的梯度？
- register_hook()的作用和使用方法？

#### 子小节3.2.2：梯度管理

**梯度清零**
- 为什么需要手动清零梯度？
- optimizer.zero_grad()和model.zero_grad()的区别？
- set_to_none=True参数的作用是什么？
- 梯度不清零会导致什么问题？

**梯度裁剪**
- 为什么需要梯度裁剪？
- torch.nn.utils.clip_grad_norm_()和clip_grad_value_()的区别？
- 如何实现自适应梯度裁剪？
- 梯度裁剪对训练稳定性的影响是什么？

**梯度累积**
- 梯度累积的原理是什么？
- 如何正确实现梯度累积？
- 梯度累积对BatchNorm的影响是什么？
- 梯度累积与增大batch size的等价性是什么？

### 小节3.3：高级自动微分

#### 子小节3.3.1：高阶导数

**二阶导数计算**
- 如何计算二阶导数？
- create_graph参数的作用是什么？
- 计算Hessian矩阵的方法有哪些？
- 高阶导数的应用场景有哪些？

**雅可比矩阵和Hessian矩阵**
- torch.autograd.functional.jacobian()如何使用？
- 如何高效计算Hessian向量积？
- vmap在计算雅可比矩阵中的作用？
- functorch库提供了哪些高级自动微分功能？

#### 子小节3.3.2：自定义autograd Function

**Function类的实现**
- 如何继承torch.autograd.Function实现自定义操作？
- forward()和backward()方法的签名规则是什么？
- ctx对象的作用是什么？常用的方法有哪些？
- save_for_backward()和ctx.save_for_backward()的使用？

**自定义Function的应用**
- 为什么需要自定义Function？典型应用场景有哪些？
- 如何在自定义Function中处理不可导的操作？
- Straight-Through Estimator是什么？如何实现？
- 如何为自定义Function编写梯度检查测试？

**注意事项与调试**
- torch.autograd.gradcheck()如何使用？
- 自定义Function中的常见错误有哪些？
- 如何调试自定义Function的梯度计算？
- once_differentiable装饰器的作用是什么？

### 小节3.4：上下文管理与控制

#### 子小节3.4.1：梯度上下文

**no_grad与enable_grad**
- torch.no_grad()、torch.enable_grad()、torch.set_grad_enabled()的区别？
- inference_mode()和no_grad()的区别是什么？
- 在no_grad()上下文中创建的requires_grad=True张量会怎样？
- 如何在推理时避免不必要的梯度计算开销？

**梯度模式切换**
- 如何在训练和推理模式间切换？
- model.eval()对autograd有什么影响？
- 评估模式下的Dropout和BatchNorm行为如何变化？
- torch.jit.script下的autograd行为有何不同？

#### 子小节3.4.2：异常检测与调试

**梯度异常检测**
- torch.autograd.set_detect_anomaly()的作用？
- 如何定位梯度爆炸或梯度消失的位置？
- NaN梯度的常见原因和调试方法？
- 如何使用hook检查梯度值？

**计算图可视化**
- 如何可视化PyTorch的计算图？
- make_dot工具的使用方法？
- TensorBoard如何展示计算图？
- 如何理解和分析复杂的计算图结构？

---

## 章节4：神经网络模块(nn.Module)

### 小节4.1：Module设计模式

#### 子小节4.1.1：Module基础

**Module类的核心概念**
- nn.Module的作用是什么？为什么需要它？
- Module的子模块和参数是如何管理的？
- __init__()和forward()方法的设计原则是什么？
- Module实例可以像函数一样调用的原理是什么(__call__方法)？

**参数管理**
- Parameter和普通Tensor的区别是什么？
- named_parameters()和parameters()的区别和使用场景？
- buffer和parameter的区别是什么？何时使用buffer？
- register_parameter()和register_buffer()的作用？

**模块注册**
- 子模块是如何被自动注册的？
- 使用列表和字典存储子模块有什么问题？如何解决？
- ModuleList和ModuleDict的作用和使用方法？
- ParameterList和ParameterDict的使用场景？

#### 子小节4.1.2：Module的状态与行为

**训练与评估模式**
- train()和eval()方法做了什么？
- training属性如何影响模块的行为？
- 哪些层在训练和评估时行为不同？
- 如何实现自定义层的模式切换行为？

**状态字典**
- state_dict()返回的是什么？
- load_state_dict()的strict参数作用是什么？
- 如何只加载部分预训练权重？
- 保存state_dict和保存整个模型有什么区别？

**模块钩子(Hooks)**
- register_forward_hook()的作用和使用场景？
- register_forward_pre_hook()和register_full_backward_hook()的区别？
- hook函数的签名是什么？返回值有何影响？
- 如何使用hook实现特征提取或梯度监控？

### 小节4.2：常用层与组件

#### 子小节4.2.1：线性层与卷积层

**全连接层**
- nn.Linear的参数和计算过程？
- bias参数的作用？什么情况下可以禁用bias？
- 如何初始化Linear层的权重？
- Linear层的权重矩阵形状是(in_features, out_features)还是反过来？

**卷积层**
- nn.Conv2d的各个参数(kernel_size, stride, padding等)含义？
- padding='same'和padding='valid'的区别？
- groups参数的作用？如何实现深度可分离卷积？
- dilation参数的作用和应用场景？

**卷积变体**
- 转置卷积(ConvTranspose2d)的原理和用途？
- 深度可分离卷积如何用PyTorch实现？
- 1x1卷积的作用是什么？
- 3D卷积与2D卷积的区别和应用场景？

#### 子小节4.2.2：归一化层

**Batch Normalization**
- BatchNorm的原理和作用是什么？
- BatchNorm在训练和推理时的行为差异？
- running_mean和running_var是如何更新的？
- momentum参数的含义是什么？

**其他归一化方法**
- LayerNorm、InstanceNorm、GroupNorm的区别和使用场景？
- 为什么Transformer使用LayerNorm而不是BatchNorm？
- RMSNorm相比LayerNorm有什么优势？
- 如何选择合适的归一化方法？

**归一化的实现细节**
- affine参数的作用是什么？
- track_running_stats参数何时设为False？
- BatchNorm的num_batches_tracked参数的作用？
- 如何冻结BatchNorm层？

#### 子小节4.2.3：激活函数

**常用激活函数**
- ReLU、LeakyReLU、PReLU、ELU的区别和特点？
- Sigmoid和Tanh的优缺点和使用场景？
- GELU和Swish(SiLU)激活函数的特点？
- Mish激活函数为什么在某些任务上效果更好？

**激活函数的选择**
- 如何为不同任务选择合适的激活函数？
- inplace=True参数的作用和注意事项？
- 激活函数对梯度流动的影响是什么？
- 死亡ReLU问题是什么？如何缓解？

#### 子小节4.2.4：池化与采样

**池化层**
- MaxPool和AvgPool的区别和使用场景？
- AdaptiveAvgPool和普通AvgPool的区别？
- Global Average Pooling的作用是什么？
- return_indices参数在MaxPool中的作用？

**采样层**
- Upsample和ConvTranspose2d的区别？
- 上采样的不同插值方法(nearest, bilinear等)的特点？
- PixelShuffle的原理和应用？
- 如何实现可学习的上采样？

#### 子小节4.2.5：循环层与注意力

**循环神经网络层**
- nn.RNN、nn.LSTM、nn.GRU的区别？
- batch_first参数的作用？
- bidirectional参数如何影响输出形状？
- hidden state和cell state的区别是什么(LSTM)？

**注意力机制**
- nn.MultiheadAttention的使用方法？
- Self-Attention和Cross-Attention的区别？
- attention mask的作用和使用方法？
- key_padding_mask和attn_mask的区别？

**Transformer组件**
- TransformerEncoderLayer和TransformerDecoderLayer的区别？
- 如何使用nn.Transformer构建完整的Transformer模型？
- 位置编码应该如何添加？
- 因果mask如何实现？

### 小节4.3：容器与组合

#### 子小节4.3.1：Sequential与容器

**Sequential容器**
- nn.Sequential的作用和使用方法？
- 如何在Sequential中添加不同类型的层？
- Sequential的局限性是什么？
- 如何访问Sequential中的特定层？

**ModuleList与ModuleDict**
- ModuleList和Python list的区别？
- 何时应该使用ModuleDict？
- 如何遍历ModuleList中的模块？
- ParameterList的使用场景？

#### 子小节4.3.2：自定义模块

**自定义层的实现**
- 如何实现一个自定义的神经网络层？
- 可学习参数应该如何定义和注册？
- extra_repr()方法的作用是什么？
- 如何在自定义模块中实现权重初始化？

**模块组合模式**
- 如何组合多个模块构建复杂网络？
- ResNet的残差连接如何实现？
- DenseNet的dense连接如何实现？
- 如何实现条件执行的模块（如分支结构）？

### 小节4.4：损失函数

#### 子小节4.4.1：分类损失

**交叉熵损失**
- nn.CrossEntropyLoss的输入输出格式是什么？
- CrossEntropyLoss内部包含了Softmax吗？
- label smoothing如何在CrossEntropyLoss中实现？
- weight参数的作用和使用场景？

**其他分类损失**
- BCELoss和BCEWithLogitsLoss的区别？
- NLLLoss和CrossEntropyLoss的关系？
- MultiLabelSoftMarginLoss的使用场景？
- Focal Loss如何在PyTorch中实现？

#### 子小节4.4.2：回归与重构损失

**回归损失**
- MSELoss、L1Loss、SmoothL1Loss的区别和特点？
- Huber Loss相比MSE的优势是什么？
- 如何选择合适的回归损失函数？
- reduction参数('mean', 'sum', 'none')的作用？

**重构与度量损失**
- CosineEmbeddingLoss的作用和使用方法？
- TripletMarginLoss的原理和应用场景？
- ContrastiveLoss如何实现？
- KL散度损失的使用场景（如VAE）？

#### 子小节4.4.3：自定义损失函数

**损失函数设计**
- 如何实现自定义损失函数？
- 复合损失函数如何设计和平衡权重？
- 如何在损失函数中处理mask或ignore index？
- 损失函数的数值稳定性如何保证？

---

## 章节5：数据处理

### 小节5.1：Dataset与数据加载

#### 子小节5.1.1：Dataset类

**Dataset基础**
- torch.utils.data.Dataset的作用是什么？
- Map-style Dataset和Iterable-style Dataset的区别？
- 如何实现自定义Dataset？必须实现哪些方法？
- __getitem__()和__len__()方法的设计要点？

**内置Dataset**
- TensorDataset的使用场景和限制？
- ConcatDataset和Subset的作用？
- ChainDataset和StackDataset的区别？
- 如何使用random_split()划分数据集？

**数据集组合与变换**
- 如何合并多个数据集？
- 如何对数据集进行子集选择？
- WeightedRandomSampler的使用场景？
- 如何实现数据集的缓存机制？

#### 子小节5.1.2：DataLoader

**DataLoader基础**
- DataLoader的各个参数(batch_size, shuffle, num_workers等)的作用？
- collate_fn参数的作用？如何自定义？
- pin_memory参数为什么能加速GPU训练？
- drop_last参数的使用场景？

**多进程数据加载**
- num_workers设置为多少合适？
- 多进程加载时的内存占用问题如何解决？
- worker_init_fn的作用和使用场景？
- persistent_workers参数的作用？

**采样器(Sampler)**
- Sampler的作用是什么？
- RandomSampler、SequentialSampler、WeightedRandomSampler的区别？
- 如何实现自定义采样器？
- BatchSampler和普通Sampler的区别？

### 小节5.2：数据预处理与增强

#### 子小节5.2.1：torchvision.transforms

**基本变换**
- 常用的图像变换有哪些(Resize, Crop, Normalize等)？
- Compose的作用是什么？
- ToTensor()做了什么操作？
- Normalize的mean和std参数如何确定？

**数据增强**
- RandomCrop和CenterCrop的区别？
- RandomHorizontalFlip的概率如何控制？
- ColorJitter的各个参数的含义？
- RandomAffine和RandomPerspective的使用场景？

**高级增强技术**
- AutoAugment策略是什么？
- RandAugment和AutoAugment的区别？
- Mixup和CutMix如何实现？
- 如何实现自定义的数据增强操作？

#### 子小节5.2.2：数据预处理策略

**归一化策略**
- 为什么要对数据进行归一化？
- 不同的归一化方法(Min-Max, Z-score)的选择？
- 如何计算数据集的均值和方差？
- 数据增强应该在归一化前还是后？

**数据加载优化**
- 如何在数据加载时进行on-the-fly预处理？
- 预处理应该在Dataset还是transform中实现？
- 如何平衡CPU预处理和GPU计算的速度？
- 数据预加载(prefetch)的实现方法？

### 小节5.3：分布式数据处理

#### 子小节5.3.1：分布式采样

**DistributedSampler**
- DistributedSampler的作用是什么？
- 如何在分布式训练中使用DistributedSampler？
- shuffle参数在分布式训练中如何设置？
- set_epoch()方法为什么必须调用？

**数据划分策略**
- 如何保证每个进程加载不同的数据？
- drop_last在分布式训练中的重要性？
- 如何处理数据集大小不能被进程数整除的情况？
- 如何实现自定义的分布式采样策略？

#### 子小节5.3.2：数据并行加载

**多GPU数据加载**
- 每个GPU都需要独立的DataLoader吗？
- 如何避免不同进程加载重复数据？
- 分布式训练时的有效batch size如何计算？
- 如何在多机多卡环境下正确加载数据？

---

## 章节6：训练与优化

### 小节6.1：优化器

#### 子小节6.1.1：基础优化器

**SGD优化器**
- SGD的参数(lr, momentum, weight_decay)的作用？
- momentum和nesterov的区别？
- weight_decay在SGD中是如何实现的？
- 批量大小对SGD的影响是什么？

**Adam系列**
- Adam的两个动量参数(beta1, beta2)的含义？
- AdamW和Adam的区别是什么？
- AMSGrad变体的改进点是什么？
- Adam在什么情况下效果不如SGD？

**其他优化器**
- RMSprop的原理和适用场景？
- Adagrad的优缺点是什么？
- AdamW、Adafactor、LAMB的区别和应用？
- 如何为不同的任务选择合适的优化器？

#### 子小节6.1.2：优化器使用

**参数组**
- 如何为不同的参数设置不同的学习率？
- param_groups的结构和使用方法？
- 如何为特定层冻结或调整学习率？
- 差异化学习率在迁移学习中的应用？

**优化器状态**
- optimizer.state_dict()包含哪些信息？
- 如何保存和加载优化器状态？
- 优化器状态在断点续训中的作用？
- 如何清空优化器的动量等状态？

**梯度更新**
- step()方法做了什么？
- zero_grad()应该在什么时候调用？
- 梯度累积时优化器如何使用？
- 如何实现梯度的手动调整？

### 小节6.2：学习率调度

#### 子小节6.2.1：学习率调度器

**基本调度器**
- StepLR、MultiStepLR、ExponentialLR的区别？
- CosineAnnealingLR的原理和参数设置？
- ReduceLROnPlateau的使用方法和触发机制？
- LinearLR和ConstantLR的使用场景？

**组合调度器**
- SequentialLR如何组合多个调度器？
- ChainedScheduler和SequentialLR的区别？
- Warmup如何与其他调度器结合？
- 如何实现先warmup后cosine衰减？

**自定义调度器**
- 如何实现自定义的学习率调度策略？
- LambdaLR如何使用？
- get_lr()和get_last_lr()的区别？
- 如何实现周期性学习率（Cyclic LR）？

#### 子小节6.2.2：学习率策略

**Warmup策略**
- 为什么需要学习率预热(Warmup)？
- 线性warmup和指数warmup的区别？
- warmup的步数如何确定？
- 大batch训练时warmup的重要性？

**调度策略选择**
- 不同任务应该选择什么样的学习率调度策略？
- One-Cycle策略是什么？
- 学习率衰减对模型收敛的影响？
- 如何通过学习率调度避免过拟合？

### 小节6.3：训练循环与技巧

#### 子小节6.3.1：标准训练流程

**训练循环设计**
- 一个标准的训练循环包含哪些步骤？
- 训练和验证循环的区别是什么？
- 如何正确地切换模型的训练和评估模式？
- 在训练循环中如何正确处理异常？

**批处理训练**
- 如何从DataLoader中获取数据？
- 输入数据和标签应该如何处理（如移动到GPU）？
- 批次训练中的维度处理注意事项？
- 如何处理最后一个不完整的batch？

**性能监控**
- 如何记录和监控训练过程中的指标？
- 训练loss、验证loss、准确率如何统计？
- 如何使用tqdm显示训练进度？
- TensorBoard如何集成到训练循环中？

#### 子小节6.3.2：训练技巧

**混合精度训练**
- torch.cuda.amp的作用是什么？
- GradScaler的工作原理？
- autocast上下文管理器的使用方法？
- 哪些操作不适合使用fp16？

**梯度累积**
- 如何实现梯度累积？
- 梯度累积对BatchNorm的影响如何处理？
- 损失值在梯度累积时如何正确计算？
- 梯度累积与大batch size的等价性？

**早停与正则化**
- Early Stopping如何实现？
- 如何监控验证集性能避免过拟合？
- Dropout在训练和评估时的行为差异？
- L1、L2正则化如何在PyTorch中实现？

**模型集成**
- 如何实现模型的Ensemble？
- Snapshot Ensemble是什么？
- Exponential Moving Average(EMA)如何应用于模型参数？
- 多模型投票策略如何实现？

### 小节6.4：评估与指标

#### 子小节6.4.1：评估流程

**模型评估**
- 评估模式下需要注意哪些设置？
- torch.no_grad()在评估中的作用？
- 如何批量评估大规模数据集？
- 评估时的batch size如何选择？

**指标计算**
- 如何在训练过程中计算准确率？
- top-k准确率如何计算？
- 多分类和多标签任务的指标有什么区别？
- 如何使用torchmetrics库计算指标？

#### 子小节6.4.2：验证与测试

**交叉验证**
- 如何在PyTorch中实现K折交叉验证？
- 分层采样在交叉验证中的应用？
- 交叉验证与随机划分的区别？
- 如何保证可重复的数据划分？

**测试集评估**
- 训练集、验证集、测试集的作用区别？
- 如何避免在测试集上过度调参？
- 如何正确报告模型性能？
- 统计显著性检验在模型比较中的应用？

---

## 章节7：模型管理

### 小节7.1：模型保存与加载

#### 子小节7.1.1：保存机制

**模型保存方式**
- torch.save()和pickle的关系？
- 保存整个模型vs保存state_dict的区别？
- 推荐的模型保存方式是什么？为什么？
- 如何保存模型架构和权重？

**Checkpoint保存**
- 完整的checkpoint应该包含哪些内容？
- 如何在checkpoint中保存epoch、optimizer等信息？
- 保存checkpoint的文件格式选择（.pt, .pth, .bin）？
- 如何实现自动保存最佳模型？

**保存策略**
- 如何只保存最近的N个checkpoint？
- 按验证集性能保存vs按epoch保存的区别？
- 如何避免保存过程中的文件损坏？
- 大模型保存时的内存和磁盘优化？

#### 子小节7.1.2：加载机制

**模型加载**
- torch.load()的使用方法和参数？
- map_location参数的作用？
- 如何加载在GPU上训练的模型到CPU？
- weights_only参数的安全性考虑？

**部分加载**
- strict=False时会发生什么？
- 如何加载不完全匹配的权重？
- 如何处理模型结构变化后的权重加载？
- 迁移学习时如何只加载部分层的权重？

**加载异常处理**
- 加载失败的常见原因有哪些？
- 如何处理版本不兼容的模型文件？
- 权重维度不匹配如何调试？
- 如何验证加载的权重是否正确？

### 小节7.2：模型导出与部署

#### 子小节7.2.1：ONNX导出

**ONNX基础**
- ONNX是什么？为什么需要它？
- torch.onnx.export()的使用方法？
- dynamic_axes参数的作用？
- 如何处理导出时的动态形状？

**导出注意事项**
- 哪些PyTorch操作不支持ONNX导出？
- 如何验证导出的ONNX模型？
- opset_version参数如何选择？
- 控制流（if/for）如何处理？

**ONNX Runtime**
- 如何使用ONNX Runtime进行推理？
- ONNX Runtime相比PyTorch的性能优势？
- 如何优化ONNX模型？
- 量化和图优化在ONNX中如何应用？

#### 子小节7.2.2：TorchScript

**TorchScript基础**
- TorchScript是什么？与Python代码的区别？
- torch.jit.script和torch.jit.trace的区别？
- 何时使用script，何时使用trace？
- TorchScript的限制和不支持的Python特性？

**模型脚本化**
- 如何将模型转换为TorchScript？
- 类型注解在TorchScript中的重要性？
- 如何处理trace时的分支和循环？
- 如何调试TorchScript模型？

**TorchScript部署**
- 如何保存和加载TorchScript模型？
- TorchScript在C++中的使用？
- 移动端部署(PyTorch Mobile)如何使用TorchScript？
- TorchScript的性能优化技巧？

### 小节7.3：预训练模型与迁移学习

#### 子小节7.3.1：预训练模型使用

**加载预训练模型**
- torchvision.models如何使用？
- pretrained参数和weights参数的区别？
- 如何从Hugging Face加载预训练模型？
- 如何处理预训练模型的输入要求（尺寸、归一化等）？

**模型修改**
- 如何替换预训练模型的最后一层？
- 如何修改模型的输入通道数？
- 如何为预训练模型添加新的层？
- 如何处理类别数不匹配的问题？

#### 子小节7.3.2：微调策略

**参数冻结**
- 如何冻结模型的部分层？
- requires_grad_()方法的使用？
- 冻结BatchNorm层的最佳实践？
- 渐进式解冻(Gradual Unfreezing)如何实现？

**学习率策略**
- 微调时的学习率如何设置？
- 差异化学习率在微调中的应用？
- discriminative fine-tuning是什么？
- 如何为不同层设置不同的学习率？

**微调技巧**
- 微调需要训练多少个epoch？
- 数据量较小时的微调策略？
- 如何避免微调时的灾难性遗忘？
- 特征提取vs微调的选择依据？

---

## 章节8：分布式训练

### 小节8.1：数据并行

#### 子小节8.1.1：DataParallel

**DP基础**
- nn.DataParallel的工作原理？
- 如何使用DataParallel包装模型？
- DataParallel的性能瓶颈是什么？
- device_ids参数的作用？

**DP的局限性**
- 为什么DataParallel不推荐使用？
- 单进程多线程带来的问题？
- GIL对DataParallel性能的影响？
- 负载不均衡问题如何产生？

#### 子小节8.1.2：DistributedDataParallel

**DDP基础**
- DDP的工作原理和架构？
- DDP相比DP的优势是什么？
- 如何初始化分布式环境（init_process_group）？
- backend参数(nccl, gloo, mpi)如何选择？

**DDP使用**
- 如何用DDP包装模型？
- find_unused_parameters参数的作用？
- broadcast_buffers参数何时设为False？
- DDP的梯度同步机制是什么？

**启动分布式训练**
- torch.distributed.launch和torchrun的区别？
- RANK、LOCAL_RANK、WORLD_SIZE等环境变量的含义？
- 如何在多机多卡环境下启动训练？
- init_method参数的不同设置方式？

### 小节8.2：分布式通信

#### 子小节8.2.1：通信原语

**集合通信操作**
- all_reduce、broadcast、gather、scatter的区别和用途？
- 何时需要手动调用分布式通信操作？
- reduce_op参数的可选值（SUM, MEAN等）？
- 异步通信如何实现？

**通信后端**
- NCCL的特点和适用场景？
- Gloo和NCCL的性能差异？
- CPU训练应该用哪个后端？
- 如何选择和配置通信后端？

#### 子小节8.2.2：同步与协调

**进程同步**
- barrier()的作用和使用场景？
- 如何保证所有进程在相同的状态？
- 主进程和子进程的协调方式？
- 如何在分布式训练中处理随机性？

**数据一致性**
- 如何保证不同进程的模型初始化一致？
- 随机数种子在分布式训练中如何设置？
- BatchNorm的统计量如何在多进程间同步？
- 如何验证分布式训练的正确性？

### 小节8.3：混合精度与优化

#### 子小节8.3.1：混合精度训练

**自动混合精度(AMP)**
- 混合精度训练的原理和优势？
- GradScaler如何防止梯度下溢？
- autocast如何自动选择精度？
- 哪些操作会保持fp32精度？

**混合精度配置**
- 如何在分布式训练中使用混合精度？
- DDP + AMP的正确使用方式？
- 梯度缩放因子如何动态调整？
- 混合精度训练的数值稳定性问题？

#### 子小节8.3.2：通信优化

**梯度通信优化**
- 梯度bucketing是什么？
- 如何调整DDP的bucket_cap_mb参数？
- 梯度压缩技术有哪些？
- ZeRO优化器的原理和使用？

**内存优化**
- 激活值重计算(Activation Checkpointing)的原理？
- 如何在分布式训练中使用gradient checkpointing？
- ZeRO-Offload如何将部分计算卸载到CPU？
- 大模型训练的内存优化策略？

### 小节8.4：高级分布式技术

#### 子小节8.4.1：模型并行

**张量并行**
- 张量并行和数据并行的区别？
- 如何手动实现简单的张量并行？
- Megatron-LM的张量并行策略？
- 通信开销如何影响张量并行效率？

**流水线并行**
- 流水线并行的原理和实现？
- GPipe和PipeDream的区别？
- 如何使用PyTorch的pipeline parallel？
- bubble time如何减少？

**混合并行**
- 3D并行(数据+张量+流水线)是什么？
- 如何为超大模型设计并行策略？
- DeepSpeed和FairScale的作用？
- Fully Sharded Data Parallel (FSDP)的原理？

#### 子小节8.4.2：分布式优化技术

**通信与计算重叠**
- 如何实现梯度计算和通信的重叠？
- DDP中的overlap机制是如何工作的？
- 流水线并行中的调度策略？
- 如何profile分布式训练的性能瓶颈？

**大规模训练技巧**
- LARS和LAMB优化器在大batch训练中的作用？
- Layer-wise Adaptive Rate Scaling的原理？
- 如何处理大batch训练的收敛问题？
- 分布式训练的调试和错误处理技巧？

---

## 章节9：性能优化

### 小节9.1：JIT编译与图优化

#### 子小节9.1.1：TorchScript优化

**JIT编译**
- Just-In-Time编译如何加速PyTorch代码？
- @torch.jit.script装饰器的作用？
- torch.jit.trace的优化效果？
- 何时使用JIT能获得明显加速？

**图优化**
- TorchScript的图优化包括哪些？
- 常量折叠、死代码消除等优化是什么？
- 如何查看优化后的计算图？
- freeze()方法的作用和使用场景？

#### 子小节9.1.2：torch.compile

**PyTorch 2.0编译器**
- torch.compile()的工作原理？
- 与TorchScript相比有什么优势？
- mode参数('default', 'reduce-overhead', 'max-autotune')的区别？
- 哪些代码模式会导致编译失败？

**编译优化**
- 动态形状对编译性能的影响？
- 如何使用fullgraph=True？
- 编译缓存机制是什么？
- 如何调试torch.compile的问题？

### 小节9.2：算子与内核优化

#### 子小节9.2.1：算子融合

**融合操作**
- 算子融合(Operator Fusion)的原理和好处？
- PyTorch中哪些操作会自动融合？
- GELU、LayerNorm等融合操作的性能提升？
- 如何使用torch.jit.fuser启用融合？

**自定义融合**
- 如何实现自定义的融合操作？
- Triton语言在PyTorch中的应用？
- FlashAttention的融合策略是什么？
- 如何平衡融合粒度和灵活性？

#### 子小节9.2.2：高效算子

**内置优化算子**
- torch.nn.functional中的优化实现？
- scaled_dot_product_attention的优化？
- fused_adam等融合优化器的使用？
- 如何选择高效的算子实现？

**自定义CUDA算子**
- 何时需要编写自定义CUDA算子？
- 如何使用CUDA扩展编译自定义算子？
- load_inline和cpp_extension的使用？
- 自定义算子的性能调优方法？

### 小节9.3：内存优化

#### 子小节9.3.1：显存管理

**显存分析**
- 如何分析模型的显存占用？
- torch.cuda.memory_summary()的使用？
- 显存占用的主要组成部分？
- 如何定位显存泄漏？

**显存优化技术**
- Gradient Checkpointing如何节省显存？
- 激活值重计算的trade-off是什么？
- torch.utils.checkpoint.checkpoint的使用方法？
- 如何选择checkpoint的粒度？

**显存释放**
- del和torch.cuda.empty_cache()的区别？
- 何时调用empty_cache()有效？
- 如何避免显存碎片化？
- 显存池(Memory Pool)的管理策略？

#### 子小节9.3.2：内存效率

**Inplace操作**
- inplace操作的优缺点？
- 哪些操作支持inplace？如何使用？
- inplace操作对autograd的影响？
- 何时应该避免使用inplace操作？

**内存复用**
- 如何实现张量的内存复用？
- buffer复用的技巧？
- 如何减少临时张量的分配？
- 内存对齐对性能的影响？

### 小节9.4：性能分析与调优

#### 子小节9.4.1：Profiling工具

**PyTorch Profiler**
- torch.profiler的使用方法？
- profile、record_function的作用？
- 如何查看profiler的结果？
- TensorBoard中如何可视化性能数据？

**性能指标**
- 如何测量GPU利用率？
- 算子执行时间如何统计？
- 内存分配和释放的时间开销？
- 通信时间在分布式训练中如何测量？

#### 子小节9.4.2：性能调优实践

**瓶颈分析**
- 如何识别训练过程中的性能瓶颈？
- CPU-GPU数据传输的优化？
- 数据加载速度不足如何处理？
- 模型计算和数据IO的平衡？

**优化策略**
- 如何系统性地进行性能优化？
- 常见的性能优化checklist？
- benchmark和实际训练性能的差异？
- 如何评估优化效果？

**特定场景优化**
- 小模型和大模型的优化策略差异？
- 推理优化的关键点？
- 实时推理的性能要求？
- 批处理推理的优化技巧？

---

## 章节10：PyTorch内部机制

### 小节10.1：架构与设计

#### 子小节10.1.1：核心架构

**整体架构**
- PyTorch的分层架构是怎样的？
- Python前端和C++后端如何交互？
- ATen库的作用是什么？
- Autograd引擎在架构中的位置？

**执行流程**
- 一个张量操作的完整执行路径？
- Python调用到C++实现的过程？
- 动态调度机制是如何工作的？
- Eager模式和Graph模式的执行差异？

#### 子小节10.1.2：Dispatcher机制

**调度系统**
- PyTorch Dispatcher的作用是什么？
- 如何根据设备类型分发到不同实现？
- 多dispatch key的优先级顺序？
- Autograd key在调度中的作用？

**扩展机制**
- 如何为新设备添加支持？
- 注册自定义backend的方法？
- torch::dispatch的使用？
- CompositeImplicitAutograd和CompositeExplicitAutograd的区别？

### 小节10.2：张量存储机制

#### 子小节10.2.1：Storage与Tensor

**底层存储**
- Tensor和Storage的关系？
- 多个Tensor如何共享同一Storage？
- offset和stride如何确定Tensor的视图？
- 存储分配器的工作原理？

**内存布局**
- 行优先(row-major)和列优先(column-major)？
- channels_last内存格式是什么？
- 内存布局对性能的影响？
- 如何转换不同的内存布局？

#### 子小节10.2.2：张量元数据

**TensorImpl**
- TensorImpl包含哪些信息？
- 元数据和实际数据的分离设计？
- 引用计数机制如何工作？
- COW(Copy-On-Write)在PyTorch中的应用？

**类型系统**
- ScalarType的定义和作用？
- Device、Layout、MemoryFormat的组合？
- 类型提升(type promotion)规则？
- 如何扩展新的数据类型？

### 小节10.3：Autograd引擎

#### 子小节10.3.1：反向传播引擎

**图构建与执行**
- 计算图的节点和边分别表示什么？
- 反向传播的拓扑排序算法？
- 如何处理多输出的反向传播？
- 累积梯度的实现机制？

**Function与Node**
- Function和Node的关系？
- backward()方法的执行流程？
- saved tensors的管理策略？
- 如何优化saved tensors的内存占用？

#### 子小节10.3.2：高级Autograd特性

**高阶导数**
- 高阶导数的实现原理？
- grad_mode的嵌套如何处理？
- 双向传播(double backward)的计算图？
- 如何高效计算Hessian-vector积？

**自定义Autograd**
- 如何深入理解Function.apply的机制？
- @once_differentiable装饰器的实现？
- materialize_grads参数的作用？
- 如何处理不可微分点？

### 小节10.4：扩展PyTorch

#### 子小节10.4.1：C++扩展

**扩展基础**
- 为什么需要C++扩展？
- pybind11在PyTorch中的使用？
- 如何编写C++扩展模块？
- setup.py中的CppExtension配置？

**CUDA扩展**
- 如何编写CUDA扩展？
- CUDAExtension的编译选项？
- AT_DISPATCH宏的作用？
- CUDA kernel的错误处理？

**即时编译**
- torch.utils.cpp_extension.load的使用？
- load_inline和load的区别？
- ninja编译系统的作用？
- 编译缓存机制如何工作？

#### 子小节10.4.2：自定义算子

**算子注册**
- 如何向PyTorch注册自定义算子？
- torch.library的使用方法？
- TORCH_LIBRARY宏的作用？
- 算子的schema定义规则？

**多设备支持**
- 如何为自定义算子提供CPU和CUDA实现？
- dispatch key的选择？
- 自动微分支持如何添加？
- backward formula的注册方式？

**性能与优化**
- 自定义算子的性能测试方法？
- 如何确保自定义算子的正确性？
- 与原生算子的性能对比？
- 集成到模型训练的最佳实践？

### 小节10.5：前沿特性

#### 子小节10.5.1：functorch

**函数式转换**
- functorch库的设计理念？
- vmap的实现原理和使用？
- grad和jacrev/jacfwd的区别？
- 如何组合多个函数式转换？

**应用场景**
- vmap在批量雅可比计算中的应用？
- per-sample gradients如何高效计算？
- ensemble模型如何用vmap优化？
- functorch与torch.compile的结合？

#### 子小节10.5.2：其他高级特性

**torch.fx**
- FX图表示的优势？
- symbolic_trace的工作原理？
- 如何用FX实现自定义优化pass？
- 量化和剪枝中FX的应用？

**torch.export**
- export API与torch.jit的区别？
- Ahead-Of-Time(AOT)编译的优势？
- 动态形状如何在export中处理？
- ExecuTorch与torch.export的关系？

**前沿研究方向**
- PyTorch 2.x的主要改进方向？
- 编译器栈的演进趋势？
- 量化和稀疏化的原生支持？
- 分布式训练的未来发展？

---

**结语：本大纲涵盖了PyTorch从基础到高级、从使用到原理的全方位知识体系，适用于应届生岗位的全面技术考核。每个问题点都可以进一步扩展为独立的深入讨论主题。**

