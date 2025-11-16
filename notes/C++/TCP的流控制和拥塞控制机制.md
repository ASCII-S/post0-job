---
created: '2025-10-19'
last_reviewed: null
next_review: '2025-10-19'
review_count: 0
difficulty: medium
mastery_level: 0.0
tags:
- C++
- C++/TCP的流控制和拥塞控制机制.md
related_outlines: []
---
# TCP的流控制和拥塞控制机制

## 标准答案（面试简答）

**流控制**是点对点的，使用滑动窗口机制，接收方通过通告窗口大小（rwnd）告知发送方自己的接收缓冲区大小，防止发送方发送速度过快导致接收方缓冲区溢出。**拥塞控制**是全局性的，通过维护拥塞窗口（cwnd），采用慢启动、拥塞避免、快重传、快恢复四个算法，防止过多数据注入网络造成拥塞。发送窗口大小取rwnd和cwnd的最小值。

---

## 详细讲解

### 一、流控制（Flow Control）

#### 1. 基本概念

**目的**：防止发送方发送速度过快，导致接收方来不及处理而丢失数据

**原理**：接收方控制发送方的发送速率（端到端的控制）

**机制**：滑动窗口（Sliding Window）

#### 2. 滑动窗口机制

**窗口的含义**：
```
发送窗口：发送方允许发送但还未收到确认的最大数据量
接收窗口（rwnd）：接收方剩余缓冲区大小
```

**窗口滑动过程**：
```
发送方窗口：
┌─────────────────────────────────────────┐
│ 已发送已确认 │ 已发送未确认 │ 可发送 │ 不可发送 │
└─────────────────────────────────────────┘
              ↑                        ↑
          窗口左边界                窗口右边界

当收到ACK时：
1. 左边界右移（确认了数据）
2. 右边界根据新的rwnd调整
```

**示例**：
```
时刻T0: rwnd=6, 已发送seq=1-3，未确认
┌───┬───┬───┬───┬───┬───┬───┬───┐
│ ✓ │ 1 │ 2 │ 3 │ 4 │ 5 │ 6 │ X │
└───┴───┴───┴───┴───┴───┴───┴───┘
      └─────────────────┘
           发送窗口

时刻T1: 收到ACK=2, rwnd=5
┌───┬───┬───┬───┬───┬───┬───┬───┐
│ ✓ │ ✓ │ 2 │ 3 │ 4 │ 5 │ 6 │ X │
└───┴───┴───┴───┴───┴───┴───┴───┘
          └─────────────────┘
               发送窗口（右边界左移）
```

#### 3. 零窗口问题

**场景**：接收方缓冲区满，通告rwnd=0

**问题**：发送方停止发送，如何知道接收方何时恢复？

**解决**：零窗口探测（Zero Window Probe）
```
1. 发送方收到rwnd=0后，启动持续定时器
2. 定时器超时，发送零窗口探测报文（1字节数据）
3. 接收方响应当前窗口大小
4. 如果rwnd>0，恢复发送
```

#### 4. 糊涂窗口综合症（Silly Window Syndrome）

**现象**：发送或接收少量数据，导致网络效率低下

**发送方糊涂窗口**：
```
问题：应用程序每次只产生少量数据，发送方就发送
解决：Nagle算法
  - 如果数据达到MSS，立即发送
  - 如果未达到MSS，等待：
    1. 直到累积到MSS
    2. 或收到之前数据的ACK
```

**接收方糊涂窗口**：
```
问题：接收方每次只读取少量数据，频繁通告小窗口
解决：延迟确认
  - 等到缓冲区至少有MSS空间
  - 或等到缓冲区有一半空间
  - 才通告窗口更新
```

### 二、拥塞控制（Congestion Control）

#### 1. 基本概念

**目的**：防止过多数据注入网络，避免网络拥塞

**拥塞的标志**：
- 超时（认为网络严重拥塞）
- 收到3个重复ACK（轻度拥塞）

**关键变量**：
```
cwnd (Congestion Window): 拥塞窗口
ssthresh (Slow Start Threshold): 慢启动阈值
发送窗口 = min(rwnd, cwnd)
```

#### 2. 慢启动（Slow Start）

**目的**：连接建立初期，探测网络容量

**算法**：
```
1. 初始化：cwnd = 1 MSS (或更大，如10 MSS)
2. 每收到一个ACK：cwnd += 1 MSS
3. 结果：cwnd指数增长（1, 2, 4, 8, 16...）
4. 停止条件：cwnd >= ssthresh，转入拥塞避免
```

**过程示例**：
```
RTT 0: cwnd=1,  发送1个段
RTT 1: cwnd=2,  发送2个段
RTT 2: cwnd=4,  发送4个段
RTT 3: cwnd=8,  发送8个段
RTT 4: cwnd=16, 发送16个段
...
```

#### 3. 拥塞避免（Congestion Avoidance）

**触发条件**：cwnd >= ssthresh

**算法**：
```
每个RTT：cwnd += 1 MSS
（即每收到cwnd个ACK，cwnd才加1）

结果：cwnd线性增长
```

**加性增、乘性减（AIMD）**：
```
加性增：拥塞避免阶段，线性增加
乘性减：检测到拥塞，cwnd减半
```

#### 4. 快重传（Fast Retransmit）

**触发条件**：收到3个重复ACK（duplicate ACK）

**原因分析**：
```
收到重复ACK说明：
1. 接收方收到了乱序的段
2. 某个段可能丢失了
3. 后续段正常到达（网络未严重拥塞）
```

**算法**：
```
1. 收到3个重复ACK
2. 立即重传被怀疑丢失的段
3. 不等超时定时器
```

**示例**：
```
发送：1, 2, 3, 4, 5
接收：1, 3, 4, 5 (段2丢失)

接收方发送：
ACK=2 (收到1)
ACK=2 (收到3，期待2，重复ACK)
ACK=2 (收到4，期待2，重复ACK)
ACK=2 (收到5，期待2，重复ACK)

发送方收到3个重复ACK=2，立即重传段2
```

#### 5. 快恢复（Fast Recovery）

**触发条件**：快重传后

**目的**：认为网络未严重拥塞（收到重复ACK说明后续数据在传输），不需要慢启动

**算法（Reno版本）**：
```
1. ssthresh = cwnd / 2
2. cwnd = ssthresh + 3 MSS (3是重复ACK的数量)
3. 重传丢失的段
4. 每收到一个重复ACK，cwnd += 1 MSS（临时膨胀）
5. 收到新的ACK（确认重传的段）：
   - cwnd = ssthresh
   - 进入拥塞避免阶段
```

**为什么+3**：
```
3个重复ACK说明有3个段已经离开网络
可以发送3个新段来填充这个"空缺"
```

### 三、完整的状态转换

#### 状态机示例

```
初始状态：
cwnd = 1 MSS
ssthresh = 64 KB (典型初始值)

┌─────────────┐
│ 慢启动阶段   │ cwnd < ssthresh
│ 指数增长     │ cwnd *= 2 (每个RTT)
└──────┬──────┘
       │ cwnd >= ssthresh
       ↓
┌─────────────┐
│ 拥塞避免阶段 │ cwnd >= ssthresh
│ 线性增长     │ cwnd += 1 (每个RTT)
└──────┬──────┘
       │
       ├─────→ 超时 ──→ ssthresh = cwnd/2
       │                cwnd = 1
       │                回到慢启动
       │
       └─────→ 3个重复ACK ──→ 快重传+快恢复
                               ssthresh = cwnd/2
                               cwnd = ssthresh + 3
                               收到新ACK后进入拥塞避免
```

#### 数值演化示例

```
假设MSS=1KB, 初始ssthresh=16KB

RTT  事件              cwnd      ssthresh   阶段
0    开始              1KB       16KB       慢启动
1    正常ACK           2KB       16KB       慢启动
2    正常ACK           4KB       16KB       慢启动
3    正常ACK           8KB       16KB       慢启动
4    正常ACK           16KB      16KB       慢启动→拥塞避免
5    正常ACK           17KB      16KB       拥塞避免
6    正常ACK           18KB      16KB       拥塞避免
7    3个重复ACK        12KB      9KB        快恢复
8    新ACK             9KB       9KB        拥塞避免
9    正常ACK           10KB      9KB        拥塞避免
10   超时              1KB       5KB        慢启动
11   正常ACK           2KB       5KB        慢启动
...
```

### 四、流控制 vs 拥塞控制

| 对比项   | 流控制           | 拥塞控制                         |
| -------- | ---------------- | -------------------------------- |
| 目的     | 防止接收方溢出   | 防止网络拥塞                     |
| 控制对象 | 点对点           | 端到端 + 网络                    |
| 控制方   | 接收方主导       | 发送方主导                       |
| 窗口     | rwnd（接收窗口） | cwnd（拥塞窗口）                 |
| 调整依据 | 接收缓冲区大小   | 网络状况                         |
| 算法     | 滑动窗口         | 慢启动、拥塞避免、快重传、快恢复 |

**实际发送窗口**：
```cpp
send_window = min(rwnd, cwnd);
```

### 五、现代TCP改进

#### 1. TCP Tahoe
```
- 最早实现慢启动、拥塞避免
- 超时和3个重复ACK都回到慢启动（cwnd=1）
```

#### 2. TCP Reno
```
- 引入快重传、快恢复
- 3个重复ACK不回到慢启动
- 目前最常用
```

#### 3. TCP New Reno
```
- 改进快恢复
- 处理多个段丢失的情况
```

#### 4. TCP CUBIC（Linux默认）
```
- 改进拥塞避免算法
- 窗口增长函数为三次函数（cubic）
- 更适合高带宽长延迟网络
- 快速探测可用带宽
```

#### 5. TCP BBR（Google）
```
- 基于带宽和RTT的拥塞控制
- 不依赖丢包作为拥塞信号
- 主动测量瓶颈带宽
- 在高丢包环境表现更好
```

### 六、代码示例（概念性）

```cpp
// 发送方的窗口控制（伪代码）
class TCPSender {
    uint32_t rwnd;           // 接收方通告的窗口
    uint32_t cwnd;           // 拥塞窗口
    uint32_t ssthresh;       // 慢启动阈值
    uint32_t next_seq;       // 下一个要发送的序号
    uint32_t send_base;      // 最早未确认的序号
    
    enum State {
        SLOW_START,
        CONGESTION_AVOIDANCE,
        FAST_RECOVERY
    } state;
    
    // 发送数据
    void send() {
        uint32_t window = std::min(rwnd, cwnd);
        uint32_t in_flight = next_seq - send_base;
        
        // 可发送的数据量
        if (in_flight < window) {
            uint32_t can_send = window - in_flight;
            // 发送数据...
        }
    }
    
    // 收到ACK
    void on_ack(uint32_t ack_num, bool is_duplicate) {
        if (!is_duplicate) {
            // 新的ACK
            send_base = ack_num;
            
            if (state == SLOW_START) {
                cwnd += MSS;  // 指数增长
                if (cwnd >= ssthresh) {
                    state = CONGESTION_AVOIDANCE;
                }
            } else if (state == CONGESTION_AVOIDANCE) {
                cwnd += MSS * MSS / cwnd;  // 线性增长
            } else if (state == FAST_RECOVERY) {
                cwnd = ssthresh;
                state = CONGESTION_AVOIDANCE;
            }
        } else {
            // 重复ACK
            dup_ack_count++;
            
            if (dup_ack_count == 3) {
                // 快重传
                retransmit(ack_num);
                
                // 快恢复
                ssthresh = cwnd / 2;
                cwnd = ssthresh + 3 * MSS;
                state = FAST_RECOVERY;
            } else if (state == FAST_RECOVERY) {
                cwnd += MSS;  // 临时膨胀
            }
        }
    }
    
    // 超时
    void on_timeout() {
        ssthresh = cwnd / 2;
        cwnd = MSS;
        state = SLOW_START;
        retransmit(send_base);
    }
};
```

### 七、实际应用考虑

#### 1. 带宽延迟积（BDP）
```
BDP = 带宽 × RTT

例如：
带宽 = 100 Mbps = 12.5 MB/s
RTT = 100 ms
BDP = 12.5 MB/s × 0.1s = 1.25 MB

理想窗口大小应该 ≥ BDP，才能充分利用带宽
```

#### 2. 长肥网络（Long Fat Network）
```
高带宽 × 长延迟 = 大窗口需求
传统TCP窗口字段只有16位（最大64KB）
需要窗口扩大选项（Window Scaling）
```

#### 3. 无线网络优化
```
问题：无线网络丢包不一定是拥塞
传统TCP会误判，降低cwnd
解决：区分拥塞丢包和链路丢包
```

### 八、总结

**流控制核心**：
- 滑动窗口机制
- 接收方控制发送方速率
- 保护接收方不被淹没

**拥塞控制核心**：
- 四个算法：慢启动、拥塞避免、快重传、快恢复
- 动态调整cwnd
- 平衡网络利用率和公平性

**实际窗口**：
```
发送窗口 = min(rwnd, cwnd)
既要保护接收方，也要保护网络
```

**常见面试追问**：
1. 为什么慢启动是指数增长？→ 快速探测网络容量
2. 为什么快恢复不回到慢启动？→ 重复ACK说明网络未严重拥塞
3. 超时和重复ACK处理有何不同？→ 超时更严重，cwnd回到1；重复ACK较轻，cwnd减半
4. CUBIC和Reno的区别？→ 窗口增长函数不同，CUBIC更适合高速网络


---

## 相关笔记
<!-- 自动生成 -->

暂无相关笔记

