---
created: '2025-10-19'
last_reviewed: null
next_review: '2025-10-19'
review_count: 0
difficulty: medium
mastery_level: 0.0
tags:
- C++
- C++/阻塞I
related_outlines: []
---
# 阻塞I/O、非阻塞I/O、I/O多路复用、异步I/O的区别

## 标准答案（面试简答）

四种I/O模型的核心区别在于等待数据和拷贝数据的方式：**阻塞I/O**在等待数据和拷贝数据时都阻塞进程；**非阻塞I/O**在数据未就绪时立即返回错误，需要轮询，数据就绪后拷贝时仍阻塞；**I/O多路复用**通过select/epoll等机制同时监控多个fd，等待阶段阻塞在多路复用函数上，数据到达后仍需阻塞拷贝；**异步I/O**在发起请求后立即返回，等待和拷贝都由内核完成，完成后通知应用程序，是真正的异步。前三种都是同步I/O（需要应用程序自己拷贝数据），只有异步I/O是真正异步的。

---

## 详细讲解

### 一、I/O操作的两个阶段

在理解不同I/O模型前，需要明确I/O操作的两个阶段：

```
以读取socket数据为例：

阶段1：等待数据（Waiting for data）
├─ 等待网络数据到达网卡
├─ 数据从网卡拷贝到内核缓冲区
└─ 数据在内核缓冲区准备好

阶段2：拷贝数据（Copying data）
├─ 将数据从内核缓冲区拷贝到用户空间
└─ 应用程序可以访问数据

完整流程：
网卡 → 内核缓冲区 → 用户空间
     (阶段1)      (阶段2)
```

**关键点**：
- 阶段1可能很长（网络延迟）
- 阶段2通常很快（内存拷贝）
- 不同I/O模型在这两个阶段的处理方式不同

### 二、阻塞I/O（Blocking I/O）

#### 1. 工作原理

```
用户进程                     内核
   │                          │
   │──── read() ─────────────>│
   │                          │ 等待数据
   │        阻塞              │   ↓
   │                          │ 数据就绪
   │                          │   ↓
   │        阻塞              │ 拷贝数据
   │                          │   ↓
   │<──── 返回数据 ───────────│
   │                          │
   │ 处理数据                 │
   ↓                          ↓
```

#### 2. 代码示例

```cpp
#include <sys/socket.h>
#include <unistd.h>

void blocking_io_example(int sockfd) {
    char buf[1024];
    
    // read()会阻塞，直到数据到达
    ssize_t n = read(sockfd, buf, sizeof(buf));
    // ↑ 在这里阻塞等待，进程什么都做不了
    
    if (n > 0) {
        // 数据到达后才会执行到这里
        process_data(buf, n);
    }
}
```

#### 3. 特点

**优点**：
```
1. 编程简单直观
2. 不需要轮询
3. CPU利用率高（进程休眠不占CPU）
```

**缺点**：
```
1. 一个线程只能处理一个连接
2. 多连接需要多线程/多进程
3. 线程切换开销大
4. 无法充分利用单线程
```

#### 4. 适用场景

```
- 连接数少（< 100）
- 每个连接都很活跃
- 对延迟不敏感
- 代码简单性优先
```

### 三、非阻塞I/O（Non-blocking I/O）

#### 1. 工作原理

```
用户进程                     内核
   │                          │
   │──── read() ─────────────>│
   │<──── EAGAIN ─────────────│ 数据未就绪，立即返回
   │                          │
   │──── read() ─────────────>│ 
   │<──── EAGAIN ─────────────│ 还是没数据
   │                          │
   │      轮询...              │
   │                          │
   │──── read() ─────────────>│
   │                          │ 数据就绪
   │        阻塞              │ 拷贝数据
   │<──── 返回数据 ───────────│
   │                          │
   ↓                          ↓
```

#### 2. 代码示例

```cpp
#include <fcntl.h>
#include <errno.h>
#include <unistd.h>

// 设置非阻塞
void set_nonblocking(int fd) {
    int flags = fcntl(fd, F_GETFL, 0);
    fcntl(fd, F_SETFL, flags | O_NONBLOCK);
}

void nonblocking_io_example(int sockfd) {
    set_nonblocking(sockfd);
    
    char buf[1024];
    
    while (true) {
        ssize_t n = read(sockfd, buf, sizeof(buf));
        
        if (n > 0) {
            // 读到数据
            process_data(buf, n);
            break;
        } else if (n == 0) {
            // 连接关闭
            break;
        } else {  // n < 0
            if (errno == EAGAIN || errno == EWOULDBLOCK) {
                // 数据未就绪，继续轮询
                // 可以在这里做其他事情
                do_other_work();
                continue;
            } else {
                // 真正的错误
                perror("read error");
                break;
            }
        }
    }
}
```

#### 3. 轮询模式（忙等待）

```cpp
// 处理多个连接的非阻塞I/O
void handle_multiple_nonblocking(std::vector<int>& fds) {
    while (true) {
        for (int fd : fds) {
            char buf[1024];
            ssize_t n = read(fd, buf, sizeof(buf));
            
            if (n > 0) {
                process_data(fd, buf, n);
            } else if (n < 0 && errno != EAGAIN) {
                handle_error(fd);
            }
            // EAGAIN：跳过，检查下一个fd
        }
        
        // 不断循环检查所有fd
        // 问题：CPU空转，资源浪费
    }
}
```

#### 4. 特点

**优点**：
```
1. 不会阻塞在单个I/O上
2. 可以在等待时做其他事情
3. 一个线程可以处理多个连接（理论上）
```

**缺点**：
```
1. 需要不断轮询，CPU占用高
2. 数据拷贝阶段仍然阻塞
3. 轮询间隔难以把握（太快浪费CPU，太慢响应慢）
4. 编程复杂度增加
```

#### 5. 适用场景

```
很少单独使用，通常作为其他模型的基础
- 配合I/O多路复用使用（epoll的ET模式）
- 需要在等待I/O时执行其他任务
```

### 四、I/O多路复用（I/O Multiplexing）

#### 1. 工作原理

```
用户进程                     内核
   │                          │
   │── select()/epoll_wait() >│
   │                          │ 监控多个fd
   │        阻塞              │   ↓
   │                          │ fd1: 未就绪
   │                          │ fd2: 未就绪
   │        阻塞              │ fd3: 就绪 ✓
   │<─── 返回就绪fd ──────────│
   │                          │
   │──── read(fd3) ──────────>│
   │        阻塞              │ 拷贝数据
   │<──── 返回数据 ───────────│
   │                          │
   ↓                          ↓
```

#### 2. 代码示例（epoll）

```cpp
#include <sys/epoll.h>
#include <unistd.h>
#include <vector>

void io_multiplexing_example(int listen_fd) {
    // 创建epoll实例
    int epfd = epoll_create1(0);
    
    // 添加监听socket
    struct epoll_event ev;
    ev.events = EPOLLIN;
    ev.data.fd = listen_fd;
    epoll_ctl(epfd, EPOLL_CTL_ADD, listen_fd, &ev);
    
    const int MAX_EVENTS = 10;
    struct epoll_event events[MAX_EVENTS];
    
    while (true) {
        // 阻塞等待事件（可以同时监控多个fd）
        int nfds = epoll_wait(epfd, events, MAX_EVENTS, -1);
        // ↑ 在这里阻塞，但是监控的是多个fd
        
        // 处理所有就绪的fd
        for (int i = 0; i < nfds; i++) {
            int fd = events[i].data.fd;
            
            if (fd == listen_fd) {
                // 新连接
                int client_fd = accept(listen_fd, NULL, NULL);
                struct epoll_event client_ev;
                client_ev.events = EPOLLIN;
                client_ev.data.fd = client_fd;
                epoll_ctl(epfd, EPOLL_CTL_ADD, client_fd, &client_ev);
            } else {
                // 数据到达
                char buf[1024];
                ssize_t n = read(fd, buf, sizeof(buf));
                // ↑ 这里仍然会阻塞，直到数据拷贝完成
                
                if (n > 0) {
                    process_data(fd, buf, n);
                } else if (n == 0) {
                    epoll_ctl(epfd, EPOLL_CTL_DEL, fd, NULL);
                    close(fd);
                }
            }
        }
    }
    
    close(epfd);
}
```

#### 3. 特点

**优点**：
```
1. 单线程处理多个连接
2. 避免轮询，事件驱动
3. 连接数增加时性能优秀（特别是epoll）
4. 减少线程切换开销
```

**缺点**：
```
1. 仍然是同步I/O（需要自己read/write）
2. 数据拷贝阶段阻塞
3. 编程复杂度高
4. 不同平台API不同（select/epoll/kqueue/IOCP）
```

#### 4. 适用场景

```
- 大量并发连接（> 1000）
- 连接大部分时间空闲
- 高性能网络服务器
- 需要单线程处理多连接
```

### 五、异步I/O（Asynchronous I/O）

#### 1. 工作原理

```
用户进程                     内核
   │                          │
   │── aio_read() ───────────>│ 发起异步读
   │<─── 立即返回 ────────────│ 不等待
   │                          │   ↓
   │ 继续执行其他任务          │ 等待数据
   │       ↓                  │   ↓
   │ 做其他事情                │ 数据就绪
   │       ↓                  │   ↓
   │ 做更多事情                │ 拷贝数据
   │       ↓                  │   ↓
   │<─── 通知完成 ────────────│ 数据已在用户空间
   │                          │
   │ 处理数据                 │
   ↓                          ↓
```

#### 2. 代码示例（Linux AIO）

```cpp
#include <aio.h>
#include <signal.h>
#include <string.h>

// 异步I/O完成的回调函数
void aio_completion_handler(sigval_t sigval) {
    struct aiocb *req = (struct aiocb *)sigval.sival_ptr;
    
    // 检查是否成功
    ssize_t ret = aio_return(req);
    if (ret > 0) {
        // 数据已经在缓冲区中，可以直接使用
        process_data((char*)req->aio_buf, ret);
    }
    
    delete[] (char*)req->aio_buf;
    delete req;
}

void async_io_example(int fd) {
    // 准备异步读请求
    struct aiocb *req = new struct aiocb;
    memset(req, 0, sizeof(struct aiocb));
    
    char *buf = new char[1024];
    
    req->aio_fildes = fd;
    req->aio_buf = buf;
    req->aio_nbytes = 1024;
    req->aio_offset = 0;
    
    // 设置完成通知方式
    req->aio_sigevent.sigev_notify = SIGEV_THREAD;
    req->aio_sigevent.sigev_notify_function = aio_completion_handler;
    req->aio_sigevent.sigev_value.sival_ptr = req;
    
    // 发起异步读，立即返回
    if (aio_read(req) == -1) {
        perror("aio_read");
        return;
    }
    
    // 这里可以立即做其他事情
    // 不需要等待I/O完成
    do_other_work();
    do_more_work();
    
    // 当I/O完成时，aio_completion_handler会被调用
}
```

#### 3. io_uring（现代Linux异步I/O）

```cpp
#include <liburing.h>

void io_uring_example(int fd) {
    struct io_uring ring;
    io_uring_queue_init(32, &ring, 0);
    
    // 获取一个提交队列项
    struct io_uring_sqe *sqe = io_uring_get_sqe(&ring);
    
    char *buf = new char[1024];
    
    // 准备读操作
    io_uring_prep_read(sqe, fd, buf, 1024, 0);
    
    // 设置用户数据（用于识别请求）
    io_uring_sqe_set_data(sqe, buf);
    
    // 提交请求
    io_uring_submit(&ring);
    
    // 可以继续做其他事情
    do_other_work();
    
    // 等待完成（也可以不等待，继续干别的）
    struct io_uring_cqe *cqe;
    io_uring_wait_cqe(&ring, &cqe);
    
    // 获取结果
    char *completed_buf = (char*)io_uring_cqe_get_data(cqe);
    ssize_t ret = cqe->res;
    
    if (ret > 0) {
        process_data(completed_buf, ret);
    }
    
    io_uring_cqe_seen(&ring, cqe);
    io_uring_queue_exit(&ring);
    delete[] buf;
}
```

#### 4. 特点

**优点**：
```
1. 真正的异步，发起后立即返回
2. 等待和拷贝都由内核完成
3. 应用程序可以持续做其他事情
4. 充分利用CPU和I/O并行
5. 性能最优
```

**缺点**：
```
1. 编程复杂度最高
2. 调试困难（回调地狱）
3. Linux AIO功能有限（主要支持直接I/O）
4. Windows IOCP和Linux实现差异大
5. 需要操作系统支持
```

#### 5. 适用场景

```
- 高性能数据库
- 高并发文件服务器
- 需要极致性能的场景
- 可以容忍编程复杂度
- Windows平台（IOCP天然异步）
```

### 六、四种模型对比

#### 1. 核心差异表

| 特性           | 阻塞I/O | 非阻塞I/O | I/O多路复用  | 异步I/O |
| -------------- | ------- | --------- | ------------ | ------- |
| **等待数据**   | 阻塞    | 轮询      | 阻塞(select) | 不阻塞  |
| **拷贝数据**   | 阻塞    | 阻塞      | 阻塞         | 不阻塞  |
| **同步/异步**  | 同步    | 同步      | 同步         | 异步    |
| **单线程并发** | 否      | 理论可以  | 是           | 是      |
| **CPU占用**    | 低      | 高(轮询)  | 低           | 低      |
| **编程复杂度** | 低      | 中        | 高           | 最高    |
| **性能**       | 差      | 差        | 好           | 最好    |
| **适用连接数** | 少      | 少        | 多           | 极多    |

#### 2. 同步I/O vs 异步I/O

**关键区别**：
```
同步I/O：应用程序自己负责数据拷贝
- 阻塞I/O：等待+拷贝都阻塞
- 非阻塞I/O：等待轮询，拷贝阻塞
- I/O多路复用：等待多个fd，拷贝阻塞

异步I/O：内核负责数据拷贝
- 应用程序发起请求后立即返回
- 内核完成等待和拷贝
- 完成后通知应用程序
```

#### 3. 阻塞 vs 非阻塞

```
阻塞：
- 函数调用后不立即返回
- 直到操作完成或出错

非阻塞：
- 函数调用立即返回
- 返回状态码（成功/EAGAIN/错误）
- 需要应用程序处理未就绪的情况
```

### 七、时间线对比

假设一次I/O操作耗时：等待数据5秒，拷贝数据0.1秒

#### 阻塞I/O
```
0s ──────── 5s ──── 5.1s
│          │       │
read()─────┘       └──> 返回
     (阻塞5.1秒)

进程状态：休眠5.1秒
```

#### 非阻塞I/O
```
0s    1s    2s    3s    4s    5s ── 5.1s
│     │     │     │     │     │    │
read()read()read()read()read()read()└> 返回
EAGAIN EAGAIN EAGAIN EAGAIN EAGAIN 成功

进程状态：持续轮询5秒，拷贝阻塞0.1秒
```

#### I/O多路复用
```
0s ─────────── 5s ── 5.1s ── 5.2s
│             │     │      │
epoll_wait()──┘     │      └──> 返回
  (阻塞5秒)         read()
                  (阻塞0.1秒)

进程状态：休眠5秒，拷贝阻塞0.1秒
(但可以监控多个fd)
```

#### 异步I/O
```
0s ────────────────── 5.1s
│                     │
aio_read()            └──> 回调通知
立即返回

进程状态：立即返回，可以做其他事情5.1秒
```

### 八、性能对比

#### 场景：处理10000个并发连接，每秒1000个活跃

**阻塞I/O**：
```
需要10000个线程
内存占用：10000 × 8MB(栈) = 78GB
线程切换开销：巨大
吞吐量：低
结论：不可行
```

**非阻塞I/O（纯轮询）**：
```
1个线程
CPU占用：100%（忙等待）
响应延迟：取决于轮询频率
吞吐量：中等
结论：CPU浪费严重
```

**I/O多路复用**：
```
1-8个线程
CPU占用：5-20%
响应延迟：很低
吞吐量：高
结论：适合大多数场景
```

**异步I/O**：
```
1-8个线程
CPU占用：3-10%
响应延迟：最低
吞吐量：最高
结论：性能最优，但复杂度高
```

### 九、实际应用中的选择

#### 1. Web服务器

**Nginx**：
```
I/O多路复用(epoll/kqueue) + 非阻塞I/O
- 单worker进程处理数万连接
- 事件驱动架构
- 性能优秀
```

**Apache（传统模式）**：
```
阻塞I/O + 多进程/多线程
- 一个连接一个进程/线程
- 简单但并发受限
```

#### 2. Redis

```
I/O多路复用(epoll) + 单线程
- 单线程处理所有客户端连接
- 命令执行快，不需要多线程
- 避免了锁的开销
```

#### 3. Node.js

```
异步I/O(libuv) + 事件循环
- 所有I/O都是异步的
- JavaScript层面是异步回调
- 底层使用线程池处理文件I/O
```

#### 4. 数据库

**MySQL**：
```
每个连接一个线程（阻塞I/O）
- 简单直观
- 连接数有限
```

**PostgreSQL**：
```
每个连接一个进程（阻塞I/O）
- 隔离性好
- 连接池必不可少
```

**现代数据库**：
```
倾向于使用异步I/O
- 提高磁盘I/O吞吐量
- 减少线程数量
```

### 十、编程模型对比

#### 1. 回调模式（异步I/O常用）

```cpp
void on_read_complete(char* data, size_t len) {
    process_data(data, len);
    
    // 发起下一个异步读
    async_read(fd, buffer, on_read_complete);
}

// 发起异步读
async_read(fd, buffer, on_read_complete);
// 立即返回，继续执行

问题：回调地狱，代码可读性差
```

#### 2. 协程模式（同步写法，异步执行）

```cpp
// C++20协程
Task<void> handle_client(int fd) {
    char buf[1024];
    
    // 看起来是同步的，实际是异步的
    ssize_t n = co_await async_read(fd, buf, sizeof(buf));
    
    if (n > 0) {
        process_data(buf, n);
    }
}

优点：同步的写法，异步的性能
```

#### 3. Future/Promise模式

```cpp
std::future<std::string> result = async_read_file("data.txt");

// 做其他事情
do_other_work();

// 需要结果时获取
std::string data = result.get();  // 如果还没完成会阻塞
```

### 十一、常见误区

**误区1：非阻塞I/O就是异步I/O**
```
错误！
非阻塞I/O仍然是同步的，需要应用程序自己拷贝数据
异步I/O才是真正的异步
```

**误区2：I/O多路复用就不阻塞**
```
错误！
I/O多路复用在select/epoll_wait时会阻塞
在read/write时也会阻塞（数据拷贝）
只是阻塞在多个fd上，而不是单个fd
```

**误区3：异步I/O一定比同步快**
```
不一定！
- 如果只有少量I/O，异步的开销可能更大
- 异步I/O需要更多的内核资源
- 编程复杂度带来的bug可能降低整体性能
```

**误区4：I/O多路复用必须用非阻塞I/O**
```
LT模式下可以用阻塞I/O
ET模式下必须用非阻塞I/O（否则可能饿死）
```

### 十二、总结

#### 选择决策树

```
连接数很少（< 100）且都很活跃？
└─ 是 → 阻塞I/O（简单）

连接数很多（> 1000）？
└─ 是 → I/O多路复用（epoll）
    └─ 追求极致性能？
        └─ 是 → 异步I/O
        └─ 否 → I/O多路复用足够

需要跨平台？
└─ 是 → I/O多路复用（select/poll）
    └─ 性能要求高？
        └─ 是 → 封装平台差异（epoll/kqueue/IOCP）

Windows平台？
└─ 是 → IOCP（Windows的异步I/O）
```

#### 核心要点

1. **阻塞I/O**：简单，但一个连接一个线程
2. **非阻塞I/O**：轮询浪费CPU，很少单独使用
3. **I/O多路复用**：单线程处理多连接，性能优秀
4. **异步I/O**：真正的异步，性能最好，但复杂度高

#### 实践建议

- **默认选择**：I/O多路复用（epoll/kqueue）
- **简单场景**：阻塞I/O + 线程池
- **高性能场景**：异步I/O（io_uring/IOCP）
- **编码友好**：协程 + 异步I/O


---

## 相关笔记
<!-- 自动生成 -->

暂无相关笔记

