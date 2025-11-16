---
created: '2025-10-19'
last_reviewed: null
next_review: '2025-10-19'
review_count: 0
difficulty: medium
mastery_level: 0.0
tags:
- C++
- C++/select()、poll()、epoll()的工作原理和区别.md
related_outlines: []
---
# select()、poll()、epoll()的工作原理和区别

## 标准答案（面试简答）

select、poll、epoll都是I/O多路复用机制，用于同时监控多个文件描述符的状态。select使用固定大小的位图，有1024文件描述符数量限制，每次调用需要将fd_set从用户态拷贝到内核态并遍历所有fd；poll使用链表结构，无数量限制但仍需全量拷贝和遍历；epoll使用红黑树和就绪队列，通过事件驱动避免全量遍历，支持ET和LT两种模式，在大量连接中性能最优。

---

## 详细讲解

### 一、I/O多路复用基本概念

#### 1. 为什么需要I/O多路复用

**传统阻塞I/O的问题**：
```cpp
// 单线程阻塞I/O
int sock = accept(listen_fd, ...);  // 阻塞等待
read(sock, buf, size);              // 阻塞读取

// 一次只能处理一个连接
// 处理多个连接需要多线程/多进程 → 资源开销大
```

**非阻塞I/O的问题**：
```cpp
// 轮询方式
while (true) {
    for (int fd : all_fds) {
        ret = read(fd, buf, size);  // 非阻塞
        if (ret > 0) {
            // 处理数据
        }
    }
}
// CPU空转，资源浪费
```

**I/O多路复用的优势**：
```
- 一个线程监控多个fd
- 阻塞等待，但等待多个fd中任何一个就绪
- 避免轮询，由内核通知
```

### 二、select()详解

#### 1. 函数原型

```cpp
#include <sys/select.h>

int select(int nfds,                    // 最大fd + 1
          fd_set *readfds,              // 读fd集合
          fd_set *writefds,             // 写fd集合  
          fd_set *exceptfds,            // 异常fd集合
          struct timeval *timeout);     // 超时时间

// 操作fd_set的宏
void FD_ZERO(fd_set *set);              // 清空集合
void FD_SET(int fd, fd_set *set);       // 添加fd
void FD_CLR(int fd, fd_set *set);       // 移除fd
int FD_ISSET(int fd, fd_set *set);      // 检查fd是否在集合中

// 返回值：就绪的fd数量，0表示超时，-1表示错误
```

#### 2. 数据结构

```cpp
// fd_set本质是位图
typedef struct {
    long fds_bits[1024/sizeof(long)];
} fd_set;

// 每个bit代表一个fd
// 最大支持1024个fd（FD_SETSIZE）
```

#### 3. 工作原理

```
用户态：
1. 创建fd_set，设置要监控的fd
2. 调用select()

内核态：
3. 将fd_set从用户态拷贝到内核态
4. 遍历所有fd，检查是否就绪
   - 如果有就绪，返回
   - 如果都未就绪，进程休眠，等待设备驱动唤醒
5. 有fd就绪或超时，唤醒进程
6. 将fd_set从内核态拷贝回用户态（标记就绪的fd）

用户态：
7. 遍历fd_set，用FD_ISSET检查哪些fd就绪
8. 处理就绪的fd
```

#### 4. 使用示例

```cpp
#include <sys/select.h>
#include <sys/socket.h>
#include <unistd.h>
#include <iostream>

void select_server(int listen_fd) {
    fd_set read_fds, master_fds;
    FD_ZERO(&master_fds);
    FD_SET(listen_fd, &master_fds);
    
    int max_fd = listen_fd;
    
    while (true) {
        read_fds = master_fds;  // 每次都要重新拷贝（select会修改）
        
        struct timeval timeout;
        timeout.tv_sec = 5;
        timeout.tv_usec = 0;
        
        int ret = select(max_fd + 1, &read_fds, NULL, NULL, &timeout);
        
        if (ret < 0) {
            perror("select error");
            break;
        } else if (ret == 0) {
            std::cout << "timeout" << std::endl;
            continue;
        }
        
        // 遍历所有可能的fd
        for (int fd = 0; fd <= max_fd; fd++) {
            if (FD_ISSET(fd, &read_fds)) {
                if (fd == listen_fd) {
                    // 新连接
                    int client_fd = accept(listen_fd, NULL, NULL);
                    FD_SET(client_fd, &master_fds);
                    if (client_fd > max_fd) {
                        max_fd = client_fd;
                    }
                } else {
                    // 数据到达
                    char buf[1024];
                    int n = read(fd, buf, sizeof(buf));
                    if (n <= 0) {
                        // 关闭连接
                        close(fd);
                        FD_CLR(fd, &master_fds);
                    } else {
                        // 处理数据
                    }
                }
            }
        }
    }
}
```

#### 5. select的缺点

```
1. 文件描述符数量限制：最大1024（FD_SETSIZE）
2. 性能问题：
   - 每次调用都需要从用户态拷贝fd_set到内核态
   - 内核需要遍历所有fd来检查就绪状态
   - 返回时需要将fd_set拷贝回用户态
   - 用户态还要再遍历一次找到就绪的fd
3. fd_set在调用后被修改，需要每次重新设置
4. 不知道哪些fd就绪，需要全量遍历
```

### 三、poll()详解

#### 1. 函数原型

```cpp
#include <poll.h>

int poll(struct pollfd *fds,    // fd数组
        nfds_t nfds,            // 数组大小
        int timeout);           // 超时（毫秒）

struct pollfd {
    int fd;           // 文件描述符
    short events;     // 要监听的事件（输入）
    short revents;    // 实际发生的事件（输出）
};

// events/revents可用的标志
POLLIN      // 有数据可读
POLLOUT     // 可写
POLLERR     // 错误
POLLHUP     // 挂断
POLLNVAL    // 非法请求
```

#### 2. 数据结构

```cpp
// poll使用pollfd数组，不再是位图
struct pollfd fds[NFDS];

// 没有1024的限制（只受系统内存限制）
```

#### 3. 工作原理

```
用户态：
1. 创建pollfd数组，设置fd和events
2. 调用poll()

内核态：
3. 将pollfd数组从用户态拷贝到内核态
4. 遍历所有pollfd，检查对应的fd是否就绪
5. 如果有就绪，设置revents字段
6. 如果都未就绪，进程休眠
7. 有fd就绪或超时，唤醒进程
8. 将pollfd数组拷贝回用户态

用户态：
9. 遍历pollfd数组，检查revents字段
10. 处理就绪的fd
```

#### 4. 使用示例

```cpp
#include <poll.h>
#include <sys/socket.h>
#include <unistd.h>
#include <vector>
#include <iostream>

void poll_server(int listen_fd) {
    std::vector<struct pollfd> fds;
    
    struct pollfd pfd;
    pfd.fd = listen_fd;
    pfd.events = POLLIN;
    fds.push_back(pfd);
    
    while (true) {
        int ret = poll(fds.data(), fds.size(), 5000);  // 5秒超时
        
        if (ret < 0) {
            perror("poll error");
            break;
        } else if (ret == 0) {
            std::cout << "timeout" << std::endl;
            continue;
        }
        
        // 遍历所有fd（但知道就绪的数量）
        for (size_t i = 0; i < fds.size(); i++) {
            if (fds[i].revents & POLLIN) {
                if (fds[i].fd == listen_fd) {
                    // 新连接
                    int client_fd = accept(listen_fd, NULL, NULL);
                    struct pollfd new_pfd;
                    new_pfd.fd = client_fd;
                    new_pfd.events = POLLIN;
                    fds.push_back(new_pfd);
                } else {
                    // 数据到达
                    char buf[1024];
                    int n = read(fds[i].fd, buf, sizeof(buf));
                    if (n <= 0) {
                        close(fds[i].fd);
                        fds.erase(fds.begin() + i);
                        i--;  // 调整索引
                    } else {
                        // 处理数据
                    }
                }
            }
        }
    }
}
```

#### 5. poll相比select的改进

```
优点：
1. 没有1024的fd数量限制
2. events和revents分离，不需要每次重新设置
3. 更清晰的事件类型

缺点（仍然存在）：
1. 仍需要全量拷贝pollfd数组到内核
2. 内核仍需要遍历所有fd
3. 用户态仍需要遍历所有fd找到就绪的
```

### 四、epoll()详解

#### 1. 函数原型

```cpp
#include <sys/epoll.h>

// 创建epoll实例
int epoll_create(int size);     // size已废弃，传>0即可
int epoll_create1(int flags);   // 推荐使用

// 控制epoll
int epoll_ctl(int epfd,         // epoll实例fd
             int op,            // 操作类型
             int fd,            // 目标fd
             struct epoll_event *event);

// op可选值
EPOLL_CTL_ADD   // 添加fd
EPOLL_CTL_MOD   // 修改fd的事件
EPOLL_CTL_DEL   // 删除fd

// 等待事件
int epoll_wait(int epfd,
              struct epoll_event *events,  // 输出缓冲区
              int maxevents,               // 最大事件数
              int timeout);                // 超时（毫秒）

struct epoll_event {
    uint32_t events;      // 事件类型
    epoll_data_t data;    // 用户数据
};

typedef union epoll_data {
    void *ptr;
    int fd;
    uint32_t u32;
    uint64_t u64;
} epoll_data_t;

// events可用标志
EPOLLIN       // 可读
EPOLLOUT      // 可写
EPOLLERR      // 错误
EPOLLHUP      // 挂断
EPOLLET       // 边缘触发模式
EPOLLONESHOT  // 一次性监听
```

#### 2. 数据结构

```cpp
内核中的数据结构：

1. 红黑树（RB-Tree）：
   - 存储所有要监控的fd
   - 快速查找、插入、删除 O(log n)

2. 就绪队列（Ready List）：
   - 双向链表
   - 只存储就绪的fd
   - epoll_wait直接从这里取

3. 等待队列：
   - 存储等待的进程
```

#### 3. 工作原理

```
初始化：
1. epoll_create() 创建epoll实例
   - 在内核创建红黑树和就绪队列

添加监控：
2. epoll_ctl(EPOLL_CTL_ADD) 添加fd
   - 将fd插入红黑树
   - 向设备驱动注册回调函数

等待事件：
3. epoll_wait() 等待事件
   - 检查就绪队列是否为空
   - 如果为空，进程休眠
   - 如果非空，将就绪事件拷贝到用户态

事件发生：
4. 设备驱动调用回调函数
   - 将fd加入就绪队列
   - 唤醒等待的进程

关键区别：
- 不需要每次传递所有fd（只在epoll_ctl时传递一次）
- 不需要遍历所有fd（直接从就绪队列取）
- 只拷贝就绪的事件到用户态
```

#### 4. LT vs ET模式

**LT（Level Triggered，水平触发）** - 默认模式
```
特点：
- 只要fd处于就绪状态，epoll_wait就会返回
- 类似select和poll的行为

示例：
1. fd有100字节数据
2. epoll_wait返回
3. 读取50字节
4. epoll_wait再次返回（还有50字节未读）

优点：编程简单，不容易丢失事件
缺点：可能频繁触发
```

**ET（Edge Triggered，边缘触发）**
```
特点：
- 只在状态改变时触发一次
- fd必须设置为非阻塞
- 必须读/写到EAGAIN

示例：
1. fd有100字节数据 → 触发
2. epoll_wait返回
3. 读取50字节
4. epoll_wait不再返回（状态未改变）
5. 除非新数据到达 → 再次触发

优点：减少触发次数，性能更高
缺点：必须完全读/写完，否则数据会"饿死"
```

#### 5. 使用示例（LT模式）

```cpp
#include <sys/epoll.h>
#include <sys/socket.h>
#include <unistd.h>
#include <fcntl.h>
#include <iostream>

void epoll_server(int listen_fd) {
    // 创建epoll实例
    int epfd = epoll_create1(0);
    if (epfd < 0) {
        perror("epoll_create1");
        return;
    }
    
    // 添加监听socket
    struct epoll_event ev;
    ev.events = EPOLLIN;
    ev.data.fd = listen_fd;
    epoll_ctl(epfd, EPOLL_CTL_ADD, listen_fd, &ev);
    
    const int MAX_EVENTS = 10;
    struct epoll_event events[MAX_EVENTS];
    
    while (true) {
        // 等待事件
        int nfds = epoll_wait(epfd, events, MAX_EVENTS, 5000);
        
        if (nfds < 0) {
            perror("epoll_wait");
            break;
        } else if (nfds == 0) {
            std::cout << "timeout" << std::endl;
            continue;
        }
        
        // 只遍历就绪的fd
        for (int i = 0; i < nfds; i++) {
            if (events[i].data.fd == listen_fd) {
                // 新连接
                int client_fd = accept(listen_fd, NULL, NULL);
                
                struct epoll_event client_ev;
                client_ev.events = EPOLLIN;
                client_ev.data.fd = client_fd;
                epoll_ctl(epfd, EPOLL_CTL_ADD, client_fd, &client_ev);
            } else {
                // 数据到达
                int fd = events[i].data.fd;
                char buf[1024];
                int n = read(fd, buf, sizeof(buf));
                
                if (n <= 0) {
                    epoll_ctl(epfd, EPOLL_CTL_DEL, fd, NULL);
                    close(fd);
                } else {
                    // 处理数据
                }
            }
        }
    }
    
    close(epfd);
}
```

#### 6. ET模式示例

```cpp
// 设置非阻塞
void set_nonblocking(int fd) {
    int flags = fcntl(fd, F_GETFL, 0);
    fcntl(fd, F_SETFL, flags | O_NONBLOCK);
}

void epoll_server_et(int listen_fd) {
    int epfd = epoll_create1(0);
    set_nonblocking(listen_fd);
    
    struct epoll_event ev;
    ev.events = EPOLLIN | EPOLLET;  // ET模式
    ev.data.fd = listen_fd;
    epoll_ctl(epfd, EPOLL_CTL_ADD, listen_fd, &ev);
    
    const int MAX_EVENTS = 10;
    struct epoll_event events[MAX_EVENTS];
    
    while (true) {
        int nfds = epoll_wait(epfd, events, MAX_EVENTS, -1);
        
        for (int i = 0; i < nfds; i++) {
            if (events[i].data.fd == listen_fd) {
                // ET模式下，要循环accept直到EAGAIN
                while (true) {
                    int client_fd = accept(listen_fd, NULL, NULL);
                    if (client_fd < 0) {
                        if (errno == EAGAIN || errno == EWOULDBLOCK) {
                            break;  // 没有更多连接
                        } else {
                            perror("accept");
                            break;
                        }
                    }
                    
                    set_nonblocking(client_fd);
                    struct epoll_event client_ev;
                    client_ev.events = EPOLLIN | EPOLLET;
                    client_ev.data.fd = client_fd;
                    epoll_ctl(epfd, EPOLL_CTL_ADD, client_fd, &client_ev);
                }
            } else {
                // ET模式下，要循环读取直到EAGAIN
                int fd = events[i].data.fd;
                char buf[1024];
                
                while (true) {
                    int n = read(fd, buf, sizeof(buf));
                    if (n < 0) {
                        if (errno == EAGAIN || errno == EWOULDBLOCK) {
                            break;  // 数据读完
                        } else {
                            perror("read");
                            epoll_ctl(epfd, EPOLL_CTL_DEL, fd, NULL);
                            close(fd);
                            break;
                        }
                    } else if (n == 0) {
                        epoll_ctl(epfd, EPOLL_CTL_DEL, fd, NULL);
                        close(fd);
                        break;
                    } else {
                        // 处理数据
                        // 继续读取
                    }
                }
            }
        }
    }
    
    close(epfd);
}
```

### 五、三者对比总结

| 特性           | select             | poll           | epoll              |
| -------------- | ------------------ | -------------- | ------------------ |
| **fd数量限制** | 1024（FD_SETSIZE） | 无限制         | 无限制             |
| **数据结构**   | 位图（fd_set）     | 数组（pollfd） | 红黑树+链表        |
| **fd拷贝**     | 每次全量拷贝       | 每次全量拷贝   | 只在添加时拷贝一次 |
| **内核遍历**   | O(n) 全量遍历      | O(n) 全量遍历  | O(1) 只遍历就绪的  |
| **用户态遍历** | O(n) 找就绪fd      | O(n) 找就绪fd  | O(k) k为就绪数量   |
| **触发模式**   | LT                 | LT             | LT和ET             |
| **性能**       | fd多时性能差       | fd多时性能差   | fd多时性能好       |
| **可移植性**   | 最好（POSIX）      | 好             | 仅Linux            |

**时间复杂度对比**：
```
假设监控n个fd，其中k个就绪（k << n）

select:
- 从用户态拷贝到内核: O(n)
- 内核遍历检查: O(n)
- 从内核拷贝到用户态: O(n)
- 用户态遍历找就绪: O(n)
总计: O(n)

poll:
- 从用户态拷贝到内核: O(n)
- 内核遍历检查: O(n)
- 从内核拷贝到用户态: O(n)
- 用户态遍历找就绪: O(n)
总计: O(n)

epoll:
- 添加fd: O(log n) （红黑树插入）
- 等待事件: O(1) （检查就绪队列）
- 拷贝就绪事件: O(k)
- 用户态处理: O(k)
总计: O(k)
```

### 六、性能对比

```
连接数: 1000
活跃连接: 10

select/poll:
- 每次epoll_wait需要检查1000个fd
- 性能随总连接数线性下降

epoll:
- 每次epoll_wait只需要处理10个就绪fd
- 性能只与活跃连接数相关

这就是epoll在C10K问题上表现优异的原因
```

### 七、使用建议

**使用select的场景**：
- 连接数很少（< 100）
- 需要跨平台（Windows有限支持）
- 代码简单性优先

**使用poll的场景**：
- 连接数较多但不是海量
- 需要类Unix跨平台（不含Windows）

**使用epoll的场景**：
- 连接数很多（>1000）
- 只在Linux上运行
- 追求最高性能

**LT vs ET选择**：
- LT：编程简单，不易出错，性能够用就用LT
- ET：追求极致性能，愿意处理复杂的边界条件

### 八、常见面试问题

**Q: 为什么epoll比select/poll快？**
```
1. 避免全量拷贝：只在添加时拷贝，不是每次wait都拷贝
2. 避免全量遍历：内核维护就绪队列，只返回就绪的
3. 事件驱动：通过回调机制，fd就绪时主动加入就绪队列
```

**Q: epoll的ET模式为什么性能更高？**
```
- LT：可能多次触发同一个fd
- ET：状态变化时才触发一次，减少了系统调用次数
```

**Q: epoll能监控普通文件吗？**
```
不能。epoll只支持支持非阻塞poll的文件类型：
- socket
- pipe
- eventfd
- signalfd
- timerfd

普通文件不支持（总是返回就绪状态）
```

**Q: epoll如何实现线程安全？**
```
epoll本身是线程安全的，但需要注意：
- 多个线程可以等待同一个epfd
- 但同时操作同一个fd要小心（EPOLLONESHOT可以帮助）
```


---

## 相关笔记
<!-- 自动生成 -->

- [epoll为什么比select和poll性能更好？](notes/C++/epoll为什么比select和poll性能更好？.md) - 相似度: 33% | 标签: C++, C++/epoll为什么比select和poll性能更好？.md

