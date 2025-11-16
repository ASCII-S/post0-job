---
created: '2025-10-19'
last_reviewed: null
next_review: '2025-10-19'
review_count: 0
difficulty: medium
mastery_level: 0.0
tags:
- C++
- C++/Socket的创建、绑定、监听、连接流程.md
related_outlines: []
---
# Socket的创建、绑定、监听、连接流程

## 1. Socket基本概念

### 什么是Socket？
- Socket是应用层与TCP/IP协议族通信的中间软件抽象层
- 它是一种进程间通信机制，可以实现不同主机间的进程通信
- Socket提供了一组API，使得网络编程变得相对简单

### TCP vs UDP Socket
| 特性     | TCP Socket   | UDP Socket   |
| -------- | ------------ | ------------ |
| 连接性   | 面向连接     | 无连接       |
| 可靠性   | 可靠传输     | 不可靠传输   |
| 顺序性   | 保证顺序     | 不保证顺序   |
| 流控制   | 有流控制     | 无流控制     |
| 重传机制 | 自动重传     | 无重传       |
| 适用场景 | 可靠性要求高 | 实时性要求高 |

## 2. TCP Socket编程流程

### 服务端流程
```
socket() → bind() → listen() → accept() → read()/write() → close()
```

### 客户端流程
```
socket() → connect() → read()/write() → close()
```

## 3. 详细API说明

### 3.1 socket() - 创建套接字
```cpp
int socket(int domain, int type, int protocol);
```
- **domain**: 协议族 (AF_INET: IPv4, AF_INET6: IPv6)
- **type**: 套接字类型 (SOCK_STREAM: TCP, SOCK_DGRAM: UDP)
- **protocol**: 协议 (通常为0，让系统选择合适的协议)
- **返回值**: 成功返回文件描述符，失败返回-1

### 3.2 bind() - 绑定地址
```cpp
int bind(int sockfd, const struct sockaddr *addr, socklen_t addrlen);
```
- **sockfd**: socket文件描述符
- **addr**: 要绑定的地址结构
- **addrlen**: 地址结构长度
- **作用**: 将socket与特定的IP地址和端口绑定

### 3.3 listen() - 监听连接
```cpp
int listen(int sockfd, int backlog);
```
- **sockfd**: socket文件描述符
- **backlog**: 连接队列的最大长度
- **作用**: 将socket标记为被动socket，用于接受连接请求

### 3.4 accept() - 接受连接
```cpp
int accept(int sockfd, struct sockaddr *addr, socklen_t *addrlen);
```
- **sockfd**: 监听socket的文件描述符
- **addr**: 用于返回客户端地址信息
- **addrlen**: 地址结构长度
- **返回值**: 成功返回新的socket描述符，用于与客户端通信

### 3.5 connect() - 建立连接（客户端）
```cpp
int connect(int sockfd, const struct sockaddr *addr, socklen_t addrlen);
```
- **sockfd**: socket文件描述符
- **addr**: 服务器地址结构
- **addrlen**: 地址结构长度
- **作用**: 客户端向服务器发起连接请求

## 4. 完整代码示例

### 4.1 TCP服务端
```cpp
#include <iostream>
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <unistd.h>
#include <cstring>

int main() {
    // 1. 创建socket
    int server_fd = socket(AF_INET, SOCK_STREAM, 0);
    if (server_fd < 0) {
        perror("socket failed");
        return -1;
    }
    
    // 设置地址重用
    int opt = 1;
    setsockopt(server_fd, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt));
    
    // 2. 绑定地址
    struct sockaddr_in address;
    address.sin_family = AF_INET;
    address.sin_addr.s_addr = INADDR_ANY;  // 绑定所有可用接口
    address.sin_port = htons(8080);        // 端口8080
    
    if (bind(server_fd, (struct sockaddr*)&address, sizeof(address)) < 0) {
        perror("bind failed");
        close(server_fd);
        return -1;
    }
    
    // 3. 监听
    if (listen(server_fd, 3) < 0) {
        perror("listen failed");
        close(server_fd);
        return -1;
    }
    
    std::cout << "服务器正在监听端口8080..." << std::endl;
    
    // 4. 接受连接
    struct sockaddr_in client_addr;
    socklen_t client_len = sizeof(client_addr);
    
    int client_fd = accept(server_fd, (struct sockaddr*)&client_addr, &client_len);
    if (client_fd < 0) {
        perror("accept failed");
        close(server_fd);
        return -1;
    }
    
    std::cout << "客户端连接成功!" << std::endl;
    
    // 5. 数据传输
    char buffer[1024] = {0};
    int bytes_read = read(client_fd, buffer, 1024);
    std::cout << "收到消息: " << buffer << std::endl;
    
    const char* response = "Hello from server!";
    send(client_fd, response, strlen(response), 0);
    
    // 6. 关闭连接
    close(client_fd);
    close(server_fd);
    
    return 0;
}
```

### 4.2 TCP客户端
```cpp
#include <iostream>
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <unistd.h>
#include <cstring>

int main() {
    // 1. 创建socket
    int sock = socket(AF_INET, SOCK_STREAM, 0);
    if (sock < 0) {
        perror("socket creation failed");
        return -1;
    }
    
    // 2. 设置服务器地址
    struct sockaddr_in serv_addr;
    serv_addr.sin_family = AF_INET;
    serv_addr.sin_port = htons(8080);
    
    // 将IP地址从字符串转换为网络字节序
    if (inet_pton(AF_INET, "127.0.0.1", &serv_addr.sin_addr) <= 0) {
        perror("Invalid address");
        close(sock);
        return -1;
    }
    
    // 3. 连接服务器
    if (connect(sock, (struct sockaddr*)&serv_addr, sizeof(serv_addr)) < 0) {
        perror("Connection failed");
        close(sock);
        return -1;
    }
    
    std::cout << "连接服务器成功!" << std::endl;
    
    // 4. 发送数据
    const char* message = "Hello from client!";
    send(sock, message, strlen(message), 0);
    std::cout << "消息已发送" << std::endl;
    
    // 5. 接收响应
    char buffer[1024] = {0};
    int bytes_read = read(sock, buffer, 1024);
    std::cout << "服务器响应: " << buffer << std::endl;
    
    // 6. 关闭连接
    close(sock);
    
    return 0;
}
```

## 5. 关键概念详解

### 5.1 三次握手过程
1. **SYN**: 客户端发送SYN包到服务器
2. **SYN-ACK**: 服务器回复SYN-ACK包
3. **ACK**: 客户端发送ACK包，连接建立

### 5.2 四次挥手过程
1. **FIN**: 主动关闭方发送FIN包
2. **ACK**: 被动关闭方回复ACK包
3. **FIN**: 被动关闭方发送FIN包
4. **ACK**: 主动关闭方回复ACK包

### 5.3 Socket状态转换
- **CLOSED** → **LISTEN** (服务端调用listen)
- **CLOSED** → **SYN_SENT** (客户端调用connect)
- **LISTEN** → **SYN_RCVD** (服务端收到SYN)
- **SYN_SENT** → **ESTABLISHED** (客户端收到SYN-ACK)
- **SYN_RCVD** → **ESTABLISHED** (服务端收到ACK)

## 6. 常见面试问题与答案

### Q1: socket、bind、listen、accept的作用分别是什么？
**答案:**
- **socket()**: 创建一个套接字，返回文件描述符
- **bind()**: 将套接字与特定的IP地址和端口绑定
- **listen()**: 将套接字标记为被动套接字，准备接受连接请求
- **accept()**: 从连接队列中取出一个连接请求，建立新的连接

### Q2: 为什么需要listen()函数？
**答案:**
- listen()将socket从主动模式转换为被动模式
- 设置连接队列的最大长度，管理同时到达的连接请求
- 没有调用listen()的socket无法接受连接请求

### Q3: accept()返回的是什么？
**答案:**
- accept()返回一个新的socket文件描述符
- 这个新的socket专门用于与特定客户端通信
- 原来的监听socket继续监听新的连接请求

### Q4: backlog参数的作用是什么？
**答案:**
- backlog指定连接队列的最大长度
- 当连接请求超过这个数量时，新的请求会被拒绝
- 实际队列长度可能受系统参数限制

### Q5: 如何处理多个客户端连接？
**答案:**
1. **多进程**: 每个连接fork一个子进程
2. **多线程**: 每个连接创建一个线程
3. **I/O多路复用**: 使用select/poll/epoll
4. **异步I/O**: 使用异步编程模型

### Q6: SO_REUSEADDR的作用是什么？
**答案:**
- 允许重用处于TIME_WAIT状态的地址
- 避免"Address already in use"错误
- 在服务器重启时特别有用

## 7. 错误处理和最佳实践

### 7.1 常见错误处理
```cpp
// 检查socket创建
if (sockfd < 0) {
    perror("socket failed");
    exit(EXIT_FAILURE);
}

// 检查bind结果
if (bind(sockfd, (struct sockaddr*)&address, sizeof(address)) < 0) {
    perror("bind failed");
    close(sockfd);
    exit(EXIT_FAILURE);
}

// 检查连接结果
if (connect(sockfd, (struct sockaddr*)&serv_addr, sizeof(serv_addr)) < 0) {
    perror("connect failed");
    close(sockfd);
    exit(EXIT_FAILURE);
}
```

### 7.2 最佳实践
1. **总是检查返回值**: 所有系统调用都可能失败
2. **正确关闭socket**: 使用close()释放资源
3. **设置超时**: 避免无限等待
4. **处理SIGPIPE**: 避免程序崩溃
5. **使用非阻塞I/O**: 提高程序响应性

### 7.3 性能优化
```cpp
// 设置发送缓冲区大小
int sendbuf = 32768;
setsockopt(sockfd, SOL_SOCKET, SO_SNDBUF, &sendbuf, sizeof(sendbuf));

// 设置接收缓冲区大小
int recvbuf = 32768;
setsockopt(sockfd, SOL_SOCKET, SO_RCVBUF, &recvbuf, sizeof(recvbuf));

// 禁用Nagle算法（适用于低延迟要求）
int flag = 1;
setsockopt(sockfd, IPPROTO_TCP, TCP_NODELAY, &flag, sizeof(flag));
```

## 8. UDP Socket编程

### 8.1 UDP特点
- 无连接：不需要建立连接
- 不可靠：不保证数据到达
- 无序：不保证数据顺序
- 高效：开销小，速度快

### 8.2 UDP编程流程
```cpp
// 服务端
socket() → bind() → recvfrom()/sendto() → close()

// 客户端  
socket() → sendto()/recvfrom() → close()
```

### 8.3 UDP示例代码
```cpp
// UDP服务端
int sockfd = socket(AF_INET, SOCK_DGRAM, 0);
// ... bind() ...

char buffer[1024];
struct sockaddr_in client_addr;
socklen_t len = sizeof(client_addr);

// 接收数据
ssize_t n = recvfrom(sockfd, buffer, sizeof(buffer), 0, 
                     (struct sockaddr*)&client_addr, &len);

// 发送响应
sendto(sockfd, "response", 8, 0, 
       (struct sockaddr*)&client_addr, len);
```

## 9. 高级主题

### 9.1 I/O多路复用
```cpp
#include <sys/select.h>

fd_set readfds;
FD_ZERO(&readfds);
FD_SET(sockfd, &readfds);

int activity = select(max_fd + 1, &readfds, NULL, NULL, NULL);
if (FD_ISSET(sockfd, &readfds)) {
    // 有数据可读
}
```

### 9.2 非阻塞I/O
```cpp
#include <fcntl.h>

// 设置非阻塞模式
int flags = fcntl(sockfd, F_GETFL, 0);
fcntl(sockfd, F_SETFL, flags | O_NONBLOCK);
```

这份文档涵盖了Socket编程的核心概念、详细流程、代码示例和常见面试问题，应该能很好地帮助你准备相关的技术面试。

---

## 相关笔记
<!-- 自动生成 -->

暂无相关笔记

