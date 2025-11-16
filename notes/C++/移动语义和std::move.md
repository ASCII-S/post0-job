---
created: '2025-10-19'
last_reviewed: null
next_review: '2025-10-19'
review_count: 0
difficulty: medium
mastery_level: 0.0
tags:
- C++
- C++/移动语义和std::move.md
related_outlines: []
---
# 移动语义和std::move

## 面试标准答案（可背诵）

**移动语义**是 C++11 引入的性能优化机制，通过"窃取"临时对象的资源而非深拷贝，避免不必要的内存分配和数据复制。

**std::move** 是一个类型转换函数，将左值转换为右值引用，从而触发移动语义。它本身不移动任何东西，只是一个 `static_cast`。

移动语义通过**移动构造函数**和**移动赋值运算符**实现，它们接受右值引用参数，转移资源后将源对象置为有效但未指定状态。

---

## 详细讲解

### 1. 移动语义的动机

#### 1.1 传统拷贝的问题

考虑以下代码：

```cpp
std::vector<int> createVector() {
    std::vector<int> temp(1000000);  // 100万个整数
    // ... 填充数据 ...
    return temp;
}

std::vector<int> vec = createVector();
```

在 C++11 之前，`temp` 返回时需要深拷贝：
1. 分配新内存
2. 复制 100 万个整数
3. 销毁 `temp`

**问题**：`temp` 是临时对象，马上就销毁，为什么还要拷贝？

#### 1.2 移动语义的解决

有了移动语义，编译器会：
1. 直接"偷走" `temp` 的内部指针
2. 将 `temp` 的指针置空
3. 销毁 `temp`（此时已为空，无需释放）

**结果**：只转移了 3 个指针（vector 内部实现），性能提升巨大！

### 2. 移动构造函数

#### 2.1 基本语法

```cpp
class String {
    char* data;
    size_t length;
    
public:
    // 移动构造函数
    String(String&& other) noexcept {
        // 1. 接管资源
        data = other.data;
        length = other.length;
        
        // 2. 将源对象置空（关键步骤！）
        other.data = nullptr;
        other.length = 0;
        
        std::cout << "移动构造\n";
    }
    
    ~String() {
        delete[] data;  // 源对象的 data 已为 nullptr，delete nullptr 安全
    }
};
```

**关键点**：
- 参数类型为 `String&&`（右值引用）
- 标记为 `noexcept`
- 转移资源后必须将源对象置为有效状态

#### 2.2 移动构造 vs 拷贝构造

```cpp
class String {
    char* data;
    size_t length;
    
public:
    // 拷贝构造（深拷贝，开销大）
    String(const String& other) {
        length = other.length;
        data = new char[length + 1];       // 分配新内存
        std::strcpy(data, other.data);     // 复制数据
        std::cout << "拷贝构造\n";
    }
    
    // 移动构造（资源转移，开销小）
    String(String&& other) noexcept {
        data = other.data;        // 直接接管指针
        length = other.length;
        other.data = nullptr;     // 源对象置空
        other.length = 0;
        std::cout << "移动构造\n";
    }
};

// 使用
String s1("hello");
String s2 = s1;                  // 拷贝构造（s1 是左值）
String s3 = String("world");     // 移动构造（临时对象是右值）
String s4 = std::move(s1);       // 移动构造（std::move 转换为右值）
```

### 3. 移动赋值运算符

#### 3.1 基本实现

```cpp
class String {
public:
    // 移动赋值运算符
    String& operator=(String&& other) noexcept {
        if (this != &other) {  // 自赋值检查
            // 1. 释放自己的资源
            delete[] data;
            
            // 2. 接管对方的资源
            data = other.data;
            length = other.length;
            
            // 3. 将对方置空
            other.data = nullptr;
            other.length = 0;
        }
        return *this;
    }
};

// 使用
String s1("hello");
String s2("world");

s1 = std::move(s2);  // 调用移动赋值，s2 变为空
```

#### 3.2 拷贝赋值 vs 移动赋值

```cpp
class String {
public:
    // 拷贝赋值
    String& operator=(const String& other) {
        if (this != &other) {
            delete[] data;
            length = other.length;
            data = new char[length + 1];
            std::strcpy(data, other.data);
        }
        return *this;
    }
    
    // 移动赋值
    String& operator=(String&& other) noexcept {
        if (this != &other) {
            delete[] data;
            data = other.data;
            length = other.length;
            other.data = nullptr;
            other.length = 0;
        }
        return *this;
    }
};

String s1, s2("hello"), s3("world");

s1 = s2;              // 拷贝赋值（s2 是左值）
s1 = std::move(s3);   // 移动赋值（显式转换为右值）
```

### 4. std::move 详解

#### 4.1 std::move 的实现

`std::move` 的简化实现：

```cpp
template<typename T>
typename std::remove_reference<T>::type&& move(T&& arg) noexcept {
    return static_cast<typename std::remove_reference<T>::type&&>(arg);
}
```

**本质**：就是一个 `static_cast`，将任何类型转换为右值引用。

#### 4.2 std::move 不移动任何东西

```cpp
std::string s1 = "hello";
std::string s2 = std::move(s1);

// std::move(s1) 只是返回 s1 的右值引用
// 真正的"移动"发生在 s2 的移动构造函数中
```

**理解要点**：
- `std::move` 只做类型转换，不执行任何操作
- 它告诉编译器："这个对象可以被移动"
- 实际的移动操作由移动构造/赋值函数执行

#### 4.3 std::move 的典型用法

```cpp
// 用法1：显式移动已有对象
std::vector<std::string> vec;
std::string str = "hello";
vec.push_back(std::move(str));  // str 被移动，之后为空字符串

// 用法2：转移 unique_ptr 所有权
std::unique_ptr<int> p1 = std::make_unique<int>(42);
std::unique_ptr<int> p2 = std::move(p1);  // p1 变为空指针

// 用法3：swap 的高效实现
template<typename T>
void swap(T& a, T& b) {
    T temp = std::move(a);
    a = std::move(b);
    b = std::move(temp);
}

// 用法4：从函数参数中移动
void process(std::vector<int> vec) {
    data_ = std::move(vec);  // 避免拷贝 vec
}
```

#### 4.4 何时不应使用 std::move

**错误用法1：返回局部变量**
```cpp
std::string createString() {
    std::string local = "hello";
    return std::move(local);  // ❌ 错误：阻止 RVO 优化
}

// 正确做法：
std::string createString() {
    std::string local = "hello";
    return local;  // ✅ 编译器自动优化（RVO 或移动）
}
```

**错误用法2：移动 const 对象**
```cpp
const std::string s1 = "hello";
std::string s2 = std::move(s1);  // ❌ 实际调用的是拷贝构造！

// std::move(s1) 的类型是 const std::string&&
// 移动构造签名是 String(String&&)，不匹配
// 退化为拷贝构造 String(const String&)
```

**错误用法3：移动后继续使用**
```cpp
std::string s1 = "hello";
std::string s2 = std::move(s1);
std::cout << s1;  // ❌ 不推荐：s1 处于未指定状态
```

### 5. 移动后对象的状态

标准规定：移动后的对象处于**有效但未指定**（valid but unspecified）状态。

```cpp
std::string s1 = "hello";
std::string s2 = std::move(s1);

// s1 的状态：
// ✅ 有效：可以安全销毁、重新赋值
// ❌ 未指定：内容不确定，不应读取或使用

// 安全操作：
s1 = "new value";     // ✅ 重新赋值
s1.clear();           // ✅ 调用成员函数
s1.~basic_string();   // ✅ 析构（自动进行）

// 不安全操作：
std::cout << s1;      // ⚠️ 不推荐：内容未指定
if (s1 == "hello") {} // ⚠️ 不推荐：值不确定
```

**标准库保证**：
- `std::string`：移动后变为空字符串（虽然标准未强制，但实践中通常如此）
- `std::vector`：移动后变为空容器
- `std::unique_ptr`：移动后变为 `nullptr`

### 6. noexcept 的重要性

#### 6.1 为什么需要 noexcept

```cpp
class Widget {
public:
    Widget(Widget&&) noexcept;  // 必须标记为 noexcept
    Widget& operator=(Widget&&) noexcept;
};
```

**原因**：`std::vector` 等容器在重新分配内存时：
- 如果移动构造是 `noexcept`，使用移动（高效）
- 否则为了异常安全，使用拷贝（低效）

#### 6.2 示例

```cpp
class Widget {
public:
    Widget(Widget&& other) {  // 没有 noexcept
        // ...
    }
};

std::vector<Widget> vec;
vec.reserve(100);

for (int i = 0; i < 200; ++i) {
    vec.push_back(Widget());  // 容量不足时重新分配
    // 因为移动构造没有 noexcept，会使用拷贝！
}
```

### 7. 实际应用场景

#### 场景1：容器操作优化

```cpp
std::vector<std::string> vec;

std::string s1 = "hello";
vec.push_back(s1);              // 拷贝 s1
vec.push_back(std::move(s1));   // 移动 s1（s1 变为空）
vec.push_back("world");         // 移动临时对象

vec.emplace_back("efficient");  // 直接构造，更高效
```

#### 场景2：智能指针所有权转移

```cpp
class Factory {
public:
    std::unique_ptr<Widget> create() {
        auto ptr = std::make_unique<Widget>();
        // ... 初始化 ...
        return ptr;  // 自动移动，无拷贝
    }
};

void consume(std::unique_ptr<Widget> ptr) {
    // 接管所有权
}

auto p = Factory().create();
consume(std::move(p));  // 转移所有权，p 变为空
```

#### 场景3：自定义容器/RAII 类

```cpp
class FileHandle {
    int fd;
public:
    FileHandle(const char* path) : fd(open(path, O_RDONLY)) {}
    
    // 禁止拷贝
    FileHandle(const FileHandle&) = delete;
    FileHandle& operator=(const FileHandle&) = delete;
    
    // 允许移动
    FileHandle(FileHandle&& other) noexcept : fd(other.fd) {
        other.fd = -1;
    }
    
    FileHandle& operator=(FileHandle&& other) noexcept {
        if (this != &other) {
            if (fd >= 0) close(fd);
            fd = other.fd;
            other.fd = -1;
        }
        return *this;
    }
    
    ~FileHandle() {
        if (fd >= 0) close(fd);
    }
};

// 使用
FileHandle h1("file.txt");
FileHandle h2 = std::move(h1);  // 转移文件描述符，h1 变为无效
```

#### 场景4：参数传递优化

```cpp
class Buffer {
    std::vector<char> data;
public:
    // 按值传递 + 移动，支持左值和右值
    void setData(std::vector<char> d) {
        data = std::move(d);  // 从参数移动
    }
};

Buffer buf;
std::vector<char> vec(1000);

buf.setData(vec);              // 拷贝 vec 到参数，然后移动到 data
buf.setData(std::move(vec));   // 移动 vec 到参数，然后移动到 data
buf.setData({1, 2, 3});        // 移动临时对象
```

### 8. 移动语义与五法则

如果类管理资源，应遵循**五法则**（Rule of Five）：

```cpp
class Resource {
    int* data;
public:
    // 1. 构造函数
    Resource() : data(new int(0)) {}
    
    // 2. 析构函数
    ~Resource() { delete data; }
    
    // 3. 拷贝构造
    Resource(const Resource& other) : data(new int(*other.data)) {}
    
    // 4. 拷贝赋值
    Resource& operator=(const Resource& other) {
        if (this != &other) {
            delete data;
            data = new int(*other.data);
        }
        return *this;
    }
    
    // 5. 移动构造
    Resource(Resource&& other) noexcept : data(other.data) {
        other.data = nullptr;
    }
    
    // 6. 移动赋值
    Resource& operator=(Resource&& other) noexcept {
        if (this != &other) {
            delete data;
            data = other.data;
            other.data = nullptr;
        }
        return *this;
    }
};
```

**或者遵循零法则**：使用标准库容器，让编译器自动生成：

```cpp
class Widget {
    std::string name;
    std::vector<int> data;
    std::unique_ptr<Resource> res;
    
    // 编译器自动生成高效的移动构造和移动赋值
    // 因为所有成员都支持移动语义
};
```

### 9. 性能对比

```cpp
#include <chrono>

class BigObject {
    std::vector<int> data;
public:
    BigObject() : data(10000000) {}  // 1000万个整数
    
    BigObject(const BigObject& other) : data(other.data) {
        std::cout << "拷贝：分配 " << data.size() * sizeof(int) / 1024 / 1024 << " MB\n";
    }
    
    BigObject(BigObject&& other) noexcept : data(std::move(other.data)) {
        std::cout << "移动：仅转移指针\n";
    }
};

// 测试
auto start = std::chrono::high_resolution_clock::now();

BigObject obj1;
BigObject obj2 = std::move(obj1);  // 移动：几乎瞬间完成

auto end = std::chrono::high_resolution_clock::now();
auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
std::cout << "耗时：" << duration.count() << " 微秒\n";

// 输出：
// 移动：仅转移指针
// 耗时：1 微秒

// 如果用拷贝：
// 拷贝：分配 38 MB
// 耗时：15000 微秒（15 毫秒）
```

---

## 总结

### 核心概念

1. **移动语义**：通过转移资源而非拷贝来优化性能
2. **std::move**：类型转换工具，将左值转为右值引用
3. **移动构造/赋值**：接受右值引用，实现资源转移

### 关键要点

- `std::move` 本身不移动任何东西，只做类型转换
- 移动操作应标记为 `noexcept`
- 移动后对象处于有效但未指定状态
- 不要对返回值使用 `std::move`（会阻止优化）
- const 对象无法移动

### 使用场景

- 容器元素插入/删除
- 智能指针所有权转移
- 函数返回大对象
- RAII 类的资源管理

### 最佳实践

- 遵循五法则或零法则
- 合理使用 `std::move`，避免过度使用
- 优先使用 `emplace` 系列函数
- 移动后的对象不要再使用其值


---

## 相关笔记
<!-- 自动生成 -->

- [右值引用与移动语义](notes/C++/右值引用与移动语义.md) - 相似度: 39% | 标签: C++, C++/右值引用与移动语义.md
- [右值引用](notes/C++/右值引用.md) - 相似度: 33% | 标签: C++, C++/右值引用.md

