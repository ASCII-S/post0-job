---
created: '2025-10-19'
last_reviewed: null
next_review: '2025-10-19'
review_count: 0
difficulty: medium
mastery_level: 0.0
tags:
- C++
- C++/设计一个简单的shared_ptr.md
related_outlines: []
---
# 设计一个简单的 shared_ptr

## 面试标准答案（可背诵）

**shared_ptr 的核心组件**：
1. **原始指针**：指向管理的对象
2. **引用计数指针**：指向共享的引用计数（在堆上）
3. **引用计数管理**：构造时置1，拷贝时递增，析构时递减，归零时删除对象

**关键操作**：
- **构造函数**：初始化指针，创建引用计数（初值1）
- **拷贝构造/赋值**：共享指针，引用计数+1
- **移动构造/赋值**：转移所有权，不增加计数
- **析构函数**：引用计数-1，归零时删除对象和计数
- **解引用操作**：`*` 和 `->` 运算符

**设计要点**：
1. 引用计数必须在堆上（多个 shared_ptr 共享）
2. 需要处理自赋值问题
3. 线程安全版本需要原子操作
4. 支持删除器需要更复杂的控制块

---

## 详细设计思路

### 1. 设计演进过程

#### 1.1 最简单的版本（仅单个对象）

```cpp
#include <iostream>

// 版本1：最基础的实现
template<typename T>
class SimpleSharedPtr_V1 {
private:
    T* ptr_;              // 原始指针
    size_t* ref_count_;   // 引用计数指针
    
public:
    // 构造函数
    explicit SimpleSharedPtr_V1(T* ptr = nullptr) 
        : ptr_(ptr)
        , ref_count_(new size_t(1)) {  // 创建引用计数，初值1
        
        std::cout << "构造，引用计数 = " << *ref_count_ << std::endl;
    }
    
    // 析构函数
    ~SimpleSharedPtr_V1() {
        release();
    }
    
    // 获取引用计数
    size_t use_count() const {
        return ref_count_ ? *ref_count_ : 0;
    }
    
    // 解引用
    T& operator*() const {
        return *ptr_;
    }
    
    T* operator->() const {
        return ptr_;
    }
    
    // 获取原始指针
    T* get() const {
        return ptr_;
    }
    
private:
    void release() {
        if (ref_count_) {
            --(*ref_count_);
            std::cout << "析构，引用计数 = " << *ref_count_ << std::endl;
            
            if (*ref_count_ == 0) {
                std::cout << "引用计数归零，删除对象" << std::endl;
                delete ptr_;
                delete ref_count_;
            }
        }
    }
};

// 测试
void test_v1() {
    std::cout << "=== 测试版本1 ===" << std::endl;
    
    SimpleSharedPtr_V1<int> ptr1(new int(42));
    std::cout << "值: " << *ptr1 << ", 计数: " << ptr1.use_count() << std::endl;
    
    // 问题：无法拷贝！
    // SimpleSharedPtr_V1<int> ptr2 = ptr1;  // 编译错误
}
```

**问题**：
- ✗ 无法拷贝（没有拷贝构造函数）
- ✗ 无法赋值（没有赋值运算符）
- ✗ 多个指针无法共享同一对象

#### 1.2 添加拷贝支持

```cpp
// 版本2：支持拷贝
template<typename T>
class SimpleSharedPtr_V2 {
private:
    T* ptr_;
    size_t* ref_count_;
    
public:
    // 构造函数
    explicit SimpleSharedPtr_V2(T* ptr = nullptr)
        : ptr_(ptr)
        , ref_count_(ptr ? new size_t(1) : nullptr) {
    }
    
    // 拷贝构造函数
    SimpleSharedPtr_V2(const SimpleSharedPtr_V2& other)
        : ptr_(other.ptr_)
        , ref_count_(other.ref_count_) {
        
        if (ref_count_) {
            ++(*ref_count_);
            std::cout << "拷贝构造，引用计数 = " << *ref_count_ << std::endl;
        }
    }
    
    // 拷贝赋值运算符
    SimpleSharedPtr_V2& operator=(const SimpleSharedPtr_V2& other) {
        if (this != &other) {  // 防止自赋值
            release();  // 释放当前对象
            
            // 复制新对象
            ptr_ = other.ptr_;
            ref_count_ = other.ref_count_;
            
            if (ref_count_) {
                ++(*ref_count_);
                std::cout << "拷贝赋值，引用计数 = " << *ref_count_ << std::endl;
            }
        }
        return *this;
    }
    
    // 析构函数
    ~SimpleSharedPtr_V2() {
        release();
    }
    
    size_t use_count() const {
        return ref_count_ ? *ref_count_ : 0;
    }
    
    T& operator*() const { return *ptr_; }
    T* operator->() const { return ptr_; }
    T* get() const { return ptr_; }
    
private:
    void release() {
        if (ref_count_) {
            --(*ref_count_);
            std::cout << "析构，引用计数 = " << *ref_count_ << std::endl;
            
            if (*ref_count_ == 0) {
                std::cout << "引用计数归零，删除对象" << std::endl;
                delete ptr_;
                delete ref_count_;
            }
        }
    }
};

// 测试
void test_v2() {
    std::cout << "\n=== 测试版本2 ===" << std::endl;
    
    SimpleSharedPtr_V2<int> ptr1(new int(42));
    std::cout << "ptr1 计数: " << ptr1.use_count() << std::endl;  // 1
    
    {
        SimpleSharedPtr_V2<int> ptr2 = ptr1;  // 拷贝
        std::cout << "ptr1 计数: " << ptr1.use_count() << std::endl;  // 2
        std::cout << "ptr2 计数: " << ptr2.use_count() << std::endl;  // 2
        
        SimpleSharedPtr_V2<int> ptr3(new int(100));
        ptr3 = ptr1;  // 赋值
        std::cout << "ptr1 计数: " << ptr1.use_count() << std::endl;  // 3
        
    }  // ptr2 和 ptr3 析构
    
    std::cout << "ptr1 计数: " << ptr1.use_count() << std::endl;  // 1
}
```

**改进**：
- ✓ 支持拷贝构造
- ✓ 支持拷贝赋值
- ✓ 正确管理引用计数

**问题**：
- ✗ 不支持移动语义（C++11）
- ✗ 不是线程安全的

#### 1.3 添加移动语义

```cpp
// 版本3：支持移动语义
template<typename T>
class SimpleSharedPtr_V3 {
private:
    T* ptr_;
    size_t* ref_count_;
    
public:
    // 构造函数
    explicit SimpleSharedPtr_V3(T* ptr = nullptr)
        : ptr_(ptr)
        , ref_count_(ptr ? new size_t(1) : nullptr) {
        std::cout << "构造" << std::endl;
    }
    
    // 拷贝构造
    SimpleSharedPtr_V3(const SimpleSharedPtr_V3& other)
        : ptr_(other.ptr_)
        , ref_count_(other.ref_count_) {
        if (ref_count_) {
            ++(*ref_count_);
        }
        std::cout << "拷贝构造，计数 = " << use_count() << std::endl;
    }
    
    // 移动构造（C++11）
    SimpleSharedPtr_V3(SimpleSharedPtr_V3&& other) noexcept
        : ptr_(other.ptr_)
        , ref_count_(other.ref_count_) {
        
        // 清空源对象
        other.ptr_ = nullptr;
        other.ref_count_ = nullptr;
        
        std::cout << "移动构造，计数 = " << use_count() << std::endl;
    }
    
    // 拷贝赋值
    SimpleSharedPtr_V3& operator=(const SimpleSharedPtr_V3& other) {
        if (this != &other) {
            release();
            
            ptr_ = other.ptr_;
            ref_count_ = other.ref_count_;
            
            if (ref_count_) {
                ++(*ref_count_);
            }
            std::cout << "拷贝赋值，计数 = " << use_count() << std::endl;
        }
        return *this;
    }
    
    // 移动赋值（C++11）
    SimpleSharedPtr_V3& operator=(SimpleSharedPtr_V3&& other) noexcept {
        if (this != &other) {
            release();
            
            ptr_ = other.ptr_;
            ref_count_ = other.ref_count_;
            
            other.ptr_ = nullptr;
            other.ref_count_ = nullptr;
            
            std::cout << "移动赋值，计数 = " << use_count() << std::endl;
        }
        return *this;
    }
    
    // 析构函数
    ~SimpleSharedPtr_V3() {
        release();
    }
    
    size_t use_count() const {
        return ref_count_ ? *ref_count_ : 0;
    }
    
    T& operator*() const { return *ptr_; }
    T* operator->() const { return ptr_; }
    T* get() const { return ptr_; }
    
    // 布尔转换
    explicit operator bool() const {
        return ptr_ != nullptr;
    }
    
    // reset
    void reset(T* ptr = nullptr) {
        release();
        ptr_ = ptr;
        ref_count_ = ptr ? new size_t(1) : nullptr;
    }
    
private:
    void release() {
        if (ref_count_) {
            --(*ref_count_);
            
            if (*ref_count_ == 0) {
                std::cout << "引用计数归零，删除对象" << std::endl;
                delete ptr_;
                delete ref_count_;
            }
        }
        ptr_ = nullptr;
        ref_count_ = nullptr;
    }
};

// 测试
void test_v3() {
    std::cout << "\n=== 测试版本3（移动语义）===" << std::endl;
    
    SimpleSharedPtr_V3<int> ptr1(new int(42));
    std::cout << "ptr1 计数: " << ptr1.use_count() << std::endl;
    
    // 移动
    SimpleSharedPtr_V3<int> ptr2 = std::move(ptr1);
    std::cout << "移动后 ptr1 计数: " << ptr1.use_count() << std::endl;  // 0
    std::cout << "移动后 ptr2 计数: " << ptr2.use_count() << std::endl;  // 1
}
```

**改进**：
- ✓ 支持移动构造和移动赋值
- ✓ 性能优化（移动不增加引用计数）
- ✓ 添加了 `reset()` 和 `operator bool()`

### 2. 完整实现（包含线程安全）

#### 2.1 线程安全版本

```cpp
#include <atomic>
#include <iostream>

template<typename T>
class SharedPtr {
private:
    T* ptr_;                           // 指向对象的指针
    std::atomic<size_t>* ref_count_;   // 原子引用计数
    
public:
    // 默认构造
    SharedPtr() : ptr_(nullptr), ref_count_(nullptr) {}
    
    // 从原始指针构造
    explicit SharedPtr(T* ptr) 
        : ptr_(ptr)
        , ref_count_(ptr ? new std::atomic<size_t>(1) : nullptr) {
        std::cout << "构造 SharedPtr，对象地址: " << ptr_ << std::endl;
    }
    
    // 拷贝构造
    SharedPtr(const SharedPtr& other) 
        : ptr_(other.ptr_)
        , ref_count_(other.ref_count_) {
        if (ref_count_) {
            ref_count_->fetch_add(1, std::memory_order_relaxed);
        }
    }
    
    // 移动构造
    SharedPtr(SharedPtr&& other) noexcept
        : ptr_(other.ptr_)
        , ref_count_(other.ref_count_) {
        other.ptr_ = nullptr;
        other.ref_count_ = nullptr;
    }
    
    // 拷贝赋值
    SharedPtr& operator=(const SharedPtr& other) {
        if (this != &other) {
            release();
            
            ptr_ = other.ptr_;
            ref_count_ = other.ref_count_;
            
            if (ref_count_) {
                ref_count_->fetch_add(1, std::memory_order_relaxed);
            }
        }
        return *this;
    }
    
    // 移动赋值
    SharedPtr& operator=(SharedPtr&& other) noexcept {
        if (this != &other) {
            release();
            
            ptr_ = other.ptr_;
            ref_count_ = other.ref_count_;
            
            other.ptr_ = nullptr;
            other.ref_count_ = nullptr;
        }
        return *this;
    }
    
    // 析构函数
    ~SharedPtr() {
        release();
    }
    
    // 解引用
    T& operator*() const {
        return *ptr_;
    }
    
    T* operator->() const {
        return ptr_;
    }
    
    // 获取原始指针
    T* get() const {
        return ptr_;
    }
    
    // 获取引用计数
    size_t use_count() const {
        return ref_count_ ? ref_count_->load(std::memory_order_relaxed) : 0;
    }
    
    // 是否唯一
    bool unique() const {
        return use_count() == 1;
    }
    
    // 布尔转换
    explicit operator bool() const {
        return ptr_ != nullptr;
    }
    
    // 重置
    void reset(T* ptr = nullptr) {
        release();
        ptr_ = ptr;
        ref_count_ = ptr ? new std::atomic<size_t>(1) : nullptr;
    }
    
    // 交换
    void swap(SharedPtr& other) noexcept {
        std::swap(ptr_, other.ptr_);
        std::swap(ref_count_, other.ref_count_);
    }
    
private:
    void release() {
        if (ref_count_) {
            // 原子递减
            size_t old_count = ref_count_->fetch_sub(1, std::memory_order_acq_rel);
            
            if (old_count == 1) {
                // 最后一个引用，删除对象
                std::cout << "删除对象，地址: " << ptr_ << std::endl;
                delete ptr_;
                delete ref_count_;
            }
        }
        
        ptr_ = nullptr;
        ref_count_ = nullptr;
    }
};

// 辅助函数：make_shared
template<typename T, typename... Args>
SharedPtr<T> make_shared(Args&&... args) {
    return SharedPtr<T>(new T(std::forward<Args>(args)...));
}
```

#### 2.2 测试完整版本

```cpp
class TestClass {
    int value_;
public:
    TestClass(int v) : value_(v) {
        std::cout << "  TestClass(" << value_ << ") 构造" << std::endl;
    }
    
    ~TestClass() {
        std::cout << "  ~TestClass(" << value_ << ") 析构" << std::endl;
    }
    
    void print() const {
        std::cout << "  值: " << value_ << std::endl;
    }
    
    int get() const { return value_; }
};

void test_complete() {
    std::cout << "\n=== 完整版本测试 ===" << std::endl;
    
    // 1. 基本使用
    {
        std::cout << "1. 基本使用" << std::endl;
        auto ptr1 = make_shared<TestClass>(42);
        std::cout << "计数: " << ptr1.use_count() << std::endl;
        ptr1->print();
    }
    
    // 2. 拷贝
    {
        std::cout << "\n2. 拷贝测试" << std::endl;
        auto ptr1 = make_shared<TestClass>(100);
        
        {
            auto ptr2 = ptr1;  // 拷贝
            std::cout << "ptr1 计数: " << ptr1.use_count() << std::endl;  // 2
            std::cout << "ptr2 计数: " << ptr2.use_count() << std::endl;  // 2
        }
        
        std::cout << "ptr2 销毁后 ptr1 计数: " << ptr1.use_count() << std::endl;  // 1
    }
    
    // 3. 移动
    {
        std::cout << "\n3. 移动测试" << std::endl;
        auto ptr1 = make_shared<TestClass>(200);
        std::cout << "ptr1 计数: " << ptr1.use_count() << std::endl;  // 1
        
        auto ptr2 = std::move(ptr1);
        std::cout << "移动后 ptr1 计数: " << ptr1.use_count() << std::endl;  // 0
        std::cout << "移动后 ptr2 计数: " << ptr2.use_count() << std::endl;  // 1
    }
    
    // 4. 赋值
    {
        std::cout << "\n4. 赋值测试" << std::endl;
        auto ptr1 = make_shared<TestClass>(300);
        auto ptr2 = make_shared<TestClass>(400);
        
        ptr2 = ptr1;  // ptr2 原对象被删除，共享 ptr1 的对象
        
        std::cout << "赋值后计数: " << ptr1.use_count() << std::endl;  // 2
    }
    
    // 5. 容器
    {
        std::cout << "\n5. 容器测试" << std::endl;
        std::vector<SharedPtr<TestClass>> vec;
        
        vec.push_back(make_shared<TestClass>(1));
        vec.push_back(make_shared<TestClass>(2));
        vec.push_back(make_shared<TestClass>(3));
        
        for (const auto& ptr : vec) {
            ptr->print();
        }
    }
}
```

### 3. 支持删除器的版本

```cpp
// 更高级的版本：支持自定义删除器
template<typename T>
class SharedPtrWithDeleter {
private:
    T* ptr_;
    std::atomic<size_t>* ref_count_;
    std::function<void(T*)> deleter_;  // 删除器
    
public:
    // 构造函数（带删除器）
    template<typename Deleter>
    SharedPtrWithDeleter(T* ptr, Deleter deleter)
        : ptr_(ptr)
        , ref_count_(ptr ? new std::atomic<size_t>(1) : nullptr)
        , deleter_(deleter) {
    }
    
    // 构造函数（默认删除器）
    explicit SharedPtrWithDeleter(T* ptr = nullptr)
        : ptr_(ptr)
        , ref_count_(ptr ? new std::atomic<size_t>(1) : nullptr)
        , deleter_([](T* p) { delete p; }) {  // 默认使用 delete
    }
    
    // 拷贝构造
    SharedPtrWithDeleter(const SharedPtrWithDeleter& other)
        : ptr_(other.ptr_)
        , ref_count_(other.ref_count_)
        , deleter_(other.deleter_) {
        if (ref_count_) {
            ref_count_->fetch_add(1, std::memory_order_relaxed);
        }
    }
    
    // 析构函数
    ~SharedPtrWithDeleter() {
        release();
    }
    
    // ... 其他成员函数类似
    
    T* get() const { return ptr_; }
    T& operator*() const { return *ptr_; }
    T* operator->() const { return ptr_; }
    size_t use_count() const {
        return ref_count_ ? ref_count_->load() : 0;
    }
    
private:
    void release() {
        if (ref_count_) {
            size_t old_count = ref_count_->fetch_sub(1, std::memory_order_acq_rel);
            
            if (old_count == 1) {
                deleter_(ptr_);  // 使用自定义删除器
                delete ref_count_;
            }
        }
        
        ptr_ = nullptr;
        ref_count_ = nullptr;
    }
};

// 测试删除器
void test_deleter() {
    std::cout << "\n=== 测试删除器 ===" << std::endl;
    
    // 管理 FILE*
    {
        FILE* fp = fopen("test.txt", "w");
        if (fp) {
            SharedPtrWithDeleter<FILE> file(fp, [](FILE* f) {
                std::cout << "关闭文件" << std::endl;
                fclose(f);
            });
            
            fprintf(file.get(), "Hello, SharedPtr!\n");
        }
    }
    
    // 管理数组
    {
        SharedPtrWithDeleter<int> arr(new int[10], [](int* p) {
            std::cout << "删除数组" << std::endl;
            delete[] p;
        });
    }
}
```

### 4. 与标准库的对比

```cpp
void compare_with_std() {
    std::cout << "\n=== 与标准库对比 ===" << std::endl;
    
    // 我们的实现
    {
        auto ptr1 = make_shared<TestClass>(42);
        auto ptr2 = ptr1;
        std::cout << "自定义 SharedPtr 计数: " << ptr1.use_count() << std::endl;
    }
    
    // 标准库
    {
        auto ptr1 = std::make_shared<TestClass>(42);
        auto ptr2 = ptr1;
        std::cout << "标准 shared_ptr 计数: " << ptr1.use_count() << std::endl;
    }
}
```

### 5. 关键设计要点总结

#### 5.1 为什么引用计数在堆上？

```cpp
// ✗ 错误：引用计数在栈上
class BadSharedPtr {
    T* ptr_;
    size_t ref_count_;  // 每个对象都有独立的计数！
    
    // 拷贝时无法共享计数
};

// ✓ 正确：引用计数在堆上
class GoodSharedPtr {
    T* ptr_;
    size_t* ref_count_;  // 指向堆上的共享计数
    
    // 所有拷贝共享同一个计数
};
```

#### 5.2 为什么需要原子操作？

```cpp
// 非线程安全版本
void non_thread_safe() {
    SharedPtr_V3<int> global_ptr(new int(42));
    
    // 多线程同时拷贝
    std::thread t1([&]() {
        for (int i = 0; i < 1000; ++i) {
            auto local = global_ptr;  // 竞态条件！
        }
    });
    
    std::thread t2([&]() {
        for (int i = 0; i < 1000; ++i) {
            auto local = global_ptr;  // 竞态条件！
        }
    });
    
    t1.join();
    t2.join();
    
    // 引用计数可能不正确！
}

// 线程安全版本
void thread_safe() {
    SharedPtr<int> global_ptr(new int(42));  // 使用原子计数
    
    // 多线程安全
    std::thread t1([&]() {
        for (int i = 0; i < 1000; ++i) {
            auto local = global_ptr;  // 原子操作，安全
        }
    });
    
    std::thread t2([&]() {
        for (int i = 0; i < 1000; ++i) {
            auto local = global_ptr;  // 原子操作，安全
        }
    });
    
    t1.join();
    t2.join();
}
```

#### 5.3 自赋值问题

```cpp
void self_assignment() {
    auto ptr = make_shared<TestClass>(100);
    
    // 自赋值
    ptr = ptr;  // 必须正确处理！
    
    // 实现中的检查
    // if (this != &other) { ... }
}
```

### 6. 完整的实现对比

| 特性                        | 简单版本(V1) | 拷贝版本(V2) | 移动版本(V3) | 完整版本 | 标准库 |
| --------------------------- | ------------ | ------------ | ------------ | -------- | ------ |
| **基本功能**                | ✓            | ✓            | ✓            | ✓        | ✓      |
| **拷贝构造/赋值**           | ✗            | ✓            | ✓            | ✓        | ✓      |
| **移动构造/赋值**           | ✗            | ✗            | ✓            | ✓        | ✓      |
| **线程安全**                | ✗            | ✗            | ✗            | ✓        | ✓      |
| **自定义删除器**            | ✗            | ✗            | ✗            | ✓        | ✓      |
| **make_shared 优化**        | ✗            | ✗            | ✗            | 部分     | ✓      |
| **weak_ptr 支持**           | ✗            | ✗            | ✗            | ✗        | ✓      |
| **enable_shared_from_this** | ✗            | ✗            | ✗            | ✗        | ✓      |

### 7. 进阶：控制块设计

```cpp
// 更接近标准库的实现
template<typename T>
struct ControlBlock {
    std::atomic<size_t> strong_count;  // 强引用计数
    std::atomic<size_t> weak_count;    // 弱引用计数
    T* ptr;                             // 对象指针
    std::function<void(T*)> deleter;   // 删除器
    
    ControlBlock(T* p, std::function<void(T*)> d)
        : strong_count(1)
        , weak_count(0)
        , ptr(p)
        , deleter(d) {
    }
    
    void add_strong_ref() {
        strong_count.fetch_add(1, std::memory_order_relaxed);
    }
    
    void release_strong_ref() {
        if (strong_count.fetch_sub(1, std::memory_order_acq_rel) == 1) {
            deleter(ptr);  // 删除对象
            
            if (weak_count.load() == 0) {
                delete this;  // 删除控制块
            }
        }
    }
    
    // weak_ptr 相关方法
    void add_weak_ref() {
        weak_count.fetch_add(1, std::memory_order_relaxed);
    }
    
    void release_weak_ref() {
        if (weak_count.fetch_sub(1, std::memory_order_acq_rel) == 1) {
            if (strong_count.load() == 0) {
                delete this;
            }
        }
    }
};
```

### 8. 核心要点

**设计原则**：
1. **引用计数必须共享**：在堆上分配，所有副本指向同一计数
2. **原子操作保证线程安全**：使用 `std::atomic` 进行计数操作
3. **正确处理边界情况**：自赋值、nullptr、空对象
4. **RAII 管理生命周期**：析构时自动释放
5. **性能优化**：移动语义避免不必要的计数增减

**实现要点**：
```cpp
// 核心数据成员
T* ptr_;                        // 对象指针
std::atomic<size_t>* ref_count_; // 引用计数指针（堆上）

// 核心操作
// 1. 构造：创建计数，初值1
// 2. 拷贝：共享指针，计数+1（原子）
// 3. 移动：转移指针，不增加计数
// 4. 析构：计数-1（原子），归零时删除对象和计数
// 5. 赋值：先释放旧对象，再获取新对象
```

**关键代码模式**：
```cpp
// 拷贝：增加引用
ref_count_->fetch_add(1, std::memory_order_relaxed);

// 析构：减少引用
if (ref_count_->fetch_sub(1, std::memory_order_acq_rel) == 1) {
    delete ptr_;
    delete ref_count_;
}

// 自赋值检查
if (this != &other) {
    // 安全执行赋值
}
```

**记住**：shared_ptr 的核心是"共享的引用计数"——所有副本必须看到并修改同一个计数器，这就是为什么计数器必须在堆上，并且需要原子操作来保证线程安全。

### 9. 面试中的展示顺序

1. **先画图**：解释内存布局（对象、引用计数、多个 shared_ptr）
2. **说明核心组件**：指针 + 引用计数
3. **实现基本版本**：构造、析构、引用计数管理
4. **添加拷贝支持**：拷贝构造和赋值
5. **讨论线程安全**：原子操作的必要性
6. **扩展功能**：移动语义、删除器、控制块

这样循序渐进地展示，能够充分体现你对 shared_ptr 内部机制的深入理解。


---

## 相关笔记
<!-- 自动生成 -->

- [shared_ptr和weak_ptr](notes/C++/shared_ptr和weak_ptr.md) - 相似度: 31% | 标签: C++, C++/shared_ptr和weak_ptr.md

