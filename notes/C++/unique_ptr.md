---
created: '2025-10-19'
last_reviewed: null
next_review: '2025-10-19'
review_count: 0
difficulty: medium
mastery_level: 0.0
tags:
- C++
- C++/unique_ptr.md
related_outlines: []
---
# std::unique_ptr

## 面试标准答案（可背诵）

**`std::unique_ptr` 是C++11引入的独占所有权智能指针，用于自动管理动态分配的对象。**

**核心特性**：
1. **独占所有权**：同一时刻只能有一个 `unique_ptr` 拥有对象，不可拷贝，只能移动
2. **零开销**：没有引用计数，`sizeof(unique_ptr)` 等于原始指针大小（无自定义删除器时）
3. **RAII管理**：离开作用域时自动释放资源，异常安全
4. **移动语义**：支持所有权转移（`std::move`），可以作为函数返回值
5. **自定义删除器**：可指定特殊的资源释放方式

**使用场景**：
- 管理独占资源（对象只有一个所有者）
- 函数返回值传递所有权
- 容器中存储多态对象
- 替代原始指针的默认选择

**优先使用 `unique_ptr`**，只在需要共享所有权时才使用 `shared_ptr`。

---

## 详细解析

### 1. 基本概念和用法

#### 1.1 创建 unique_ptr

```cpp
#include <memory>
#include <iostream>

void basic_usage() {
    // 方式1：使用 make_unique（C++14，推荐）
    auto ptr1 = std::make_unique<int>(42);
    
    // 方式2：直接构造（C++11）
    std::unique_ptr<int> ptr2(new int(42));
    
    // 方式3：默认初始化为 nullptr
    std::unique_ptr<int> ptr3;  // 等价于 nullptr
    std::unique_ptr<int> ptr4 = nullptr;
    
    // 访问
    std::cout << *ptr1 << std::endl;  // 42
    
    // 检查是否为空
    if (ptr1) {
        std::cout << "ptr1不为空" << std::endl;
    }
    
    // 获取原始指针（但不转移所有权）
    int* raw = ptr1.get();
    
}  // 离开作用域，所有对象自动释放
```

#### 1.2 为什么优先使用 make_unique

```cpp
class Widget {
public:
    Widget(int x, int y) {
        std::cout << "Widget(" << x << ", " << y << ")" << std::endl;
    }
    ~Widget() {
        std::cout << "~Widget()" << std::endl;
    }
};

// ✓ 推荐：使用 make_unique
auto widget1 = std::make_unique<Widget>(10, 20);

// ✗ 不推荐：直接 new
std::unique_ptr<Widget> widget2(new Widget(10, 20));

// 原因1：异常安全
void risky_function(std::unique_ptr<Widget> w1, std::unique_ptr<Widget> w2);

// 潜在问题（C++17前可能出现）
risky_function(
    std::unique_ptr<Widget>(new Widget(1, 2)),  // 可能内存泄漏
    std::unique_ptr<Widget>(new Widget(3, 4))
);
// 如果第一个new成功，但第二个new抛出异常，第一个对象可能泄漏

// ✓ 安全写法
risky_function(
    std::make_unique<Widget>(1, 2),
    std::make_unique<Widget>(3, 4)
);

// 原因2：代码更简洁
auto ptr = std::make_unique<std::vector<int>>(100, 0);
// vs
std::unique_ptr<std::vector<int>> ptr2(new std::vector<int>(100, 0));
```

### 2. 独占所有权和移动语义

#### 2.1 不可拷贝，只能移动

```cpp
void ownership_demo() {
    auto ptr1 = std::make_unique<int>(42);
    
    // ✗ 编译错误：不能拷贝
    // auto ptr2 = ptr1;
    // std::unique_ptr<int> ptr3(ptr1);
    
    // ✓ 可以移动：转移所有权
    auto ptr2 = std::move(ptr1);
    
    // 现在 ptr1 为 nullptr，ptr2 拥有对象
    if (!ptr1) {
        std::cout << "ptr1 现在是空的" << std::endl;
    }
    if (ptr2) {
        std::cout << "*ptr2 = " << *ptr2 << std::endl;  // 42
    }
}
```

#### 2.2 作为函数参数

```cpp
class Data {
public:
    Data(int val) : value(val) {}
    int value;
};

// 方式1：按值传递（转移所有权）
void take_ownership(std::unique_ptr<Data> ptr) {
    if (ptr) {
        std::cout << "拥有数据: " << ptr->value << std::endl;
    }
    // 函数结束，ptr析构，对象被删除
}

// 方式2：按引用传递（不转移所有权，只是使用）
void use_data(const std::unique_ptr<Data>& ptr) {
    if (ptr) {
        std::cout << "使用数据: " << ptr->value << std::endl;
    }
    // 函数结束，ptr仍然有效
}

// 方式3：传递原始指针（不涉及所有权）
void process_data(Data* ptr) {
    if (ptr) {
        std::cout << "处理数据: " << ptr->value << std::endl;
    }
}

void parameter_passing() {
    auto data = std::make_unique<Data>(100);
    
    // 使用但不转移所有权
    use_data(data);  // data仍然有效
    process_data(data.get());  // data仍然有效
    
    // 转移所有权
    take_ownership(std::move(data));
    
    // data现在是nullptr，不能再使用
    if (!data) {
        std::cout << "data已被转移" << std::endl;
    }
}
```

#### 2.3 作为函数返回值

```cpp
// 返回 unique_ptr：转移所有权给调用者
std::unique_ptr<Data> create_data(int value) {
    auto data = std::make_unique<Data>(value);
    
    // 一些初始化操作
    data->value *= 2;
    
    return data;  // 自动移动，无需 std::move
}

// 工厂模式
std::unique_ptr<Data> factory(const std::string& type) {
    if (type == "small") {
        return std::make_unique<Data>(10);
    } else if (type == "large") {
        return std::make_unique<Data>(1000);
    }
    return nullptr;  // 表示创建失败
}

void return_value_demo() {
    auto data1 = create_data(50);
    std::cout << data1->value << std::endl;  // 100
    
    auto data2 = factory("large");
    if (data2) {
        std::cout << data2->value << std::endl;  // 1000
    }
}
```

### 3. 管理数组

#### 3.1 数组版本的 unique_ptr

```cpp
void array_management() {
    // 管理动态数组
    auto arr1 = std::make_unique<int[]>(10);
    
    // 使用数组下标访问
    for (int i = 0; i < 10; ++i) {
        arr1[i] = i * i;
    }
    
    std::cout << arr1[5] << std::endl;  // 25
    
    // C++11 方式
    std::unique_ptr<int[]> arr2(new int[10]);
    
}  // 自动调用 delete[]，而非 delete

// ⚠️ 注意：一般情况下优先使用 std::vector
void prefer_vector() {
    // 推荐：使用 vector（更安全、功能更多）
    std::vector<int> vec(10);
    
    // 只在特殊情况下使用 unique_ptr<T[]>
    // 例如：需要与C API交互
}
```

#### 3.2 数组 vs 单对象

```cpp
// 单对象：使用 delete
std::unique_ptr<int> single = std::make_unique<int>(42);

// 数组：使用 delete[]
std::unique_ptr<int[]> array = std::make_unique<int[]>(10);

// ⚠️ 混淆会导致未定义行为
// std::unique_ptr<int> wrong(new int[10]);  // 错误！会调用delete而非delete[]
// std::unique_ptr<int[]> wrong2(new int(42));  // 错误！会调用delete[]而非delete
```

### 4. 自定义删除器

#### 4.1 基本用法

```cpp
#include <cstdio>

// 删除器函数
void file_closer(FILE* fp) {
    if (fp) {
        std::cout << "关闭文件" << std::endl;
        fclose(fp);
    }
}

void custom_deleter_function() {
    // 使用函数作为删除器
    std::unique_ptr<FILE, decltype(&file_closer)> file(
        fopen("data.txt", "w"),
        &file_closer
    );
    
    if (file) {
        fprintf(file.get(), "Hello, RAII!\n");
    }
    
}  // 自动调用 file_closer
```

#### 4.2 Lambda 删除器

```cpp
void custom_deleter_lambda() {
    // 使用 lambda 作为删除器
    auto deleter = [](int* ptr) {
        std::cout << "删除值: " << *ptr << std::endl;
        delete ptr;
    };
    
    std::unique_ptr<int, decltype(deleter)> ptr(new int(42), deleter);
    
}  // 调用 lambda 删除器

// 更简洁的写法（C++11）
void lambda_deleter_inline() {
    auto ptr = std::unique_ptr<FILE, decltype(&fclose)>(
        fopen("data.txt", "r"),
        &fclose
    );
}
```

#### 4.3 自定义删除器类

```cpp
// 删除器类
struct DatabaseDeleter {
    void operator()(void* db) const {
        if (db) {
            std::cout << "关闭数据库连接" << std::endl;
            // db_close(db);
        }
    }
};

void custom_deleter_class() {
    std::unique_ptr<void, DatabaseDeleter> db(
        /* db_open() */ nullptr,
        DatabaseDeleter{}
    );
    
    // 使用数据库
    
}  // 自动调用 DatabaseDeleter::operator()

// 管理C资源的通用模式
template<typename T, typename Deleter>
auto make_resource(T* resource, Deleter deleter) {
    return std::unique_ptr<T, Deleter>(resource, deleter);
}

void resource_wrapper() {
    auto socket = make_resource(
        /* socket_create() */ (int*)nullptr,
        [](int* sock) { 
            std::cout << "关闭socket" << std::endl;
            // socket_close(*sock); 
        }
    );
}
```

### 5. 成员函数详解

#### 5.1 常用成员函数

```cpp
void member_functions() {
    auto ptr = std::make_unique<int>(42);
    
    // get(): 获取原始指针（不转移所有权）
    int* raw = ptr.get();
    std::cout << *raw << std::endl;  // 42
    
    // operator*: 解引用
    std::cout << *ptr << std::endl;  // 42
    
    // operator->: 成员访问
    auto widget = std::make_unique<Widget>(1, 2);
    // widget->some_method();
    
    // operator bool: 检查是否为空
    if (ptr) {
        std::cout << "ptr 不为空" << std::endl;
    }
    
    // release(): 释放所有权，返回原始指针
    int* released = ptr.release();
    // 现在 ptr 为 nullptr，需要手动管理 released
    delete released;
    
    // reset(): 替换管理的对象
    auto ptr2 = std::make_unique<int>(100);
    ptr2.reset(new int(200));  // 旧对象(100)被删除
    std::cout << *ptr2 << std::endl;  // 200
    
    ptr2.reset();  // 删除对象，ptr2变为nullptr
    
    // swap(): 交换两个unique_ptr
    auto ptr3 = std::make_unique<int>(1);
    auto ptr4 = std::make_unique<int>(2);
    ptr3.swap(ptr4);
    std::cout << *ptr3 << " " << *ptr4 << std::endl;  // 2 1
}
```

#### 5.2 release() 的使用场景

```cpp
// 场景1：转移到传统API
void legacy_api(int* ptr) {
    // 旧的C风格API，接管所有权
    // ... 使用ptr
    delete ptr;  // API负责删除
}

void use_legacy_api() {
    auto ptr = std::make_unique<int>(42);
    
    // 释放所有权，传递给旧API
    legacy_api(ptr.release());
    
    // ptr现在是nullptr
}

// 场景2：条件性所有权转移
std::unique_ptr<Data> conditional_ownership(bool take_ownership) {
    auto data = std::make_unique<Data>(100);
    
    if (take_ownership) {
        return data;  // 转移所有权
    } else {
        process_data(data.get());  // 只使用，不转移
        return nullptr;
    }
}
```

### 6. 在类中使用 unique_ptr

#### 6.1 成员变量

```cpp
class ResourceOwner {
    std::unique_ptr<int> data_;
    std::unique_ptr<std::vector<int>> items_;
    
public:
    // 构造函数
    ResourceOwner(int value) 
        : data_(std::make_unique<int>(value))
        , items_(std::make_unique<std::vector<int>>(100)) {
    }
    
    // 默认析构函数自动释放资源
    ~ResourceOwner() = default;
    
    // 禁止拷贝（unique_ptr不可拷贝）
    ResourceOwner(const ResourceOwner&) = delete;
    ResourceOwner& operator=(const ResourceOwner&) = delete;
    
    // 支持移动（unique_ptr可移动）
    ResourceOwner(ResourceOwner&&) = default;
    ResourceOwner& operator=(ResourceOwner&&) = default;
    
    int get_value() const { return *data_; }
    
    void set_value(int value) { *data_ = value; }
};

void class_member_demo() {
    ResourceOwner owner(42);
    std::cout << owner.get_value() << std::endl;  // 42
    
    // 移动
    ResourceOwner owner2 = std::move(owner);
    std::cout << owner2.get_value() << std::endl;  // 42
    
}  // owner2析构，自动释放所有资源
```

#### 6.2 Pimpl 惯用法（编译防火墙）

```cpp
// Widget.h
class Widget {
public:
    Widget();
    ~Widget();  // 必须在.cpp中定义，因为需要完整的Impl定义
    
    Widget(Widget&&);  // 必须在.cpp中定义
    Widget& operator=(Widget&&);  // 必须在.cpp中定义
    
    void do_something();
    
private:
    class Impl;  // 前向声明
    std::unique_ptr<Impl> pimpl_;
};

// Widget.cpp
class Widget::Impl {
public:
    void do_something_impl() {
        std::cout << "实现细节" << std::endl;
    }
    
    // 私有实现细节，对头文件用户隐藏
    std::vector<int> data_;
    std::string name_;
};

Widget::Widget() : pimpl_(std::make_unique<Impl>()) {}

Widget::~Widget() = default;  // 必须在Impl完整定义后

Widget::Widget(Widget&&) = default;
Widget& Widget::operator=(Widget&&) = default;

void Widget::do_something() {
    pimpl_->do_something_impl();
}

// 优势：
// 1. 头文件不需要包含实现细节的头文件
// 2. 修改实现不需要重新编译使用Widget的代码
// 3. 减少编译依赖，加快编译速度
```

### 7. 在容器中使用

#### 7.1 存储多态对象

```cpp
class Animal {
public:
    virtual ~Animal() = default;
    virtual void make_sound() const = 0;
};

class Dog : public Animal {
public:
    void make_sound() const override {
        std::cout << "Woof!" << std::endl;
    }
};

class Cat : public Animal {
public:
    void make_sound() const override {
        std::cout << "Meow!" << std::endl;
    }
};

void polymorphic_container() {
    // 存储不同类型的派生类对象
    std::vector<std::unique_ptr<Animal>> animals;
    
    animals.push_back(std::make_unique<Dog>());
    animals.push_back(std::make_unique<Cat>());
    animals.push_back(std::make_unique<Dog>());
    
    // 多态调用
    for (const auto& animal : animals) {
        animal->make_sound();
    }
    
}  // 所有动物对象自动释放
```

#### 7.2 容器操作

```cpp
void container_operations() {
    std::vector<std::unique_ptr<int>> vec;
    
    // 添加元素
    vec.push_back(std::make_unique<int>(1));
    vec.push_back(std::make_unique<int>(2));
    vec.emplace_back(std::make_unique<int>(3));
    
    // 访问元素
    std::cout << *vec[0] << std::endl;  // 1
    
    // ⚠️ 不能拷贝，只能移动
    // auto copy = vec;  // 编译错误
    auto moved = std::move(vec);  // ✓ 移动整个容器
    
    // 遍历
    for (const auto& ptr : moved) {
        std::cout << *ptr << " ";
    }
    std::cout << std::endl;
    
    // 移除元素（自动删除对象）
    moved.erase(moved.begin());
    
}  // 容器析构，所有元素自动释放
```

### 8. unique_ptr vs shared_ptr vs 原始指针

#### 8.1 性能对比

```cpp
#include <chrono>

void performance_comparison() {
    const int N = 1000000;
    
    // 原始指针
    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < N; ++i) {
        int* ptr = new int(i);
        delete ptr;
    }
    auto end = std::chrono::high_resolution_clock::now();
    auto raw_time = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    
    // unique_ptr
    start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < N; ++i) {
        auto ptr = std::make_unique<int>(i);
    }
    end = std::chrono::high_resolution_clock::now();
    auto unique_time = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    
    // shared_ptr
    start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < N; ++i) {
        auto ptr = std::make_shared<int>(i);
    }
    end = std::chrono::high_resolution_clock::now();
    auto shared_time = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    
    std::cout << "原始指针: " << raw_time << "ms" << std::endl;
    std::cout << "unique_ptr: " << unique_time << "ms" << std::endl;  // 与原始指针相近
    std::cout << "shared_ptr: " << shared_time << "ms" << std::endl;  // 稍慢（引用计数开销）
}
```

#### 8.2 选择指南

```cpp
// 场景1：独占所有权 → unique_ptr
std::unique_ptr<Resource> create_resource() {
    return std::make_unique<Resource>();
}

// 场景2：共享所有权 → shared_ptr
std::shared_ptr<Config> global_config = std::make_shared<Config>();

void use_config(std::shared_ptr<Config> config) {
    // 多个地方共享同一配置
}

// 场景3：不拥有所有权，只是观察 → 原始指针或引用
void observe(const Resource* res) {  // 或 const Resource&
    // 只读访问，不负责删除
}

// 场景4：可选参数 → 原始指针
void process(Resource* optional_res = nullptr) {
    if (optional_res) {
        // 使用资源
    }
}
```

### 9. 常见陷阱和最佳实践

#### 9.1 避免的错误用法

```cpp
void common_mistakes() {
    // ⚠️ 错误1：从原始指针创建多个unique_ptr
    int* raw = new int(42);
    std::unique_ptr<int> ptr1(raw);
    // std::unique_ptr<int> ptr2(raw);  // ⚠️ 双重删除！
    
    // ⚠️ 错误2：使用已移动的unique_ptr
    auto ptr3 = std::make_unique<int>(100);
    auto ptr4 = std::move(ptr3);
    // std::cout << *ptr3;  // ⚠️ 未定义行为！ptr3是nullptr
    
    // ✓ 正确：检查后使用
    if (ptr3) {
        std::cout << *ptr3;
    } else {
        std::cout << "ptr3是空的" << std::endl;
    }
    
    // ⚠️ 错误3：delete unique_ptr管理的指针
    auto ptr5 = std::make_unique<int>(200);
    // delete ptr5.get();  // ⚠️ 双重删除！
    
    // ⚠️ 错误4：返回局部对象的unique_ptr
    // 这个实际上是可以的，因为会被移动
    // 但如果是引用就错了
}

std::unique_ptr<int> correct_return() {
    auto local = std::make_unique<int>(42);
    return local;  // ✓ 自动移动，正确
}

// ⚠️ 错误：返回局部unique_ptr的引用
// std::unique_ptr<int>& wrong_return() {
//     auto local = std::make_unique<int>(42);
//     return local;  // ✗ 悬空引用！
// }
```

#### 9.2 最佳实践

```cpp
class BestPractices {
public:
    // ✓ 1. 优先使用 make_unique
    void practice1() {
        auto ptr = std::make_unique<Widget>(1, 2);
        // 而不是: std::unique_ptr<Widget>(new Widget(1, 2));
    }
    
    // ✓ 2. 使用 auto 简化类型
    void practice2() {
        auto ptr = std::make_unique<std::vector<std::string>>();
        // 而不是: std::unique_ptr<std::vector<std::string>> ptr = ...;
    }
    
    // ✓ 3. 函数参数传递指南
    
    // 3a. 转移所有权：按值传递
    void take_ownership(std::unique_ptr<Widget> widget) {
        // widget 的所有权转移到这里
    }
    
    // 3b. 只使用，不转移：传const引用或原始指针
    void use_only(const std::unique_ptr<Widget>& widget) {
        // 只使用，不转移所有权
    }
    
    void use_only2(Widget* widget) {
        // 只使用，不涉及所有权
        if (widget) {
            // ...
        }
    }
    
    // 3c. 可能修改指针本身：传非const引用
    void maybe_reset(std::unique_ptr<Widget>& widget) {
        if (should_reset) {
            widget.reset(new Widget(3, 4));
        }
    }
    
    // ✓ 4. 返回值：直接返回，自动移动
    std::unique_ptr<Widget> create_widget() {
        auto widget = std::make_unique<Widget>(1, 2);
        // 初始化操作
        return widget;  // 自动移动，无需std::move
    }
    
    // ✓ 5. 在容器中优先使用emplace
    void practice5() {
        std::vector<std::unique_ptr<int>> vec;
        
        // 推荐
        vec.emplace_back(std::make_unique<int>(42));
        
        // 也可以
        vec.push_back(std::make_unique<int>(42));
    }
    
private:
    bool should_reset = false;
};
```

### 10. 实战示例

#### 10.1 工厂模式

```cpp
class Shape {
public:
    virtual ~Shape() = default;
    virtual void draw() const = 0;
    virtual double area() const = 0;
};

class Circle : public Shape {
    double radius_;
public:
    Circle(double r) : radius_(r) {}
    void draw() const override {
        std::cout << "绘制圆形" << std::endl;
    }
    double area() const override {
        return 3.14159 * radius_ * radius_;
    }
};

class Rectangle : public Shape {
    double width_, height_;
public:
    Rectangle(double w, double h) : width_(w), height_(h) {}
    void draw() const override {
        std::cout << "绘制矩形" << std::endl;
    }
    double area() const override {
        return width_ * height_;
    }
};

// 工厂函数：返回unique_ptr
std::unique_ptr<Shape> create_shape(const std::string& type) {
    if (type == "circle") {
        return std::make_unique<Circle>(5.0);
    } else if (type == "rectangle") {
        return std::make_unique<Rectangle>(4.0, 6.0);
    }
    return nullptr;  // 未知类型
}

void factory_example() {
    auto shape1 = create_shape("circle");
    auto shape2 = create_shape("rectangle");
    
    if (shape1) {
        shape1->draw();
        std::cout << "面积: " << shape1->area() << std::endl;
    }
    
    if (shape2) {
        shape2->draw();
        std::cout << "面积: " << shape2->area() << std::endl;
    }
}
```

#### 10.2 资源管理链

```cpp
class FileLogger {
    std::unique_ptr<std::ofstream> file_;
    std::unique_ptr<std::mutex> mtx_;
    
public:
    FileLogger(const std::string& filename) 
        : file_(std::make_unique<std::ofstream>(filename))
        , mtx_(std::make_unique<std::mutex>()) {
        
        if (!file_->is_open()) {
            throw std::runtime_error("无法打开日志文件");
        }
    }
    
    void log(const std::string& message) {
        std::lock_guard<std::mutex> lock(*mtx_);
        *file_ << message << std::endl;
    }
    
    // Rule of Zero：编译器生成的移动操作足够
    // 自动释放所有资源
};

void logger_example() {
    auto logger = std::make_unique<FileLogger>("app.log");
    logger->log("应用启动");
    logger->log("执行操作");
    
}  // logger、file和mutex全部自动释放
```

### 11. 总结对比表

| 特性             | unique_ptr                 | shared_ptr             | 原始指针       |
| ---------------- | -------------------------- | ---------------------- | -------------- |
| **所有权**       | 独占                       | 共享（引用计数）       | 不管理         |
| **开销**         | 零开销（无自定义删除器）   | 引用计数开销           | 零开销         |
| **拷贝**         | 不可拷贝                   | 可拷贝                 | 可拷贝         |
| **移动**         | 可移动                     | 可移动                 | 不适用         |
| **自动释放**     | ✓ 是                       | ✓ 是（最后一个释放）   | ✗ 否           |
| **异常安全**     | ✓ 是                       | ✓ 是                   | ✗ 否           |
| **数组支持**     | ✓ 是（`unique_ptr<T[]>`）  | ✗ 否（需自定义删除器） | ✓ 是           |
| **自定义删除器** | ✓ 是（类型的一部分）       | ✓ 是（类型无关）       | ✗ 否           |
| **线程安全**     | 对象本身不是，但无竞争条件 | 引用计数是原子的       | 不适用         |
| **sizeof**       | 等于指针（无自定义删除器） | 2倍指针大小            | 1倍指针大小    |
| **适用场景**     | 独占资源、工厂模式、Pimpl  | 共享资源、缓存         | 观察、可选参数 |

### 12. 核心要点

**什么时候使用 unique_ptr？**
- ✓ 默认选择（优先于 shared_ptr 和原始指针）
- ✓ 对象只有一个明确的所有者
- ✓ 需要转移所有权
- ✓ 工厂函数返回值
- ✓ Pimpl 惯用法
- ✓ 容器中存储多态对象

**关键原则**：
1. **优先使用 make_unique**（异常安全、简洁）
2. **不可拷贝，只能移动**（独占所有权）
3. **零开销抽象**（性能等同于原始指针）
4. **遵循 RAII**（自动管理生命周期）
5. **默认选择**（只在需要共享时用 shared_ptr）

**记住**：`unique_ptr` 是现代C++中管理动态内存的首选方案，它提供了与原始指针相同的性能，却具有自动内存管理和异常安全的优势。


---

## 相关笔记
<!-- 自动生成 -->

- [删除器是什么](notes/C++/删除器是什么.md) - 相似度: 31% | 标签: C++, C++/删除器是什么.md

