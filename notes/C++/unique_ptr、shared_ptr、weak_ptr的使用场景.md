---
created: '2025-10-19'
last_reviewed: null
next_review: '2025-10-19'
review_count: 0
difficulty: medium
mastery_level: 0.0
tags:
- C++
- C++/unique_ptr、shared_ptr、weak_ptr的使用场景.md
related_outlines: []
---
# unique_ptr、shared_ptr、weak_ptr 的使用场景

## 面试标准答案（可背诵）

### 选择原则（优先级从高到低）

1. **默认使用 `unique_ptr`**：独占所有权，零开销，性能最优
2. **需要共享时用 `shared_ptr`**：多个所有者共享对象
3. **需要观察但不拥有时用 `weak_ptr`**：配合 `shared_ptr` 使用，解决循环引用

### 典型使用场景

**unique_ptr**：
- ✓ 工厂函数返回值（转移所有权）
- ✓ Pimpl 惯用法（隐藏实现细节）
- ✓ 容器中存储多态对象
- ✓ 管理独占资源（对象只有一个所有者）

**shared_ptr**：
- ✓ 多个对象共享同一资源（配置、缓存）
- ✓ 对象所有权不明确或动态变化
- ✓ 回调函数需要延长对象生命周期
- ✓ 跨线程共享对象

**weak_ptr**：
- ✓ 打破 `shared_ptr` 循环引用
- ✓ 缓存系统（缓存不应阻止对象删除）
- ✓ 观察者模式
- ✓ 父子双向引用关系

**核心理念**：能用 `unique_ptr` 就不用 `shared_ptr`，能用裸指针观察就不用 `weak_ptr`。

---

## 详细解析

### 1. unique_ptr 的使用场景

#### 1.1 工厂函数返回值

```cpp
#include <memory>
#include <string>

// 抽象基类
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
    void draw() const override { std::cout << "绘制圆形" << std::endl; }
    double area() const override { return 3.14159 * radius_ * radius_; }
};

class Rectangle : public Shape {
    double width_, height_;
public:
    Rectangle(double w, double h) : width_(w), height_(h) {}
    void draw() const override { std::cout << "绘制矩形" << std::endl; }
    double area() const override { return width_ * height_; }
};

// 工厂函数：返回 unique_ptr，转移所有权给调用者
std::unique_ptr<Shape> create_shape(const std::string& type, double param1, double param2 = 0) {
    if (type == "circle") {
        return std::make_unique<Circle>(param1);
    } else if (type == "rectangle") {
        return std::make_unique<Rectangle>(param1, param2);
    }
    return nullptr;  // 未知类型
}

void factory_pattern_example() {
    // 调用者获得对象的唯一所有权
    auto shape = create_shape("circle", 5.0);
    
    if (shape) {
        shape->draw();
        std::cout << "面积: " << shape->area() << std::endl;
    }
    
    // 可以转移所有权
    auto another_shape = std::move(shape);
    
}  // 自动释放
```

**为什么用 unique_ptr？**
- 工厂函数明确转移所有权给调用者
- 调用者完全控制对象生命周期
- 零开销，性能最优
- 返回值优化（RVO），无需显式 `std::move`

#### 1.2 Pimpl 惯用法（编译防火墙）

```cpp
// Widget.h - 头文件
class Widget {
public:
    Widget();
    ~Widget();  // 必须在 .cpp 中定义
    
    // 移动操作
    Widget(Widget&&) noexcept;
    Widget& operator=(Widget&&) noexcept;
    
    // 禁止拷贝
    Widget(const Widget&) = delete;
    Widget& operator=(const Widget&) = delete;
    
    void do_something();
    int get_value() const;
    
private:
    class Impl;  // 前向声明
    std::unique_ptr<Impl> pimpl_;  // 指向实现的指针
};

// Widget.cpp - 实现文件
class Widget::Impl {
public:
    void do_something() {
        std::cout << "实现细节: " << data_ << std::endl;
    }
    
    int get_value() const { return value_; }
    
private:
    // 实现细节，对用户隐藏
    std::vector<int> data_{1, 2, 3, 4, 5};
    std::string name_ = "Widget";
    int value_ = 42;
};

Widget::Widget() : pimpl_(std::make_unique<Impl>()) {}

Widget::~Widget() = default;  // 必须在 Impl 完整定义之后

Widget::Widget(Widget&&) noexcept = default;
Widget& Widget::operator=(Widget&&) noexcept = default;

void Widget::do_something() {
    pimpl_->do_something();
}

int Widget::get_value() const {
    return pimpl_->get_value();
}

// 使用
void pimpl_example() {
    Widget w;
    w.do_something();
    std::cout << "值: " << w.get_value() << std::endl;
}
```

**为什么用 unique_ptr？**
- Widget 独占 Impl，明确的所有权关系
- 实现细节完全隐藏，头文件不需要包含实现依赖
- 修改实现不需要重新编译使用 Widget 的代码
- 编译时间显著减少

#### 1.3 容器中存储多态对象

```cpp
void polymorphic_container() {
    // 存储不同类型的 Shape 对象
    std::vector<std::unique_ptr<Shape>> shapes;
    
    shapes.push_back(std::make_unique<Circle>(3.0));
    shapes.push_back(std::make_unique<Rectangle>(4.0, 5.0));
    shapes.push_back(std::make_unique<Circle>(2.0));
    
    // 多态调用
    double total_area = 0;
    for (const auto& shape : shapes) {
        shape->draw();
        total_area += shape->area();
    }
    
    std::cout << "总面积: " << total_area << std::endl;
    
    // 删除第一个元素
    shapes.erase(shapes.begin());
    
}  // 所有对象自动释放
```

**为什么用 unique_ptr？**
- 容器拥有对象，对象不需要共享
- 性能最优，没有引用计数开销
- 支持移动语义，可以高效地重新排列

#### 1.4 管理独占资源

```cpp
class FileLogger {
    std::unique_ptr<std::ofstream> file_;
    
public:
    FileLogger(const std::string& filename) 
        : file_(std::make_unique<std::ofstream>(filename)) {
        
        if (!file_->is_open()) {
            throw std::runtime_error("无法打开日志文件");
        }
    }
    
    void log(const std::string& message) {
        if (file_) {
            *file_ << message << std::endl;
        }
    }
    
    // 自动管理文件资源
};

void resource_ownership() {
    FileLogger logger("app.log");
    logger.log("应用启动");
    logger.log("执行操作");
    
}  // logger 销毁，文件自动关闭
```

**为什么用 unique_ptr？**
- 文件资源只有一个所有者
- 明确的资源所有权
- RAII 自动管理

### 2. shared_ptr 的使用场景

#### 2.1 共享配置或资源

```cpp
// 全局配置类
class Configuration {
public:
    Configuration(const std::string& config_file) {
        std::cout << "加载配置: " << config_file << std::endl;
        // 从文件加载配置
    }
    
    ~Configuration() {
        std::cout << "释放配置" << std::endl;
    }
    
    std::string get(const std::string& key) const {
        // 返回配置值
        return "value";
    }
};

// 多个对象共享同一配置
class ServiceA {
    std::shared_ptr<Configuration> config_;
public:
    ServiceA(std::shared_ptr<Configuration> config) 
        : config_(config) {}
    
    void process() {
        std::string value = config_->get("key");
        std::cout << "ServiceA 使用配置" << std::endl;
    }
};

class ServiceB {
    std::shared_ptr<Configuration> config_;
public:
    ServiceB(std::shared_ptr<Configuration> config) 
        : config_(config) {}
    
    void execute() {
        std::string value = config_->get("key");
        std::cout << "ServiceB 使用配置" << std::endl;
    }
};

void shared_config_example() {
    // 创建共享的配置对象
    auto config = std::make_shared<Configuration>("config.ini");
    
    ServiceA serviceA(config);
    ServiceB serviceB(config);
    
    serviceA.process();
    serviceB.execute();
    
    std::cout << "引用计数: " << config.use_count() << std::endl;  // 3
    
}  // 配置在所有服务销毁后才释放
```

**为什么用 shared_ptr？**
- 配置被多个服务共享
- 所有权不明确（任何服务都可能最后释放）
- 自动管理生命周期

#### 2.2 回调函数延长对象生命周期

```cpp
class AsyncTask {
    int id_;
public:
    AsyncTask(int id) : id_(id) {
        std::cout << "创建任务 " << id_ << std::endl;
    }
    
    ~AsyncTask() {
        std::cout << "销毁任务 " << id_ << std::endl;
    }
    
    void execute() {
        std::cout << "执行任务 " << id_ << std::endl;
    }
};

class TaskScheduler {
public:
    using Callback = std::function<void()>;
    
    void schedule(Callback callback, int delay_ms) {
        // 模拟异步调度
        std::thread([callback, delay_ms]() {
            std::this_thread::sleep_for(std::chrono::milliseconds(delay_ms));
            callback();
        }).detach();
    }
};

void async_callback_example() {
    TaskScheduler scheduler;
    
    {
        auto task = std::make_shared<AsyncTask>(1);
        
        // 回调捕获 shared_ptr，延长对象生命周期
        scheduler.schedule([task]() {
            task->execute();
        }, 100);
        
        // 即使 task 离开作用域，对象仍然存在
    }  // task 局部变量销毁，但对象未删除（回调还持有）
    
    std::this_thread::sleep_for(std::chrono::milliseconds(200));
    // 回调执行后，对象才真正释放
}
```

**为什么用 shared_ptr？**
- 回调可能在对象离开作用域后执行
- `shared_ptr` 确保对象在回调执行期间仍然有效
- 自动管理异步场景下的生命周期

#### 2.3 跨线程共享对象

```cpp
class ThreadSafeCounter {
    int count_ = 0;
    mutable std::mutex mtx_;
    
public:
    void increment() {
        std::lock_guard<std::mutex> lock(mtx_);
        ++count_;
    }
    
    int get() const {
        std::lock_guard<std::mutex> lock(mtx_);
        return count_;
    }
};

void multi_thread_example() {
    // 多个线程共享同一计数器
    auto counter = std::make_shared<ThreadSafeCounter>();
    
    std::vector<std::thread> threads;
    
    for (int i = 0; i < 10; ++i) {
        threads.emplace_back([counter]() {  // 拷贝 shared_ptr，增加引用计数
            for (int j = 0; j < 1000; ++j) {
                counter->increment();
            }
        });
    }
    
    for (auto& t : threads) {
        t.join();
    }
    
    std::cout << "最终计数: " << counter->get() << std::endl;  // 10000
    
}  // 所有线程结束后，counter 才释放
```

**为什么用 shared_ptr？**
- 对象被多个线程共享
- `shared_ptr` 引用计数是线程安全的
- 确保对象在所有线程使用完毕后才释放

#### 2.4 对象图/复杂所有权关系

```cpp
class Document {
public:
    std::string content;
    
    Document(const std::string& c) : content(c) {
        std::cout << "创建文档" << std::endl;
    }
    
    ~Document() {
        std::cout << "销毁文档" << std::endl;
    }
};

class Editor {
    std::shared_ptr<Document> doc_;
public:
    void open(std::shared_ptr<Document> doc) {
        doc_ = doc;
    }
    
    void edit() {
        if (doc_) {
            doc_->content += " [已编辑]";
        }
    }
};

class Viewer {
    std::shared_ptr<Document> doc_;
public:
    void display(std::shared_ptr<Document> doc) {
        doc_ = doc;
    }
    
    void show() {
        if (doc_) {
            std::cout << "内容: " << doc_->content << std::endl;
        }
    }
};

void complex_ownership() {
    auto doc = std::make_shared<Document>("原始内容");
    
    Editor editor;
    Viewer viewer;
    
    editor.open(doc);
    viewer.display(doc);
    
    editor.edit();
    viewer.show();  // 看到编辑后的内容
    
    std::cout << "引用计数: " << doc.use_count() << std::endl;  // 3
    
}  // 文档在所有引用者释放后才删除
```

**为什么用 shared_ptr？**
- 文档被编辑器和查看器共享
- 所有权关系复杂，难以确定谁应该最后释放
- `shared_ptr` 自动处理

### 3. weak_ptr 的使用场景

#### 3.1 打破循环引用

```cpp
// 双向链表节点
class ListNode {
public:
    int value;
    std::shared_ptr<ListNode> next;      // 强引用
    std::weak_ptr<ListNode> prev;        // 弱引用，打破循环
    
    ListNode(int v) : value(v) {
        std::cout << "创建节点 " << value << std::endl;
    }
    
    ~ListNode() {
        std::cout << "销毁节点 " << value << std::endl;
    }
};

void doubly_linked_list() {
    auto node1 = std::make_shared<ListNode>(1);
    auto node2 = std::make_shared<ListNode>(2);
    auto node3 = std::make_shared<ListNode>(3);
    
    // 建立双向链接
    node1->next = node2;
    node2->prev = node1;
    
    node2->next = node3;
    node3->prev = node2;
    
    // 访问前一个节点
    if (auto prev = node2->prev.lock()) {
        std::cout << "node2 的前一个节点: " << prev->value << std::endl;
    }
    
}  // 所有节点正确释放，无内存泄漏
```

**为什么用 weak_ptr？**
- 双向引用会导致循环引用
- 一个方向用 `shared_ptr`，另一个方向用 `weak_ptr`
- 打破循环，避免内存泄漏

#### 3.2 缓存系统

```cpp
class ExpensiveObject {
    std::string id_;
public:
    ExpensiveObject(const std::string& id) : id_(id) {
        std::cout << "创建昂贵对象: " << id_ << std::endl;
        // 模拟昂贵的初始化
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }
    
    ~ExpensiveObject() {
        std::cout << "销毁昂贵对象: " << id_ << std::endl;
    }
    
    void use() {
        std::cout << "使用对象: " << id_ << std::endl;
    }
};

// 使用 weak_ptr 的缓存
class ObjectCache {
    std::map<std::string, std::weak_ptr<ExpensiveObject>> cache_;
    std::mutex mtx_;
    
public:
    std::shared_ptr<ExpensiveObject> get(const std::string& id) {
        std::lock_guard<std::mutex> lock(mtx_);
        
        // 查找缓存
        auto it = cache_.find(id);
        if (it != cache_.end()) {
            // 尝试获取对象
            if (auto obj = it->second.lock()) {
                std::cout << "缓存命中: " << id << std::endl;
                return obj;
            } else {
                std::cout << "缓存过期: " << id << std::endl;
                cache_.erase(it);
            }
        }
        
        // 创建新对象
        std::cout << "缓存未命中，创建对象: " << id << std::endl;
        auto obj = std::make_shared<ExpensiveObject>(id);
        cache_[id] = obj;  // 存储 weak_ptr
        return obj;
    }
    
    void cleanup() {
        std::lock_guard<std::mutex> lock(mtx_);
        
        for (auto it = cache_.begin(); it != cache_.end(); ) {
            if (it->second.expired()) {
                std::cout << "清理过期缓存: " << it->first << std::endl;
                it = cache_.erase(it);
            } else {
                ++it;
            }
        }
    }
};

void cache_example() {
    ObjectCache cache;
    
    {
        auto obj1 = cache.get("object1");  // 创建
        auto obj2 = cache.get("object1");  // 缓存命中
        
        obj1->use();
        obj2->use();
        
        std::cout << "obj1 和 obj2 是同一对象: " 
                  << (obj1 == obj2 ? "是" : "否") << std::endl;
        
    }  // obj1 和 obj2 销毁，对象被删除
    
    auto obj3 = cache.get("object1");  // 缓存过期，重新创建
    
    cache.cleanup();  // 清理过期缓存
}
```

**为什么用 weak_ptr？**
- 缓存不应该阻止对象被删除
- 当所有用户不再使用对象时，对象应该被释放
- `weak_ptr` 允许缓存"观察"对象但不拥有它

#### 3.3 观察者模式

```cpp
class Subject;

class Observer {
public:
    virtual ~Observer() = default;
    virtual void update(const std::string& message) = 0;
};

class ConcreteObserver : public Observer {
    std::string name_;
public:
    ConcreteObserver(const std::string& name) : name_(name) {
        std::cout << "创建观察者: " << name_ << std::endl;
    }
    
    ~ConcreteObserver() {
        std::cout << "销毁观察者: " << name_ << std::endl;
    }
    
    void update(const std::string& message) override {
        std::cout << name_ << " 收到消息: " << message << std::endl;
    }
};

class Subject {
    std::vector<std::weak_ptr<Observer>> observers_;
    
public:
    void attach(std::shared_ptr<Observer> observer) {
        observers_.push_back(observer);  // 存储 weak_ptr
        std::cout << "注册观察者" << std::endl;
    }
    
    void notify(const std::string& message) {
        std::cout << "通知观察者..." << std::endl;
        
        // 清理失效观察者，通知有效观察者
        auto it = observers_.begin();
        while (it != observers_.end()) {
            if (auto observer = it->lock()) {
                observer->update(message);
                ++it;
            } else {
                std::cout << "移除失效观察者" << std::endl;
                it = observers_.erase(it);
            }
        }
    }
};

void observer_pattern() {
    Subject subject;
    
    {
        auto observer1 = std::make_shared<ConcreteObserver>("观察者1");
        auto observer2 = std::make_shared<ConcreteObserver>("观察者2");
        
        subject.attach(observer1);
        subject.attach(observer2);
        
        subject.notify("消息1");  // 两个观察者收到
        
    }  // 观察者销毁
    
    subject.notify("消息2");  // 自动清理失效观察者，无人收到
}
```

**为什么用 weak_ptr？**
- Subject 不应该控制 Observer 的生命周期
- Observer 可以随时删除
- Subject 自动感知 Observer 失效

#### 3.4 父子双向引用

```cpp
class Child;

class Parent {
public:
    std::string name;
    std::vector<std::shared_ptr<Child>> children;  // 父 -> 子：强引用
    
    Parent(const std::string& n) : name(n) {
        std::cout << "创建父节点: " << name << std::endl;
    }
    
    ~Parent() {
        std::cout << "销毁父节点: " << name << std::endl;
    }
    
    void add_child(std::shared_ptr<Child> child);
};

class Child {
public:
    std::string name;
    std::weak_ptr<Parent> parent;  // 子 -> 父：弱引用
    
    Child(const std::string& n) : name(n) {
        std::cout << "创建子节点: " << name << std::endl;
    }
    
    ~Child() {
        std::cout << "销毁子节点: " << name << std::endl;
    }
    
    void print_parent() {
        if (auto p = parent.lock()) {
            std::cout << name << " 的父节点: " << p->name << std::endl;
        } else {
            std::cout << name << " 没有父节点" << std::endl;
        }
    }
};

void Parent::add_child(std::shared_ptr<Child> child) {
    children.push_back(child);
    child->parent = shared_from_this();  // 需要继承 enable_shared_from_this
}

// 修正版 Parent（继承 enable_shared_from_this）
class GoodParent : public std::enable_shared_from_this<GoodParent> {
public:
    std::string name;
    std::vector<std::shared_ptr<Child>> children;
    
    GoodParent(const std::string& n) : name(n) {
        std::cout << "创建父节点: " << name << std::endl;
    }
    
    ~GoodParent() {
        std::cout << "销毁父节点: " << name << std::endl;
    }
    
    void add_child(std::shared_ptr<Child> child) {
        children.push_back(child);
        child->parent = shared_from_this();
    }
};

void parent_child_example() {
    auto parent = std::make_shared<GoodParent>("父节点");
    auto child1 = std::make_shared<Child>("子节点1");
    auto child2 = std::make_shared<Child>("子节点2");
    
    parent->add_child(child1);
    parent->add_child(child2);
    
    child1->print_parent();
    child2->print_parent();
    
}  // 正确释放：先子节点，再父节点
```

**为什么用 weak_ptr？**
- 父子双向引用会导致循环引用
- 父拥有子（`shared_ptr`），子观察父（`weak_ptr`）
- 打破循环，正确释放

### 4. 选择决策树

```cpp
/*
选择智能指针的决策流程：

开始
  │
  ├─ 需要管理动态对象？
  │   ├─ 否 → 使用栈对象或引用
  │   └─ 是 ↓
  │
  ├─ 对象是否需要共享？
  │   ├─ 否 → 使用 unique_ptr ✓
  │   └─ 是 ↓
  │
  ├─ 是否只是观察，不拥有？
  │   ├─ 是 → 使用 weak_ptr（配合 shared_ptr）✓
  │   └─ 否 ↓
  │
  ├─ 多个所有者共享对象？
  │   └─ 是 → 使用 shared_ptr ✓
  │
  └─ 特殊情况：
      ├─ 工厂函数返回 → unique_ptr
      ├─ 回调需要延长生命周期 → shared_ptr
      ├─ 缓存系统 → weak_ptr
      ├─ 双向引用 → shared_ptr + weak_ptr
      └─ 跨线程共享 → shared_ptr
*/
```

### 5. 实战对比示例

#### 5.1 同一场景的不同选择

```cpp
// 场景：文档管理系统

// 方案1：使用 unique_ptr（单一编辑器）
class DocumentEditor_Unique {
    std::unique_ptr<Document> current_doc_;
    
public:
    void open(std::unique_ptr<Document> doc) {
        current_doc_ = std::move(doc);  // 转移所有权
    }
    
    void close() {
        current_doc_.reset();  // 释放文档
    }
};

// 方案2：使用 shared_ptr（多个编辑器）
class DocumentEditor_Shared {
    std::shared_ptr<Document> current_doc_;
    
public:
    void open(std::shared_ptr<Document> doc) {
        current_doc_ = doc;  // 共享所有权
    }
    
    void close() {
        current_doc_.reset();  // 减少引用计数
    }
};

// 方案3：使用 weak_ptr（观察者）
class DocumentViewer {
    std::weak_ptr<Document> current_doc_;
    
public:
    void observe(std::shared_ptr<Document> doc) {
        current_doc_ = doc;  // 观察，不拥有
    }
    
    void refresh() {
        if (auto doc = current_doc_.lock()) {
            std::cout << "刷新视图: " << doc->content << std::endl;
        } else {
            std::cout << "文档已关闭" << std::endl;
        }
    }
};

void document_system_example() {
    // 场景1：单一编辑器
    {
        DocumentEditor_Unique editor;
        editor.open(std::make_unique<Document>("内容1"));
        editor.close();
    }
    
    // 场景2：多个编辑器共享文档
    {
        auto doc = std::make_shared<Document>("内容2");
        
        DocumentEditor_Shared editor1, editor2;
        editor1.open(doc);
        editor2.open(doc);  // 两个编辑器共享同一文档
        
        // 两个编辑器都可以编辑
        doc->content += " [修改]";
    }
    
    // 场景3：编辑器 + 观察者
    {
        auto doc = std::make_shared<Document>("内容3");
        
        DocumentEditor_Shared editor;
        DocumentViewer viewer;
        
        editor.open(doc);
        viewer.observe(doc);  // 观察者不拥有文档
        
        viewer.refresh();  // 可以查看
        
        editor.close();
        doc.reset();  // 编辑器关闭文档
        
        viewer.refresh();  // 检测到文档已关闭
    }
}
```

#### 5.2 错误使用示例

```cpp
// ✗ 错误1：应该用 unique_ptr 却用 shared_ptr
void unnecessary_shared() {
    // 对象从不共享，引用计数开销浪费
    std::shared_ptr<int> ptr = std::make_shared<int>(42);
    // ✓ 应该用 unique_ptr
    // auto ptr = std::make_unique<int>(42);
}

// ✗ 错误2：应该用 weak_ptr 却用 shared_ptr（循环引用）
struct BadNode {
    std::shared_ptr<BadNode> next;
    std::shared_ptr<BadNode> prev;  // ⚠️ 内存泄漏！
};

// ✗ 错误3：应该用 shared_ptr 却用 unique_ptr
void premature_unique() {
    auto doc = std::make_unique<Document>("内容");
    
    // 需要传给多个对象，但 unique_ptr 只能移动
    // editor1.open(std::move(doc));  // doc 变为 nullptr
    // editor2.open(std::move(doc));  // 错误！doc 已经是 nullptr
    
    // ✓ 应该用 shared_ptr
}

// ✗ 错误4：过度使用裸指针
void dangerous_raw_pointer() {
    auto doc = std::make_shared<Document>("内容");
    Document* raw = doc.get();
    
    doc.reset();  // 对象被删除
    
    // raw->content;  // ⚠️ 悬空指针！
    
    // ✓ 应该用 weak_ptr
}
```

### 6. 性能对比

```cpp
#include <chrono>

void performance_benchmark() {
    const int N = 1000000;
    
    // 1. unique_ptr 性能
    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < N; ++i) {
        auto ptr = std::make_unique<int>(i);
    }
    auto end = std::chrono::high_resolution_clock::now();
    auto unique_time = std::chrono::duration_cast<std::chrono::milliseconds>(
        end - start
    ).count();
    
    // 2. shared_ptr 性能（创建）
    start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < N; ++i) {
        auto ptr = std::make_shared<int>(i);
    }
    end = std::chrono::high_resolution_clock::now();
    auto shared_create_time = std::chrono::duration_cast<std::chrono::milliseconds>(
        end - start
    ).count();
    
    // 3. shared_ptr 性能（拷贝，引用计数操作）
    auto shared = std::make_shared<int>(42);
    start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < N; ++i) {
        auto copy = shared;  // 原子递增
    }  // 原子递减
    end = std::chrono::high_resolution_clock::now();
    auto shared_copy_time = std::chrono::duration_cast<std::chrono::milliseconds>(
        end - start
    ).count();
    
    std::cout << "性能对比（" << N << " 次操作）：" << std::endl;
    std::cout << "  unique_ptr 创建: " << unique_time << "ms" << std::endl;
    std::cout << "  shared_ptr 创建: " << shared_create_time << "ms" << std::endl;
    std::cout << "  shared_ptr 拷贝: " << shared_copy_time << "ms" << std::endl;
    
    // 典型结果（相对值）：
    // unique_ptr: 1.0x (最快)
    // shared_ptr 创建: 1.2-1.5x
    // shared_ptr 拷贝: 2-3x (原子操作开销)
}
```

### 7. 综合对比表

| 维度           | unique_ptr                | shared_ptr             | weak_ptr                   |
| -------------- | ------------------------- | ---------------------- | -------------------------- |
| **所有权**     | 独占                      | 共享（引用计数）       | 不拥有（观察）             |
| **性能**       | 最优（零开销）            | 引用计数开销           | 与 shared_ptr 类似         |
| **内存占用**   | 8 字节                    | 16 字节 + 控制块       | 16 字节                    |
| **拷贝**       | 不可拷贝                  | 可拷贝（增加计数）     | 可拷贝（不影响强引用）     |
| **移动**       | 可移动                    | 可移动                 | 可移动                     |
| **线程安全**   | 对象本身不是              | 引用计数是原子的       | 引用计数是原子的           |
| **典型场景**   | 工厂函数、Pimpl、独占资源 | 共享资源、回调、跨线程 | 缓存、观察者、打破循环引用 |
| **使用复杂度** | 简单                      | 中等                   | 复杂（需要 lock()）        |
| **默认选择**   | ✓ 是（优先）              | 需要共享时             | 需要观察时                 |

### 8. 核心建议

#### 8.1 选择原则

```cpp
// 决策流程（按优先级）：

// 1. 能不用智能指针就不用（优先栈对象）
void use_stack() {
    Widget widget;  // ✓ 最佳
}

// 2. 需要堆对象时，默认用 unique_ptr
std::unique_ptr<Widget> create() {
    return std::make_unique<Widget>();  // ✓ 默认选择
}

// 3. 需要共享时才用 shared_ptr
std::shared_ptr<Config> global_config;  // ✓ 确实需要共享

// 4. 需要观察但不拥有时用 weak_ptr
std::weak_ptr<Widget> observer;  // ✓ 配合 shared_ptr

// 5. 只是传递/观察时用裸指针或引用
void process(Widget* widget) {  // ✓ 不涉及所有权
    if (widget) {
        widget->use();
    }
}
```

#### 8.2 最佳实践

```cpp
// ✓ 好的做法
class GoodClass {
    // 1. 成员变量：优先 unique_ptr
    std::unique_ptr<Impl> pimpl_;
    
    // 2. 需要共享时用 shared_ptr
    std::shared_ptr<Config> config_;
    
    // 3. 观察者用 weak_ptr
    std::vector<std::weak_ptr<Observer>> observers_;
    
public:
    // 4. 工厂返回 unique_ptr
    static std::unique_ptr<GoodClass> create() {
        return std::make_unique<GoodClass>();
    }
    
    // 5. 参数传递：
    // - 转移所有权：值传递 unique_ptr
    void take(std::unique_ptr<Widget> widget);
    
    // - 共享所有权：值传递 shared_ptr
    void share(std::shared_ptr<Config> config);
    
    // - 只使用：const引用或裸指针
    void use(const Widget& widget);
    void process(Widget* widget);
};
```

### 9. 总结

**核心原则**：
1. **默认使用 `unique_ptr`**：性能最优，语义清晰
2. **确需共享才用 `shared_ptr`**：有引用计数开销
3. **配合使用 `weak_ptr`**：解决循环引用、实现观察者
4. **优先栈对象**：不需要智能指针时不用

**记住**：
- `unique_ptr`：我独占这个对象
- `shared_ptr`：我们共享这个对象
- `weak_ptr`：我只是看看，不拥有

正确选择智能指针，既能保证内存安全，又能获得最佳性能。过度使用 `shared_ptr` 和滥用裸指针都是应该避免的。


---

## 相关笔记
<!-- 自动生成 -->

- [shared_ptr和weak_ptr](notes/C++/shared_ptr和weak_ptr.md) - 相似度: 39% | 标签: C++, C++/shared_ptr和weak_ptr.md

