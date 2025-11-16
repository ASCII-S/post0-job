---
created: '2025-10-19'
last_reviewed: null
next_review: '2025-10-19'
review_count: 0
difficulty: medium
mastery_level: 0.0
tags:
- C++
- C++/shared_ptr和weak_ptr.md
related_outlines: []
---
# std::shared_ptr 和 std::weak_ptr

## 面试标准答案（可背诵）

### shared_ptr

**`std::shared_ptr` 是共享所有权的智能指针，使用引用计数管理对象生命周期。**

**核心特性**：
1. **共享所有权**：多个 `shared_ptr` 可以指向同一对象，对象在最后一个 `shared_ptr` 销毁时释放
2. **引用计数**：内部维护强引用计数和弱引用计数，线程安全的原子操作
3. **可拷贝可移动**：拷贝增加引用计数，移动不增加计数
4. **控制块**：额外的内存开销，存储引用计数和删除器

**使用场景**：共享资源、观察者模式、缓存、对象池。

### weak_ptr

**`std::weak_ptr` 是不拥有对象的弱引用，配合 `shared_ptr` 使用，解决循环引用问题。**

**核心特性**：
1. **不影响生命周期**：不增加强引用计数，不影响对象销毁
2. **观察者角色**：可以检查对象是否还存在，使用时需要转换为 `shared_ptr`
3. **打破循环引用**：在双向引用中使用 `weak_ptr` 避免内存泄漏

**使用场景**：缓存、观察者模式、父子关系、避免循环引用。

**关键区别**：`shared_ptr` 拥有对象，`weak_ptr` 只是观察对象，不拥有。

---

## 详细解析

### 1. shared_ptr 基本概念

#### 1.1 引用计数机制

```cpp
#include <memory>
#include <iostream>

void reference_counting_demo() {
    // 创建shared_ptr，引用计数 = 1
    std::shared_ptr<int> ptr1 = std::make_shared<int>(42);
    std::cout << "引用计数: " << ptr1.use_count() << std::endl;  // 1
    
    {
        // 拷贝，引用计数 = 2
        std::shared_ptr<int> ptr2 = ptr1;
        std::cout << "引用计数: " << ptr1.use_count() << std::endl;  // 2
        std::cout << "引用计数: " << ptr2.use_count() << std::endl;  // 2
        
        {
            // 再次拷贝，引用计数 = 3
            std::shared_ptr<int> ptr3 = ptr2;
            std::cout << "引用计数: " << ptr1.use_count() << std::endl;  // 3
        }  // ptr3销毁，引用计数 = 2
        
        std::cout << "引用计数: " << ptr1.use_count() << std::endl;  // 2
    }  // ptr2销毁，引用计数 = 1
    
    std::cout << "引用计数: " << ptr1.use_count() << std::endl;  // 1
}  // ptr1销毁，引用计数 = 0，对象被删除
```

#### 1.2 创建 shared_ptr

```cpp
class Widget {
public:
    Widget(int x) : value(x) {
        std::cout << "Widget(" << value << ")" << std::endl;
    }
    ~Widget() {
        std::cout << "~Widget(" << value << ")" << std::endl;
    }
    int value;
};

void creation_methods() {
    // ✓ 方式1：make_shared（推荐）
    auto ptr1 = std::make_shared<Widget>(10);
    
    // 方式2：直接构造
    std::shared_ptr<Widget> ptr2(new Widget(20));
    
    // 方式3：从unique_ptr移动
    auto uptr = std::make_unique<Widget>(30);
    std::shared_ptr<Widget> ptr3 = std::move(uptr);
    
    // 方式4：初始化为nullptr
    std::shared_ptr<Widget> ptr4;  // nullptr
    std::shared_ptr<Widget> ptr5 = nullptr;
    
    std::cout << "引用计数 ptr1: " << ptr1.use_count() << std::endl;  // 1
    std::cout << "引用计数 ptr2: " << ptr2.use_count() << std::endl;  // 1
    std::cout << "引用计数 ptr3: " << ptr3.use_count() << std::endl;  // 1
}
```

#### 1.3 为什么优先使用 make_shared

```cpp
void why_make_shared() {
    // ✓ 推荐：make_shared
    auto ptr1 = std::make_shared<Widget>(100);
    
    // ✗ 不推荐：直接new
    std::shared_ptr<Widget> ptr2(new Widget(200));
}

// 原因1：性能 - 一次内存分配 vs 两次
// make_shared: 对象和控制块在一起分配
// new Widget:  对象一次分配，控制块另一次分配

// 原因2：异常安全
void process(std::shared_ptr<Widget> w1, std::shared_ptr<Widget> w2);

void exception_safety_issue() {
    // 潜在问题（C++17前）
    process(
        std::shared_ptr<Widget>(new Widget(1)),  // 可能泄漏
        std::shared_ptr<Widget>(new Widget(2))
    );
    // 执行顺序可能是：new Widget(1) -> new Widget(2) -> shared_ptr构造
    // 如果第二个new抛异常，第一个对象泄漏
    
    // ✓ 安全
    process(
        std::make_shared<Widget>(1),
        std::make_shared<Widget>(2)
    );
}

// 原因3：代码简洁
auto ptr3 = std::make_shared<std::vector<std::string>>(10, "hello");
// vs
std::shared_ptr<std::vector<std::string>> ptr4(
    new std::vector<std::string>(10, "hello")
);
```

### 2. shared_ptr 的使用

#### 2.1 拷贝和移动

```cpp
void copy_and_move() {
    auto ptr1 = std::make_shared<int>(42);
    std::cout << "ptr1计数: " << ptr1.use_count() << std::endl;  // 1
    
    // 拷贝：增加引用计数
    auto ptr2 = ptr1;
    std::cout << "ptr1计数: " << ptr1.use_count() << std::endl;  // 2
    std::cout << "ptr2计数: " << ptr2.use_count() << std::endl;  // 2
    
    // 移动：不增加引用计数
    auto ptr3 = std::move(ptr1);
    std::cout << "ptr1计数: " << (ptr1 ? ptr1.use_count() : 0) << std::endl;  // 0
    std::cout << "ptr2计数: " << ptr2.use_count() << std::endl;  // 2
    std::cout << "ptr3计数: " << ptr3.use_count() << std::endl;  // 2
    
    // ptr1现在是nullptr
    if (!ptr1) {
        std::cout << "ptr1已被移动" << std::endl;
    }
}
```

#### 2.2 作为函数参数

```cpp
class Data {
public:
    Data(int v) : value(v) {}
    int value;
};

// 方式1：按值传递（拷贝，增加引用计数）
void take_copy(std::shared_ptr<Data> ptr) {
    std::cout << "函数内计数: " << ptr.use_count() << std::endl;
    // 使用ptr
}

// 方式2：按const引用传递（不增加引用计数）
void use_data(const std::shared_ptr<Data>& ptr) {
    std::cout << "函数内计数: " << ptr.use_count() << std::endl;
    if (ptr) {
        std::cout << "值: " << ptr->value << std::endl;
    }
}

// 方式3：传递原始指针（不涉及所有权）
void process_data(Data* data) {
    if (data) {
        std::cout << "值: " << data->value << std::endl;
    }
}

void parameter_passing() {
    auto ptr = std::make_shared<Data>(100);
    std::cout << "原始计数: " << ptr.use_count() << std::endl;  // 1
    
    take_copy(ptr);  // 临时增加计数
    std::cout << "调用后计数: " << ptr.use_count() << std::endl;  // 1
    
    use_data(ptr);   // 不增加计数
    process_data(ptr.get());  // 不增加计数
}
```

#### 2.3 作为返回值

```cpp
// 返回shared_ptr：共享所有权
std::shared_ptr<Data> create_shared_data(int value) {
    auto data = std::make_shared<Data>(value);
    // 初始化操作
    return data;  // 可以自动移动或拷贝
}

// 工厂模式
std::shared_ptr<Data> factory(const std::string& type) {
    if (type == "small") {
        return std::make_shared<Data>(10);
    } else if (type == "large") {
        return std::make_shared<Data>(1000);
    }
    return nullptr;
}

void return_value_demo() {
    auto data1 = create_shared_data(50);
    std::cout << "data1计数: " << data1.use_count() << std::endl;  // 1
    
    auto data2 = data1;  // 共享
    std::cout << "data1计数: " << data1.use_count() << std::endl;  // 2
    std::cout << "data2计数: " << data2.use_count() << std::endl;  // 2
}
```

### 3. shared_ptr 的内部机制

#### 3.1 控制块（Control Block）

```cpp
/*
shared_ptr的内存布局：

对象内存（堆上）：
┌─────────────┐
│   Object    │  <- 实际对象
└─────────────┘

控制块（堆上）：
┌─────────────────┐
│ 强引用计数      │
├─────────────────┤
│ 弱引用计数      │
├─────────────────┤
│ 删除器          │
├─────────────────┤
│ 分配器          │
└─────────────────┘

shared_ptr对象（栈上或成员）：
┌─────────────────┐
│ 对象指针        │ -> 指向Object
├─────────────────┤
│ 控制块指针      │ -> 指向控制块
└─────────────────┘

make_shared优化（一次分配）：
┌─────────────────────────────┐
│   Object   │   控制块       │
└─────────────────────────────┘
*/

void control_block_demo() {
    // make_shared: 对象和控制块在一起
    auto ptr1 = std::make_shared<int>(42);
    std::cout << "sizeof(shared_ptr): " << sizeof(ptr1) << std::endl;  // 16字节(64位系统)
    
    // new: 对象和控制块分开
    std::shared_ptr<int> ptr2(new int(42));
    
    // 两个指针：8字节（对象指针）+ 8字节（控制块指针）= 16字节
}
```

#### 3.2 线程安全性

```cpp
#include <thread>
#include <vector>

void thread_safety_demo() {
    auto ptr = std::make_shared<int>(0);
    
    // ✓ 安全：引用计数的增减是线程安全的
    std::vector<std::thread> threads;
    for (int i = 0; i < 10; ++i) {
        threads.emplace_back([ptr]() {  // 拷贝shared_ptr，增加计数
            // 每个线程有自己的shared_ptr副本
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
        });
    }
    
    for (auto& t : threads) {
        t.join();
    }
    
    // ⚠️ 注意：对象本身的访问不是线程安全的
    auto shared_value = std::make_shared<int>(0);
    
    std::vector<std::thread> threads2;
    for (int i = 0; i < 10; ++i) {
        threads2.emplace_back([shared_value]() {
            for (int j = 0; j < 1000; ++j) {
                (*shared_value)++;  // ⚠️ 竞态条件！需要额外同步
            }
        });
    }
    
    for (auto& t : threads2) {
        t.join();
    }
    
    std::cout << "最终值: " << *shared_value << std::endl;  // 可能不是10000
}
```

### 4. 循环引用问题

#### 4.1 循环引用导致内存泄漏

```cpp
class Node {
public:
    std::shared_ptr<Node> next;
    std::shared_ptr<Node> prev;  // ⚠️ 循环引用！
    
    int value;
    
    Node(int v) : value(v) {
        std::cout << "Node(" << value << ")" << std::endl;
    }
    
    ~Node() {
        std::cout << "~Node(" << value << ")" << std::endl;
    }
};

void circular_reference_problem() {
    auto node1 = std::make_shared<Node>(1);
    auto node2 = std::make_shared<Node>(2);
    
    node1->next = node2;  // node1 -> node2, node2计数=2
    node2->prev = node1;  // node2 -> node1, node1计数=2
    
    std::cout << "node1计数: " << node1.use_count() << std::endl;  // 2
    std::cout << "node2计数: " << node2.use_count() << std::endl;  // 2
    
}  // node1和node2离开作用域
   // node1计数: 2->1 (仍被node2->prev引用)
   // node2计数: 2->1 (仍被node1->next引用)
   // ⚠️ 两个对象都不会被删除！内存泄漏！

/*
循环引用示意图：
node1 ──────> node2
  ^             │
  │             │
  └─────────────┘

引用计数都是2，离开作用域后都变成1，永远不会归零
*/
```

#### 4.2 父子关系中的循环引用

```cpp
class Child;

class Parent {
public:
    std::string name;
    std::vector<std::shared_ptr<Child>> children;
    
    Parent(const std::string& n) : name(n) {
        std::cout << "Parent(" << name << ")" << std::endl;
    }
    
    ~Parent() {
        std::cout << "~Parent(" << name << ")" << std::endl;
    }
};

class Child {
public:
    std::string name;
    std::shared_ptr<Parent> parent;  // ⚠️ 循环引用！
    
    Child(const std::string& n) : name(n) {
        std::cout << "Child(" << name << ")" << std::endl;
    }
    
    ~Child() {
        std::cout << "~Child(" << name << ")" << std::endl;
    }
};

void parent_child_leak() {
    auto parent = std::make_shared<Parent>("父节点");
    auto child = std::make_shared<Child>("子节点");
    
    parent->children.push_back(child);  // parent -> child
    child->parent = parent;              // child -> parent
    
    // ⚠️ 离开作用域后，parent和child都不会被删除！
}
```

### 5. weak_ptr 解决循环引用

#### 5.1 weak_ptr 基本概念

```cpp
void weak_ptr_basics() {
    std::weak_ptr<int> weak;
    
    {
        auto shared = std::make_shared<int>(42);
        weak = shared;  // weak_ptr指向shared_ptr管理的对象
        
        std::cout << "shared计数: " << shared.use_count() << std::endl;  // 1
        std::cout << "weak计数: " << weak.use_count() << std::endl;      // 1
        // weak_ptr不增加强引用计数！
        
        // 使用weak_ptr：需要先转换为shared_ptr
        if (auto temp_shared = weak.lock()) {  // lock()返回shared_ptr
            std::cout << "值: " << *temp_shared << std::endl;  // 42
            std::cout << "shared计数: " << shared.use_count() << std::endl;  // 2
        }
        
    }  // shared销毁，对象被删除
    
    // 对象已被删除，weak_ptr过期
    if (weak.expired()) {
        std::cout << "对象已被删除" << std::endl;
    }
    
    auto locked = weak.lock();
    if (!locked) {
        std::cout << "lock()返回空指针" << std::endl;
    }
}
```

#### 5.2 使用 weak_ptr 打破循环引用

```cpp
class GoodNode {
public:
    std::shared_ptr<GoodNode> next;
    std::weak_ptr<GoodNode> prev;  // ✓ 使用weak_ptr
    
    int value;
    
    GoodNode(int v) : value(v) {
        std::cout << "GoodNode(" << value << ")" << std::endl;
    }
    
    ~GoodNode() {
        std::cout << "~GoodNode(" << value << ")" << std::endl;
    }
};

void no_circular_reference() {
    auto node1 = std::make_shared<GoodNode>(1);
    auto node2 = std::make_shared<GoodNode>(2);
    
    node1->next = node2;  // node1 -> node2, node2计数=2
    node2->prev = node1;  // weak_ptr，node1计数仍为1
    
    std::cout << "node1计数: " << node1.use_count() << std::endl;  // 1
    std::cout << "node2计数: " << node2.use_count() << std::endl;  // 2
    
    // 访问prev
    if (auto prev_node = node2->prev.lock()) {
        std::cout << "prev值: " << prev_node->value << std::endl;  // 1
    }
    
}  // node1和node2正常销毁
   // ~GoodNode(2)
   // ~GoodNode(1)

/*
正确的引用关系：
node1 ────强引用───> node2
  ^                   │
  │                   │
  └────弱引用─────────┘

离开作用域：
1. node1计数: 1->0, 删除node1
2. node2计数: 2->1->0, 删除node2
*/
```

#### 5.3 修复父子关系

```cpp
class GoodChild;

class GoodParent {
public:
    std::string name;
    std::vector<std::shared_ptr<GoodChild>> children;  // 父->子：强引用
    
    GoodParent(const std::string& n) : name(n) {
        std::cout << "GoodParent(" << name << ")" << std::endl;
    }
    
    ~GoodParent() {
        std::cout << "~GoodParent(" << name << ")" << std::endl;
    }
};

class GoodChild {
public:
    std::string name;
    std::weak_ptr<GoodParent> parent;  // ✓ 子->父：弱引用
    
    GoodChild(const std::string& n) : name(n) {
        std::cout << "GoodChild(" << name << ")" << std::endl;
    }
    
    ~GoodChild() {
        std::cout << "~GoodChild(" << name << ")" << std::endl;
    }
    
    void print_parent() {
        if (auto p = parent.lock()) {
            std::cout << "父节点: " << p->name << std::endl;
        } else {
            std::cout << "父节点已不存在" << std::endl;
        }
    }
};

void good_parent_child() {
    auto parent = std::make_shared<GoodParent>("父节点");
    auto child = std::make_shared<GoodChild>("子节点");
    
    parent->children.push_back(child);  // 强引用
    child->parent = parent;              // 弱引用
    
    child->print_parent();  // 输出: 父节点: 父节点
    
}  // 正确释放
   // ~GoodChild(子节点)
   // ~GoodParent(父节点)
```

### 6. weak_ptr 的使用场景

#### 6.1 缓存系统

```cpp
#include <map>
#include <string>

class ExpensiveObject {
public:
    ExpensiveObject(const std::string& id) : id_(id) {
        std::cout << "创建昂贵对象: " << id_ << std::endl;
    }
    
    ~ExpensiveObject() {
        std::cout << "销毁昂贵对象: " << id_ << std::endl;
    }
    
    std::string id_;
};

class Cache {
    std::map<std::string, std::weak_ptr<ExpensiveObject>> cache_;
    
public:
    std::shared_ptr<ExpensiveObject> get(const std::string& id) {
        // 尝试从缓存获取
        auto it = cache_.find(id);
        if (it != cache_.end()) {
            if (auto obj = it->second.lock()) {
                std::cout << "缓存命中: " << id << std::endl;
                return obj;  // 缓存中有效
            } else {
                std::cout << "缓存过期: " << id << std::endl;
                cache_.erase(it);  // 对象已删除，清理缓存
            }
        }
        
        // 创建新对象
        std::cout << "缓存未命中，创建对象: " << id << std::endl;
        auto obj = std::make_shared<ExpensiveObject>(id);
        cache_[id] = obj;  // 存储weak_ptr
        return obj;
    }
};

void cache_example() {
    Cache cache;
    
    {
        auto obj1 = cache.get("object1");  // 创建
        auto obj2 = cache.get("object1");  // 缓存命中
        
        std::cout << "obj1和obj2是同一对象: " 
                  << (obj1 == obj2 ? "是" : "否") << std::endl;
    }  // obj1和obj2销毁，对象被删除
    
    auto obj3 = cache.get("object1");  // 缓存过期，重新创建
}
```

#### 6.2 观察者模式

```cpp
class Observer {
public:
    virtual ~Observer() = default;
    virtual void update(const std::string& message) = 0;
};

class ConcreteObserver : public Observer {
    std::string name_;
public:
    ConcreteObserver(const std::string& name) : name_(name) {}
    
    void update(const std::string& message) override {
        std::cout << name_ << " 收到消息: " << message << std::endl;
    }
};

class Subject {
    std::vector<std::weak_ptr<Observer>> observers_;
    
public:
    void attach(std::shared_ptr<Observer> observer) {
        observers_.push_back(observer);  // 存储weak_ptr
    }
    
    void notify(const std::string& message) {
        // 清理已失效的观察者，通知有效的观察者
        auto it = observers_.begin();
        while (it != observers_.end()) {
            if (auto observer = it->lock()) {
                observer->update(message);
                ++it;
            } else {
                it = observers_.erase(it);  // 移除失效的观察者
            }
        }
    }
};

void observer_pattern_example() {
    Subject subject;
    
    {
        auto observer1 = std::make_shared<ConcreteObserver>("观察者1");
        auto observer2 = std::make_shared<ConcreteObserver>("观察者2");
        
        subject.attach(observer1);
        subject.attach(observer2);
        
        subject.notify("消息1");  // 两个观察者都收到
        
    }  // observer1和observer2销毁
    
    subject.notify("消息2");  // 没有观察者收到（自动清理）
}
```

#### 6.3 检测悬空指针

```cpp
class Resource {
public:
    Resource() { std::cout << "Resource创建" << std::endl; }
    ~Resource() { std::cout << "Resource销毁" << std::endl; }
    void use() { std::cout << "使用Resource" << std::endl; }
};

void dangling_pointer_detection() {
    std::weak_ptr<Resource> weak_observer;
    
    {
        auto resource = std::make_shared<Resource>();
        weak_observer = resource;
        
        // 安全使用
        if (auto res = weak_observer.lock()) {
            res->use();  // ✓ 对象存在
        }
        
    }  // resource销毁
    
    // 检测对象是否还存在
    if (weak_observer.expired()) {
        std::cout << "Resource已被销毁" << std::endl;
    }
    
    // 尝试使用
    if (auto res = weak_observer.lock()) {
        res->use();  // 不会执行
    } else {
        std::cout << "无法使用，对象已销毁" << std::endl;
    }
}
```

### 7. enable_shared_from_this

#### 7.1 从this创建shared_ptr的问题

```cpp
class BadWidget {
public:
    std::shared_ptr<BadWidget> get_shared() {
        // ⚠️ 错误！创建了独立的控制块
        return std::shared_ptr<BadWidget>(this);
    }
};

void bad_shared_from_this() {
    auto widget1 = std::make_shared<BadWidget>();
    auto widget2 = widget1->get_shared();
    
    // ⚠️ 问题：widget1和widget2有独立的控制块
    // 当其中一个销毁时会delete this，导致另一个悬空
    std::cout << "widget1计数: " << widget1.use_count() << std::endl;  // 1
    std::cout << "widget2计数: " << widget2.use_count() << std::endl;  // 1
}
```

#### 7.2 正确使用 enable_shared_from_this

```cpp
class GoodWidget : public std::enable_shared_from_this<GoodWidget> {
public:
    std::shared_ptr<GoodWidget> get_shared() {
        return shared_from_this();  // ✓ 正确！
    }
    
    void register_callback() {
        // 在回调中需要保持对象存活
        auto callback = [self = shared_from_this()]() {
            // 使用self，确保对象在回调执行时仍然存在
            std::cout << "回调执行" << std::endl;
        };
        
        // 注册callback到某个系统
    }
};

void good_shared_from_this() {
    auto widget1 = std::make_shared<GoodWidget>();
    auto widget2 = widget1->get_shared();
    
    // ✓ widget1和widget2共享同一控制块
    std::cout << "widget1计数: " << widget1.use_count() << std::endl;  // 2
    std::cout << "widget2计数: " << widget2.use_count() << std::endl;  // 2
}

// ⚠️ 注意：只能在shared_ptr管理的对象上调用
void enable_shared_from_this_pitfall() {
    GoodWidget widget;  // 栈对象
    // auto ptr = widget.get_shared();  // ⚠️ 未定义行为！
    
    // ✓ 必须通过shared_ptr创建
    auto widget_ptr = std::make_shared<GoodWidget>();
    auto ptr = widget_ptr->get_shared();  // ✓ 正确
}
```

### 8. 成员函数详解

#### 8.1 shared_ptr 成员函数

```cpp
void shared_ptr_members() {
    auto ptr1 = std::make_shared<int>(42);
    
    // use_count(): 返回引用计数
    std::cout << "计数: " << ptr1.use_count() << std::endl;
    
    // unique(): 是否唯一所有者
    std::cout << "唯一: " << ptr1.unique() << std::endl;  // true
    
    auto ptr2 = ptr1;
    std::cout << "唯一: " << ptr1.unique() << std::endl;  // false
    
    // get(): 获取原始指针
    int* raw = ptr1.get();
    
    // reset(): 替换管理的对象
    ptr1.reset(new int(100));  // 旧对象可能被删除（如果没有其他引用）
    
    ptr1.reset();  // 释放对象，ptr1变为nullptr
    
    // swap(): 交换
    auto ptr3 = std::make_shared<int>(1);
    auto ptr4 = std::make_shared<int>(2);
    ptr3.swap(ptr4);
    
    // owner_before(): 用于比较（基于控制块地址）
    // 主要用于容器的排序
}
```

#### 8.2 weak_ptr 成员函数

```cpp
void weak_ptr_members() {
    std::weak_ptr<int> weak;
    
    {
        auto shared = std::make_shared<int>(42);
        weak = shared;
        
        // use_count(): 返回强引用计数
        std::cout << "强引用计数: " << weak.use_count() << std::endl;  // 1
        
        // expired(): 检查对象是否已销毁
        std::cout << "已过期: " << weak.expired() << std::endl;  // false
        
        // lock(): 创建shared_ptr（如果对象存在）
        if (auto locked = weak.lock()) {
            std::cout << "值: " << *locked << std::endl;  // 42
        }
        
    }  // shared销毁
    
    // 对象已销毁
    std::cout << "已过期: " << weak.expired() << std::endl;  // true
    std::cout << "强引用计数: " << weak.use_count() << std::endl;  // 0
    
    auto locked = weak.lock();
    if (!locked) {
        std::cout << "对象已不存在" << std::endl;
    }
    
    // reset(): 释放引用
    weak.reset();
    
    // swap(): 交换
    std::weak_ptr<int> weak2;
    weak.swap(weak2);
}
```

### 9. 性能考虑

#### 9.1 内存开销

```cpp
void memory_overhead() {
    // unique_ptr: 1个指针（8字节，64位系统）
    std::unique_ptr<int> uptr = std::make_unique<int>(42);
    std::cout << "unique_ptr大小: " << sizeof(uptr) << std::endl;  // 8
    
    // shared_ptr: 2个指针（16字节）
    std::shared_ptr<int> sptr = std::make_shared<int>(42);
    std::cout << "shared_ptr大小: " << sizeof(sptr) << std::endl;  // 16
    
    // weak_ptr: 2个指针（16字节）
    std::weak_ptr<int> wptr = sptr;
    std::cout << "weak_ptr大小: " << sizeof(wptr) << std::endl;  // 16
    
    // 控制块额外开销：
    // - 强引用计数: 4或8字节
    // - 弱引用计数: 4或8字节
    // - 删除器指针
    // - 分配器信息
    // 总计约 24-48字节
}
```

#### 9.2 性能对比

```cpp
#include <chrono>

void performance_comparison() {
    const int N = 1000000;
    
    // unique_ptr
    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < N; ++i) {
        auto ptr = std::make_unique<int>(i);
    }
    auto end = std::chrono::high_resolution_clock::now();
    auto unique_time = std::chrono::duration_cast<std::chrono::milliseconds>(
        end - start
    ).count();
    
    // shared_ptr
    start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < N; ++i) {
        auto ptr = std::make_shared<int>(i);
    }
    end = std::chrono::high_resolution_clock::now();
    auto shared_time = std::chrono::duration_cast<std::chrono::milliseconds>(
        end - start
    ).count();
    
    // shared_ptr拷贝（引用计数操作）
    auto shared = std::make_shared<int>(42);
    start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < N; ++i) {
        auto copy = shared;  // 原子操作
    }
    end = std::chrono::high_resolution_clock::now();
    auto copy_time = std::chrono::duration_cast<std::chrono::milliseconds>(
        end - start
    ).count();
    
    std::cout << "unique_ptr创建: " << unique_time << "ms" << std::endl;
    std::cout << "shared_ptr创建: " << shared_time << "ms" << std::endl;
    std::cout << "shared_ptr拷贝: " << copy_time << "ms" << std::endl;
}
```

### 10. 最佳实践和陷阱

#### 10.1 避免的错误

```cpp
void common_mistakes() {
    // ⚠️ 错误1：从原始指针创建多个shared_ptr
    int* raw = new int(42);
    std::shared_ptr<int> ptr1(raw);
    // std::shared_ptr<int> ptr2(raw);  // ⚠️ 双重删除！
    
    // ⚠️ 错误2：循环引用
    // （见前面的例子）
    
    // ⚠️ 错误3：在对象内部返回this的shared_ptr
    // 应该继承enable_shared_from_this
    
    // ⚠️ 错误4：不检查weak_ptr就使用
    std::weak_ptr<int> weak;
    // auto value = *weak.lock();  // ⚠️ 如果对象已销毁，崩溃！
    
    // ✓ 正确
    if (auto locked = weak.lock()) {
        auto value = *locked;
    }
    
    // ⚠️ 错误5：过度使用shared_ptr
    // 能用unique_ptr就用unique_ptr
}
```

#### 10.2 最佳实践

```cpp
// ✓ 1. 优先使用make_shared
auto ptr = std::make_shared<Widget>(10, 20);

// ✓ 2. 明确所有权
// - 独占：unique_ptr
// - 共享：shared_ptr
// - 观察：weak_ptr或原始指针

// ✓ 3. 函数参数传递
void take_ownership(std::shared_ptr<Widget> widget);  // 共享所有权
void use_widget(const std::shared_ptr<Widget>& widget);  // 只使用
void process(Widget* widget);  // 不涉及所有权

// ✓ 4. 避免循环引用
// 使用weak_ptr打破循环

// ✓ 5. 使用enable_shared_from_this
// 当需要在成员函数中创建shared_ptr时

// ✓ 6. 容器中使用
std::vector<std::shared_ptr<Base>> objects;  // 多态对象

// ✓ 7. 缓存使用weak_ptr
std::map<std::string, std::weak_ptr<Resource>> cache;
```

### 11. 实战示例：资源管理器

```cpp
class ResourceManager {
    std::map<std::string, std::weak_ptr<Resource>> resources_;
    std::mutex mtx_;
    
public:
    std::shared_ptr<Resource> get_resource(const std::string& name) {
        std::lock_guard<std::mutex> lock(mtx_);
        
        auto it = resources_.find(name);
        if (it != resources_.end()) {
            if (auto resource = it->second.lock()) {
                return resource;  // 资源仍然有效
            } else {
                resources_.erase(it);  // 清理过期资源
            }
        }
        
        // 创建新资源
        auto resource = std::make_shared<Resource>();
        resources_[name] = resource;
        return resource;
    }
    
    void cleanup() {
        std::lock_guard<std::mutex> lock(mtx_);
        
        auto it = resources_.begin();
        while (it != resources_.end()) {
            if (it->second.expired()) {
                it = resources_.erase(it);
            } else {
                ++it;
            }
        }
    }
};
```

### 12. 对比总结表

| 特性           | unique_ptr         | shared_ptr           | weak_ptr                   |
| -------------- | ------------------ | -------------------- | -------------------------- |
| **所有权**     | 独占               | 共享（引用计数）     | 不拥有（观察）             |
| **引用计数**   | 无                 | 强引用计数           | 不增加强引用计数           |
| **拷贝**       | 不可拷贝           | 可拷贝（增加计数）   | 可拷贝（不影响强引用）     |
| **移动**       | 可移动             | 可移动               | 可移动                     |
| **内存开销**   | 8字节              | 16字节+控制块        | 16字节                     |
| **线程安全**   | 无                 | 引用计数是原子的     | 引用计数是原子的           |
| **使用场景**   | 独占资源           | 共享资源、多所有者   | 缓存、观察者、打破循环引用 |
| **转换**       | 可转换为shared_ptr | 不可转换为unique_ptr | 必须转换为shared_ptr才能用 |
| **解引用**     | 直接使用           | 直接使用             | 需要lock()转换             |
| **检查有效性** | `if (ptr)`         | `if (ptr)`           | `if (!weak.expired())`     |

### 13. 核心要点

**shared_ptr**：
- ✓ 用于共享所有权（多个所有者）
- ✓ 优先使用 `make_shared`
- ✓ 引用计数线程安全，但对象本身不是
- ✓ 有额外的内存和性能开销
- ⚠️ 注意避免循环引用

**weak_ptr**：
- ✓ 不拥有对象，只是观察
- ✓ 用于打破循环引用
- ✓ 用于缓存和观察者模式
- ✓ 使用前必须 `lock()` 转换为 `shared_ptr`
- ✓ 用 `expired()` 检查对象是否存在

**选择原则**：
1. **默认选择 `unique_ptr`**（性能最优）
2. **需要共享时用 `shared_ptr`**
3. **需要观察但不拥有时用 `weak_ptr`**
4. **双向引用用 `shared_ptr` + `weak_ptr`**

**记住**：循环引用是 `shared_ptr` 最常见的陷阱，使用 `weak_ptr` 可以优雅地解决这个问题。


---

## 相关笔记
<!-- 自动生成 -->

- [unique_ptr、shared_ptr、weak_ptr的使用场景](notes/C++/unique_ptr、shared_ptr、weak_ptr的使用场景.md) - 相似度: 39% | 标签: C++, C++/unique_ptr、shared_ptr、weak_ptr的使用场景.md
- [设计一个简单的shared_ptr](notes/C++/设计一个简单的shared_ptr.md) - 相似度: 31% | 标签: C++, C++/设计一个简单的shared_ptr.md

