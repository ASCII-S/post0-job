---
created: '2025-10-19'
last_reviewed: null
next_review: '2025-10-19'
review_count: 0
difficulty: medium
mastery_level: 0.0
tags:
- C++
- C++/constexpr与const的区别.md
related_outlines: []
---
## 标准答案（可背诵）

`const`和`constexpr`都表示"不可修改"，但作用时机和用途不同：

1. **const**：运行时常量，表示变量在初始化后不可修改，值可以在运行时确定。主要用于保护变量不被意外修改。

2. **constexpr**：编译期常量，表示值必须在编译期就能确定，可用于需要编译期计算的场合（如数组大小、模板参数）。从C++11引入，C++14后还可修饰函数。

**核心区别**：
- `const`：运行时常量性承诺
- `constexpr`：编译期常量性要求，同时隐含const

---

## 详细讲解

### 1. const - 运行时常量

#### 1.1 基本概念

`const`表示一个变量一旦初始化就不能被修改，但其值可以在运行时才确定。

```cpp
int getUserInput() {
    int value;
    std::cin >> value;
    return value;
}

int main() {
    const int runtimeConst = getUserInput();  // 运行时才知道值
    // runtimeConst = 10;  // 错误：不能修改
    
    return 0;
}
```

#### 1.2 const的用途

**用途1：保护变量不被修改**

```cpp
const int maxSize = 100;
const std::string configPath = "/etc/config";
// maxSize = 200;  // 编译错误
```

**用途2：函数参数（避免拷贝和修改）**

```cpp
// 传递const引用，高效且安全
void process(const std::string& str) {
    // str[0] = 'A';  // 错误：不能修改
    std::cout << str << std::endl;  // OK：可以读取
}
```

**用途3：const成员函数**

```cpp
class MyClass {
private:
    int value;
public:
    // const成员函数承诺不修改对象状态
    int getValue() const {
        // value = 10;  // 错误：不能修改成员变量
        return value;
    }
    
    void setValue(int v) {
        value = v;  // OK：非const成员函数可以修改
    }
};
```

**用途4：指针的const**

```cpp
int value = 10;

// 1. 指向常量的指针：不能通过指针修改值
const int* ptr1 = &value;
// *ptr1 = 20;  // 错误
ptr1 = nullptr;  // OK：可以改变指针本身

// 2. 常量指针：指针本身不能改变
int* const ptr2 = &value;
*ptr2 = 20;  // OK：可以修改值
// ptr2 = nullptr;  // 错误：不能改变指针

// 3. 指向常量的常量指针
const int* const ptr3 = &value;
// *ptr3 = 20;  // 错误
// ptr3 = nullptr;  // 错误
```

---

### 2. constexpr - 编译期常量

#### 2.1 基本概念（C++11引入）

`constexpr`表示值必须在编译期就能确定，可以用于任何需要编译期常量的地方。

```cpp
constexpr int compileTimeConst = 100;  // 编译期常量
constexpr int square = 10 * 10;        // 编译期计算

// 可以用于数组大小
int array[compileTimeConst];  // OK

// 可以用于模板参数
std::array<int, compileTimeConst> arr;  // OK
```

#### 2.2 constexpr变量

```cpp
// 正确：编译期可确定
constexpr int maxValue = 100;
constexpr double pi = 3.14159;
constexpr int result = 2 + 3 * 4;

// 错误：运行时才能确定
int getUserInput();
// constexpr int userValue = getUserInput();  // 编译错误

// const可以，因为它允许运行时确定
const int userValue = getUserInput();  // OK
```

#### 2.3 constexpr函数

**C++11的constexpr函数**（限制严格）：

```cpp
// C++11：只能有一条return语句
constexpr int square(int x) {
    return x * x;
}

constexpr int getValue() {
    return 42;
}

// 编译期使用
constexpr int a = square(10);  // 编译期计算，a = 100
int array[square(5)];          // 可用于数组大小

// 运行期使用
int x = 10;
int b = square(x);  // 运行期计算
```

**C++14的constexpr函数**（放宽限制）：

```cpp
// C++14：可以有多条语句、循环、分支
constexpr int factorial(int n) {
    int result = 1;
    for (int i = 1; i <= n; ++i) {
        result *= i;
    }
    return result;
}

constexpr int fibonacci(int n) {
    if (n <= 1) return n;
    return fibonacci(n - 1) + fibonacci(n - 2);
}

// 编译期计算
constexpr int fact5 = factorial(5);  // 120，编译期就算好了
static_assert(fact5 == 120, "Factorial error");
```

**C++17的constexpr lambda**：

```cpp
auto squared = [](int x) constexpr { return x * x; };
constexpr int result = squared(5);  // 编译期计算
```

#### 2.4 constexpr与类

**constexpr构造函数**：

```cpp
class Point {
private:
    int x_, y_;
public:
    constexpr Point(int x, int y) : x_(x), y_(y) {}
    
    constexpr int getX() const { return x_; }
    constexpr int getY() const { return y_; }
    
    constexpr int distanceSquared() const {
        return x_ * x_ + y_ * y_;
    }
};

// 编译期创建对象
constexpr Point origin(0, 0);
constexpr Point p(3, 4);
constexpr int dist = p.distanceSquared();  // 编译期计算，dist = 25

// 可用于模板参数等场合
std::array<int, origin.getX()> arr;  // OK
```

---

### 3. const vs constexpr 核心区别

#### 3.1 值确定时机

```cpp
int getUserInput() { return 42; }

// const：可以是运行时值
const int a = getUserInput();        // OK：运行时确定
const int b = 100;                   // OK：编译期就知道

// constexpr：必须是编译期值
// constexpr int c = getUserInput(); // 错误：编译期不能确定
constexpr int d = 100;               // OK：编译期常量
```

#### 3.2 编译期要求

```cpp
const int constValue = 100;
constexpr int constexprValue = 100;

// 数组大小需要编译期常量
// int arr1[constValue];      // 某些编译器可能允许（VLA扩展），但非标准
int arr2[constexprValue];     // 标准C++，OK

// 模板参数需要编译期常量
// template <int N> class Array {};
// Array<constValue> a1;      // 可能错误（取决于constValue的初始化）
Array<constexprValue> a2;     // OK
```

#### 3.3 函数的差异

```cpp
// const函数：只是承诺不修改成员变量
class MyClass {
    int value;
public:
    int getValue() const {  // const成员函数
        return value;  // 可以返回运行时值
    }
};

// constexpr函数：可以在编译期求值
constexpr int square(int x) {
    return x * x;
}

// 使用
MyClass obj;
int a = obj.getValue();  // 运行时调用

constexpr int b = square(10);  // 编译期计算
int x = 10;
int c = square(x);  // 运行时计算（constexpr函数也可以运行时调用）
```

### 4. 组合使用

#### 4.1 constexpr隐含const

```cpp
constexpr int value = 100;
// 等价于
const int value = 100;  // 但const不等价于constexpr

// 对于变量，constexpr自动是const的
constexpr int* ptr = nullptr;  // 等价于 int* const ptr
```

#### 4.2 constexpr函数返回const引用

```cpp
class Data {
    int value;
public:
    constexpr Data(int v) : value(v) {}
    constexpr const int& getValue() const { return value; }
};
```

### 5. 实战应用场景

#### 5.1 编译期计算优化

```cpp
// 传统方式：运行时计算
const double PI = 3.14159;
double circleArea(double r) {
    return PI * r * r;
}

// constexpr方式：可以编译期计算
constexpr double PI = 3.14159;
constexpr double circleArea(double r) {
    return PI * r * r;
}

// 使用
double arr[static_cast<int>(circleArea(10))];  // 编译期计算大小
```

#### 5.2 编译期字符串处理（C++17及以后）

```cpp
constexpr size_t stringLength(const char* str) {
    size_t len = 0;
    while (str[len] != '\0') {
        ++len;
    }
    return len;
}

constexpr size_t len = stringLength("Hello");  // 编译期计算
static_assert(len == 5, "Length should be 5");
```

#### 5.3 编译期配置

```cpp
// 编译期决定缓冲区大小
constexpr size_t calculateBufferSize(bool isDebug) {
    return isDebug ? 1024 * 1024 : 4096;
}

constexpr bool DEBUG_MODE = true;
constexpr size_t BUFFER_SIZE = calculateBufferSize(DEBUG_MODE);

char buffer[BUFFER_SIZE];  // 根据编译期配置确定大小
```

### 6. 使用建议

#### 6.1 何时使用const

- 运行时才能确定的值
- 保护变量不被修改
- 函数参数（避免拷贝）
- 成员函数（承诺不修改对象）

```cpp
void processConfig(const std::string& configFile) {
    const auto config = loadConfig(configFile);  // 运行时加载
    // ...
}
```

#### 6.2 何时使用constexpr

- 需要编译期常量的场合（数组大小、模板参数）
- 性能关键的计算（可以提前到编译期）
- 可以用编译期计算的数学函数

```cpp
// 编译期就计算好所有阶乘值
constexpr std::array<int, 10> factorials = []() {
    std::array<int, 10> result{};
    result[0] = 1;
    for (int i = 1; i < 10; ++i) {
        result[i] = result[i - 1] * i;
    }
    return result;
}();
```

#### 6.3 渐进策略

```cpp
// 优先尝试constexpr
constexpr int value = compute();

// 如果不能编译期确定，降级为const
const int value = compute();

// 如果需要修改，使用普通变量
int value = compute();
```

### 7. C++20的进一步增强

#### 7.1 consteval - 必须编译期求值

```cpp
// C++20：consteval确保只能编译期调用
consteval int square(int x) {
    return x * x;
}

constexpr int a = square(10);  // OK：编译期
// int x = 10;
// int b = square(x);  // 错误：不能运行时调用
```

#### 7.2 constinit - 编译期初始化

```cpp
// C++20：确保静态变量在编译期初始化
constinit int globalValue = 42;

// 避免静态初始化顺序问题
```

### 8. 常见面试延伸问题

**Q: constexpr一定比const快吗？**
- 如果值能在编译期确定，constexpr会更快（零运行时开销）
- 但constexpr函数如果参数是运行时的，会退化为普通函数
- const主要是语义上的保护，不直接影响性能

**Q: 为什么需要constexpr？const不够用吗？**
- const不能保证编译期求值
- 有些场景必须要编译期常量（数组大小、模板参数、case标签）
- constexpr提供编译期计算能力，提升性能

**Q: constexpr函数什么时候会在运行期执行？**
- 当参数不是常量表达式时
- 当结果不用于需要编译期常量的地方时
- 编译器优化级别较低时

### 9. 对比总结表

| 特性         | const          | constexpr    |
| ------------ | -------------- | ------------ |
| 引入版本     | C++98          | C++11        |
| 值确定时机   | 运行时或编译时 | 必须编译时   |
| 用于变量     | ✅              | ✅            |
| 用于函数     | ✅（成员函数）  | ✅（C++11起） |
| 用于构造函数 | ✅              | ✅（C++11起） |
| 数组大小     | ❌（非标准）    | ✅            |
| 模板参数     | ❌              | ✅            |
| 隐含对方     | ❌              | ✅（变量）    |
| 性能优势     | 无             | 编译期计算   |

### 10. 总结

- **const**：表达"不可修改"的语义，值可以运行时确定，是一种约束和保护
- **constexpr**：表达"编译期常量"的要求，必须编译期确定，提供编译期计算能力

选择原则：
1. 如果值需要编译期确定 → 使用`constexpr`
2. 如果只是保护不被修改 → 使用`const`
3. 如果不确定能否编译期确定，先尝试`constexpr`，不行再用`const`

现代C++倾向于更多使用`constexpr`，因为它提供了更强的保证和更好的性能优化机会。


---

## 相关笔记
<!-- 自动生成 -->

暂无相关笔记

