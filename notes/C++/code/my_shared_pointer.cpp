/**
 * 实现一个自己的shared_pointer的完整步骤：
 *
 * 1. 设计引用计数类RefCount
 *    - 维护一个计数器count
 *    - 提供addRef()增加引用计数
 *    - 提供release()减少引用计数
 *    - 提供getCount()获取当前计数值
 *
 * 2. 设计MySharedPointer模板类的核心成员
 *    - T* ptr：指向实际对象的指针
 *    - RefCount* refCount：指向引用计数对象的指针
 *
 * 3. 实现构造函数
 *    - 接受原始指针T* obj
 *    - 创建新的RefCount对象
 *    - 调用addRef()将计数设为1
 *
 * 4. 实现拷贝构造函数
 *    - 复制ptr和refCount指针
 *    - 调用addRef()增加引用计数
 *
 * 5. 实现赋值运算符operator=
 *    - 先处理自赋值情况
 *    - 减少当前对象的引用计数，必要时释放资源
 *    - 复制新的ptr和refCount
 *    - 增加新对象的引用计数
 *
 * 6. 实现析构函数
 *    - 调用release()减少引用计数
 *    - 如果计数为0，删除实际对象和RefCount对象
 *
 * 7. 实现解引用运算符
 *    - operator*()：返回*ptr
 *    - operator->()：返回ptr
 *
 * 8. 实现辅助函数
 *    - get()：返回原始指针ptr
 *    - use_count()：返回当前引用计数
 *    - reset()：重置为nullptr或新指针
 */
#include <iostream>
#include <stdexcept>

class Resource
{
    int data;

public:
    Resource(int value) : data(value) {}
};

class RefCount
{
    int count;

public:
    RefCount() : count(0) {}
    void addRef()
    {
        count++;
    }
    void release()
    {
        count--;
    }
    size_t getCount()
    {
        return count;
    }
};

template <typename T>
class MysharedPointer
{
private:
    T *ptr;
    RefCount *refCount;

public:
    /**
     * 构造函数
     * 使用场景是初始化一个shared_pointer对象
     * 例如：MysharedPointer<Resource> ptr(new Resource(10));
     */
    MysharedPointer(T *obj = nullptr) : ptr(obj), refCount(nullptr)
    {
        if (ptr)
        {
            refCount = new RefCount();
            refCount->addRef();
        }
    }

    /**
     * 拷贝构造函数
     * 使用场景是初始化一个shared_pointer对象，使用另一个shared_pointer对象初始化新的shared_pointer对象
     * 例如：MysharedPointer<Resource> ptr2(ptr1);
     */
    MysharedPointer(const MysharedPointer &other) : ptr(other.ptr), refCount(other.refCount)
    {
        if (refCount)
        {
            refCount->addRef();
        }
    }

    /**
     * 赋值运算符
     * 使用场景是给一个shared_pointer对象赋值
     * 例如：ptr1 = ptr2;
     */
    MysharedPointer &operator=(const MysharedPointer &other)
    {
        if (this != &other)
        {
            // 先减少当前引用计数
            if (refCount)
            {
                refCount->release();
                if (refCount->getCount() == 0)
                {
                    delete ptr;
                    delete refCount;
                }
            }
            // 复制新的指针和引用计数
            ptr = other.ptr;
            refCount = other.refCount;
            // 增加新的引用计数
            if (refCount)
            {
                refCount->addRef();
            }
        }
        return *this;
    }

    /**
     * 析构函数
     * 使用场景是销毁一个shared_pointer对象
     */
    ~MysharedPointer()
    {
        if (refCount)
        {
            refCount->release();
            if (refCount->getCount() == 0)
            {
                delete ptr;
                delete refCount;
            }
        }
    }

    /**
     * 解引用运算符
     * 使用场景是获取shared_pointer对象所指向的实际对象
     * 例如：*ptr
     */
    T &operator*()
    {
        if (!ptr)
        {
            throw std::runtime_error("Dereferencing null pointer");
        }
        return *ptr;
    }

    /**
     * 箭头运算符
     * 使用场景是获取shared_pointer对象所指向的实际对象的成员
     * 例如：ptr->data
     */
    T *operator->()
    {
        if (!ptr)
        {
            throw std::runtime_error("Dereferencing null pointer");
        }
        return ptr;
    }

    /**
     * 返回计数
     */
    size_t use_count()
    {
        return refCount ? refCount->getCount() : 0;
    }

    /**
     * 返回原始指针
     */
    T *get()
    {
        return ptr;
    }

    /**
     * 重置智能指针
     */
    void reset(T *newPtr = nullptr)
    {
        // 先减少当前引用计数
        if (refCount)
        {
            refCount->release();
            if (refCount->getCount() == 0)
            {
                delete ptr;
                delete refCount;
            }
        }

        // 设置新的指针和引用计数
        ptr = newPtr;
        if (ptr)
        {
            refCount = new RefCount();
            refCount->addRef();
        }
        else
        {
            refCount = nullptr;
        }
    }
};

int main()
{
    Resource *res = new Resource(10);
    MysharedPointer<Resource> ptr(res);
    auto ptr2 = ptr;
    std::cout << "ptr.use_count() = " << ptr.use_count() << std::endl;
    std::cout << "ptr2.use_count() = " << ptr2.use_count() << std::endl;

    return 0;
}