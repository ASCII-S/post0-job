---
created: '2025-10-19'
last_reviewed: null
next_review: '2025-10-19'
review_count: 0
difficulty: medium
mastery_level: 0.0
tags:
- cuda
- cuda/CUDA_Occupancy_Calculator使用.md
related_outlines: []
---
# CUDA Occupancy Calculator使用

## 面试标准答案（可背诵）

**Q: 如何使用CUDA Occupancy Calculator优化kernel占用率？**

CUDA提供两种占用率计算工具：1）cudaOccupancyMaxPotentialBlockSize API自动计算最优块大小；2）cudaOccupancyMaxActiveBlocksPerMultiprocessor API计算给定配置下的最大活跃块数。使用方法是先调用API获取推荐配置，然后结合实际性能测试验证效果。还可以使用Excel版Occupancy Calculator进行离线分析，帮助理解不同参数对占用率的影响。

## 详细技术讲解

### 1. CUDA Occupancy Calculator概述

#### 1.1 工具类型和用途
- **运行时API**：集成在CUDA Runtime中，可以在程序运行时动态计算
- **Excel计算器**：离线分析工具，用于深入理解占用率限制因素
- **Nsight Compute**：综合性能分析工具，提供详细的占用率分析
- **编译器信息**：编译时输出的资源使用信息

#### 1.2 占用率计算的重要性
- **自动化配置**：避免手动试错，快速找到合理的启动配置
- **跨架构适配**：不同GPU架构的资源限制不同，需要动态适配
- **性能优化指导**：识别性能瓶颈，指导优化方向
- **资源平衡**：在寄存器、共享内存、块大小之间找到最佳平衡

### 2. cudaOccupancyMaxPotentialBlockSize API

#### 2.1 API接口详解
```cuda
cudaError_t cudaOccupancyMaxPotentialBlockSize(
    int *minGridSize,           // 输出：推荐的最小Grid大小
    int *blockSize,             // 输出：推荐的Block大小
    const void *func,           // 输入：要分析的kernel函数指针
    size_t dynamicSMemSize,     // 输入：动态共享内存大小（字节）
    int blockSizeLimit          // 输入：块大小上限（0表示无限制）
);
```

#### 2.2 基本使用示例
```cuda
// 示例kernel函数
__global__ void example_kernel(float* data, int N) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < N) {
        data[idx] = data[idx] * 2.0f + 1.0f;
    }
}

// 使用Occupancy Calculator
void optimize_kernel_launch(float* d_data, int N) {
    int minGridSize, blockSize;
    
    // 获取推荐的块大小
    cudaError_t result = cudaOccupancyMaxPotentialBlockSize(
        &minGridSize,           // 输出推荐的最小Grid大小
        &blockSize,             // 输出推荐的Block大小  
        example_kernel,         // kernel函数
        0,                      // 无动态共享内存
        0                       // 无块大小限制
    );
    
    if (result != cudaSuccess) {
        printf("Error: %s\n", cudaGetErrorString(result));
        return;
    }
    
    printf("Recommended block size: %d\n", blockSize);
    printf("Minimum grid size for maximum occupancy: %d\n", minGridSize);
    
    // 计算实际需要的Grid大小
    int gridSize = (N + blockSize - 1) / blockSize;
    
    printf("Actual grid size: %d\n", gridSize);
    printf("Total threads: %d\n", gridSize * blockSize);
    
    // 启动kernel
    example_kernel<<<gridSize, blockSize>>>(d_data, N);
    
    // 检查错误
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("Kernel launch error: %s\n", cudaGetErrorString(err));
    }
}
```

#### 2.3 动态共享内存的处理
```cuda
// 使用动态共享内存的kernel
__global__ void dynamic_shared_kernel(float* input, float* output, int N) {
    extern __shared__ float shared_data[];
    
    int tid = threadIdx.x;
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    
    // 使用动态共享内存
    if (idx < N) {
        shared_data[tid] = input[idx];
    } else {
        shared_data[tid] = 0.0f;
    }
    
    __syncthreads();
    
    // 简单的归约操作
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            shared_data[tid] += shared_data[tid + s];
        }
        __syncthreads();
    }
    
    if (tid == 0 && blockIdx.x < N) {
        output[blockIdx.x] = shared_data[0];
    }
}

// 考虑动态共享内存的占用率计算
void optimize_dynamic_shared_kernel(float* d_input, float* d_output, int N) {
    // 测试不同的共享内存使用量
    size_t shared_mem_sizes[] = {1024, 2048, 4096, 8192};  // 字节
    
    for (int i = 0; i < 4; i++) {
        size_t shared_size = shared_mem_sizes[i];
        int threads_for_shared = shared_size / sizeof(float);
        
        int minGridSize, blockSize;
        
        // 计算在指定共享内存使用量下的最优块大小
        cudaOccupancyMaxPotentialBlockSize(
            &minGridSize,
            &blockSize,
            dynamic_shared_kernel,
            shared_size,        // 动态共享内存大小
            threads_for_shared  // 块大小限制
        );
        
        printf("Shared memory: %zu bytes\n", shared_size);
        printf("  Recommended block size: %d\n", blockSize);
        printf("  Threads per block: %d\n", blockSize);
        printf("  Shared memory per thread: %.1f bytes\n", 
               (float)shared_size / blockSize);
        
        // 验证块大小不超过共享内存能支持的线程数
        if (blockSize <= threads_for_shared) {
            int gridSize = (N + blockSize - 1) / blockSize;
            
            printf("  Launching with grid size: %d\n", gridSize);
            dynamic_shared_kernel<<<gridSize, blockSize, shared_size>>>
                (d_input, d_output, N);
            
            cudaDeviceSynchronize();
            
            // 检查kernel执行是否成功
            cudaError_t err = cudaGetLastError();
            if (err == cudaSuccess) {
                printf("  ✓ Kernel executed successfully\n");
            } else {
                printf("  ✗ Kernel error: %s\n", cudaGetErrorString(err));
            }
        } else {
            printf("  ✗ Block size exceeds shared memory capacity\n");
        }
        printf("\n");
    }
}
```

### 3. cudaOccupancyMaxActiveBlocksPerMultiprocessor API

#### 3.1 API接口详解
```cuda
cudaError_t cudaOccupancyMaxActiveBlocksPerMultiprocessor(
    int *numBlocks,             // 输出：每个SM最大活跃块数
    const void *func,           // 输入：kernel函数指针
    int blockSize,              // 输入：块大小
    size_t dynamicSMemSize      // 输入：动态共享内存大小
);
```

#### 3.2 详细分析示例
```cuda
// 分析不同块大小的占用率
void analyze_occupancy_for_different_block_sizes() {
    printf("Occupancy Analysis for example_kernel:\n");
    printf("%-12s %-15s %-15s %-15s\n", 
           "Block Size", "Blocks per SM", "Threads per SM", "Occupancy %");
    printf("-----------------------------------------------------------\n");
    
    // 获取设备属性
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    
    int max_threads_per_sm = prop.maxThreadsPerMultiProcessor;
    
    // 测试不同的块大小
    for (int blockSize = 32; blockSize <= 1024; blockSize += 32) {
        int maxActiveBlocks;
        
        cudaError_t result = cudaOccupancyMaxActiveBlocksPerMultiprocessor(
            &maxActiveBlocks,
            example_kernel,
            blockSize,
            0  // 无动态共享内存
        );
        
        if (result == cudaSuccess) {
            int active_threads = maxActiveBlocks * blockSize;
            float occupancy = (float)active_threads / max_threads_per_sm * 100;
            
            printf("%-12d %-15d %-15d %-15.1f\n", 
                   blockSize, maxActiveBlocks, active_threads, occupancy);
        } else {
            printf("%-12d Error: %s\n", blockSize, cudaGetErrorString(result));
        }
    }
}

// 分析kernel资源使用对占用率的影响
void analyze_kernel_resource_impact() {
    // 获取kernel的资源使用信息
    cudaFuncAttributes attr;
    cudaFuncGetAttributes(&attr, example_kernel);
    
    printf("Kernel Resource Usage:\n");
    printf("  Registers per thread: %d\n", attr.numRegs);
    printf("  Shared memory per block: %zu bytes\n", attr.sharedSizeBytes);
    printf("  Local memory per thread: %zu bytes\n", attr.localSizeBytes);
    printf("  Constant memory usage: %zu bytes\n", attr.constSizeBytes);
    printf("  Max threads per block: %d\n", attr.maxThreadsPerBlock);
    printf("\n");
    
    // 获取设备限制
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    
    printf("Device Limits:\n");
    printf("  Total registers per SM: %d\n", prop.regsPerMultiprocessor);
    printf("  Shared memory per SM: %zu bytes\n", prop.sharedMemPerMultiprocessor);
    printf("  Max blocks per SM: %d\n", prop.maxBlocksPerMultiProcessor);
    printf("  Max threads per SM: %d\n", prop.maxThreadsPerMultiProcessor);
    printf("  Max warps per SM: %d\n", prop.maxThreadsPerMultiProcessor / 32);
    printf("\n");
    
    // 计算理论限制
    printf("Theoretical Limits Analysis:\n");
    
    // 寄存器限制
    if (attr.numRegs > 0) {
        int max_threads_by_regs = prop.regsPerMultiprocessor / attr.numRegs;
        printf("  Max threads limited by registers: %d\n", max_threads_by_regs);
    }
    
    // 分析不同块大小下的限制因素
    printf("\nDetailed Analysis by Block Size:\n");
    printf("%-10s %-12s %-12s %-12s %-12s %-10s\n",
           "BlockSize", "RegLimit", "ShmemLimit", "BlockLimit", "ActualLimit", "Bottleneck");
    printf("--------------------------------------------------------------------------\n");
    
    for (int blockSize = 128; blockSize <= 512; blockSize += 128) {
        // 寄存器限制的块数
        int blocks_by_regs = INT_MAX;
        if (attr.numRegs > 0) {
            int threads_by_regs = prop.regsPerMultiprocessor / attr.numRegs;
            blocks_by_regs = threads_by_regs / blockSize;
        }
        
        // 共享内存限制的块数
        int blocks_by_shmem = INT_MAX;
        if (attr.sharedSizeBytes > 0) {
            blocks_by_shmem = prop.sharedMemPerMultiprocessor / attr.sharedSizeBytes;
        }
        
        // 硬件块数限制
        int blocks_by_hw = prop.maxBlocksPerMultiProcessor;
        
        // 实际限制
        int actual_blocks = std::min({blocks_by_regs, blocks_by_shmem, blocks_by_hw});
        
        // 确定瓶颈
        const char* bottleneck = "Unknown";
        if (actual_blocks == blocks_by_regs) bottleneck = "Registers";
        else if (actual_blocks == blocks_by_shmem) bottleneck = "SharedMem";
        else if (actual_blocks == blocks_by_hw) bottleneck = "Hardware";
        
        printf("%-10d %-12d %-12d %-12d %-12d %-10s\n",
               blockSize, blocks_by_regs, blocks_by_shmem, 
               blocks_by_hw, actual_blocks, bottleneck);
    }
}
```

### 4. 高级使用技巧

#### 4.1 模板化的占用率优化
```cuda
template<typename KernelFunc>
class OccupancyOptimizer {
private:
    KernelFunc kernel;
    size_t dynamic_shared_mem;
    int block_size_limit;
    
public:
    OccupancyOptimizer(KernelFunc k, size_t shared_mem = 0, int limit = 0) 
        : kernel(k), dynamic_shared_mem(shared_mem), block_size_limit(limit) {}
    
    struct LaunchConfig {
        int grid_size;
        int block_size;
        float theoretical_occupancy;
        size_t shared_mem_size;
    };
    
    LaunchConfig get_optimal_config(int total_threads) {
        int min_grid_size, block_size;
        
        cudaOccupancyMaxPotentialBlockSize(&min_grid_size, &block_size, 
                                           kernel, dynamic_shared_mem, 
                                           block_size_limit);
        
        // 计算实际grid大小
        int grid_size = (total_threads + block_size - 1) / block_size;
        
        // 计算理论占用率
        int max_active_blocks;
        cudaOccupancyMaxActiveBlocksPerMultiprocessor(&max_active_blocks, 
                                                      kernel, block_size, 
                                                      dynamic_shared_mem);
        
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, 0);
        
        float occupancy = (float)(max_active_blocks * block_size) / 
                         prop.maxThreadsPerMultiProcessor;
        
        return {grid_size, block_size, occupancy, dynamic_shared_mem};
    }
    
    // 性能基准测试
    float benchmark_config(const LaunchConfig& config, int iterations = 10) {
        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        
        // 预热
        launch_kernel(config);
        cudaDeviceSynchronize();
        
        cudaEventRecord(start);
        for (int i = 0; i < iterations; i++) {
            launch_kernel(config);
        }
        cudaEventRecord(stop);
        
        cudaEventSynchronize(stop);
        float milliseconds;
        cudaEventElapsedTime(&milliseconds, start, stop);
        
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
        
        return milliseconds / iterations;
    }
    
private:
    void launch_kernel(const LaunchConfig& config) {
        // 这里需要根据具体kernel类型实现
        // 示例：kernel<<<config.grid_size, config.block_size, config.shared_mem_size>>>(args);
    }
};

// 使用示例
void optimize_with_template(float* data, int N) {
    OccupancyOptimizer optimizer(example_kernel);
    
    auto config = optimizer.get_optimal_config(N);
    
    printf("Optimal Configuration:\n");
    printf("  Grid size: %d\n", config.grid_size);
    printf("  Block size: %d\n", config.block_size);
    printf("  Theoretical occupancy: %.1f%%\n", config.theoretical_occupancy * 100);
    
    // 性能测试
    float time = optimizer.benchmark_config(config);
    printf("  Execution time: %.3f ms\n", time);
}
```

#### 4.2 动态配置调整
```cuda
// 根据问题规模动态调整配置
struct AdaptiveConfig {
    int small_problem_threshold;
    int large_problem_threshold;
    
    struct ConfigSet {
        int block_size;
        size_t shared_mem;
        const char* description;
    };
    
    ConfigSet small_config;
    ConfigSet medium_config;
    ConfigSet large_config;
};

LaunchConfig select_adaptive_config(int problem_size, const AdaptiveConfig& adaptive) {
    AdaptiveConfig::ConfigSet selected;
    
    if (problem_size < adaptive.small_problem_threshold) {
        selected = adaptive.small_config;
    } else if (problem_size < adaptive.large_problem_threshold) {
        selected = adaptive.medium_config;
    } else {
        selected = adaptive.large_config;
    }
    
    printf("Problem size: %d, using %s\n", problem_size, selected.description);
    
    // 验证配置的有效性
    int max_active_blocks;
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(&max_active_blocks,
                                                  example_kernel,
                                                  selected.block_size,
                                                  selected.shared_mem);
    
    int grid_size = (problem_size + selected.block_size - 1) / selected.block_size;
    
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    float occupancy = (float)(max_active_blocks * selected.block_size) / 
                     prop.maxThreadsPerMultiProcessor;
    
    return {grid_size, selected.block_size, occupancy, selected.shared_mem};
}

void demonstrate_adaptive_configuration() {
    AdaptiveConfig adaptive = {
        .small_problem_threshold = 10000,
        .large_problem_threshold = 100000,
        .small_config = {128, 0, "small problem config"},
        .medium_config = {256, 1024, "medium problem config"},
        .large_config = {512, 2048, "large problem config"}
    };
    
    int test_sizes[] = {1000, 50000, 500000};
    
    for (int i = 0; i < 3; i++) {
        auto config = select_adaptive_config(test_sizes[i], adaptive);
        printf("  Grid: %d, Block: %d, Occupancy: %.1f%%\n\n",
               config.grid_size, config.block_size, config.theoretical_occupancy * 100);
    }
}
```

### 5. Excel Occupancy Calculator

#### 5.1 Excel工具的使用
```
NVIDIA提供的Excel版Occupancy Calculator特点：
1. 离线分析工具，无需编写代码
2. 可视化界面，直观显示占用率限制因素
3. 支持不同GPU架构的分析
4. 可以进行What-if分析

使用步骤：
1. 从NVIDIA开发者网站下载Excel文件
2. 选择目标GPU架构
3. 输入kernel的资源使用情况：
   - 每线程寄存器数
   - 每块共享内存使用量
   - 块大小
4. 查看计算结果和限制因素分析
```

#### 5.2 手动计算占用率（理解原理）
```cuda
// 手动计算占用率的示例代码
void manual_occupancy_calculation() {
    // 设备属性
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    
    // Kernel属性
    cudaFuncAttributes attr;
    cudaFuncGetAttributes(&attr, example_kernel);
    
    // 配置参数
    int block_size = 256;
    size_t dynamic_shared_mem = 0;
    
    printf("Manual Occupancy Calculation:\n");
    printf("=================================\n");
    
    printf("Device Properties:\n");
    printf("  Max threads per SM: %d\n", prop.maxThreadsPerMultiProcessor);
    printf("  Max blocks per SM: %d\n", prop.maxBlocksPerMultiProcessor);
    printf("  Registers per SM: %d\n", prop.regsPerMultiprocessor);
    printf("  Shared memory per SM: %zu KB\n", prop.sharedMemPerMultiprocessor / 1024);
    printf("\n");
    
    printf("Kernel Properties:\n");
    printf("  Registers per thread: %d\n", attr.numRegs);
    printf("  Static shared memory: %zu bytes\n", attr.sharedSizeBytes);
    printf("  Block size: %d threads\n", block_size);
    printf("  Dynamic shared memory: %zu bytes\n", dynamic_shared_mem);
    printf("\n");
    
    // 计算各种限制
    printf("Occupancy Calculations:\n");
    
    // 1. 线程数限制
    int max_warps_per_sm = prop.maxThreadsPerMultiProcessor / 32;
    int warps_per_block = (block_size + 31) / 32;
    int blocks_by_threads = max_warps_per_sm / warps_per_block;
    printf("  Max blocks by thread limit: %d\n", blocks_by_threads);
    
    // 2. 硬件块数限制
    int blocks_by_hw = prop.maxBlocksPerMultiProcessor;
    printf("  Max blocks by hardware limit: %d\n", blocks_by_hw);
    
    // 3. 寄存器限制
    int blocks_by_regs = INT_MAX;
    if (attr.numRegs > 0) {
        int total_regs_per_block = attr.numRegs * block_size;
        blocks_by_regs = prop.regsPerMultiprocessor / total_regs_per_block;
    }
    printf("  Max blocks by register limit: %d\n", blocks_by_regs);
    
    // 4. 共享内存限制
    int blocks_by_shmem = INT_MAX;
    size_t total_shmem_per_block = attr.sharedSizeBytes + dynamic_shared_mem;
    if (total_shmem_per_block > 0) {
        blocks_by_shmem = prop.sharedMemPerMultiprocessor / total_shmem_per_block;
    }
    printf("  Max blocks by shared memory limit: %d\n", blocks_by_shmem);
    
    // 5. 最终结果
    int actual_blocks = std::min({blocks_by_threads, blocks_by_hw, blocks_by_regs, blocks_by_shmem});
    int actual_warps = actual_blocks * warps_per_block;
    float occupancy = (float)actual_warps / max_warps_per_sm;
    
    printf("\n");
    printf("Final Results:\n");
    printf("  Active blocks per SM: %d\n", actual_blocks);
    printf("  Active warps per SM: %d\n", actual_warps);
    printf("  Theoretical occupancy: %.1f%%\n", occupancy * 100);
    
    // 找出限制因素
    printf("  Limiting factor: ");
    if (actual_blocks == blocks_by_threads) printf("Thread count\n");
    else if (actual_blocks == blocks_by_hw) printf("Hardware block limit\n");
    else if (actual_blocks == blocks_by_regs) printf("Register usage\n");
    else if (actual_blocks == blocks_by_shmem) printf("Shared memory usage\n");
    
    // 与API结果对比
    int api_blocks;
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(&api_blocks, example_kernel, 
                                                  block_size, dynamic_shared_mem);
    printf("  API result: %d blocks (%.1f%% occupancy)\n", 
           api_blocks, (float)(api_blocks * warps_per_block) / max_warps_per_sm * 100);
}
```

### 6. 实际项目中的应用

#### 6.1 构建占用率监控系统
```cuda
class OccupancyMonitor {
private:
    struct KernelStats {
        std::string name;
        int optimal_block_size;
        float theoretical_occupancy;
        float measured_performance;
        std::chrono::time_point<std::chrono::high_resolution_clock> last_update;
    };
    
    std::unordered_map<std::string, KernelStats> kernel_database;
    
public:
    template<typename KernelFunc>
    void register_kernel(const std::string& name, KernelFunc kernel) {
        int min_grid_size, block_size;
        
        cudaOccupancyMaxPotentialBlockSize(&min_grid_size, &block_size, kernel, 0, 0);
        
        int max_active_blocks;
        cudaOccupancyMaxActiveBlocksPerMultiprocessor(&max_active_blocks, kernel, block_size, 0);
        
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, 0);
        
        float occupancy = (float)(max_active_blocks * block_size) / prop.maxThreadsPerMultiProcessor;
        
        kernel_database[name] = {
            name, block_size, occupancy, 0.0f, std::chrono::high_resolution_clock::now()
        };
        
        printf("Registered kernel '%s': block_size=%d, occupancy=%.1f%%\n", 
               name.c_str(), block_size, occupancy * 100);
    }
    
    void update_performance(const std::string& name, float execution_time) {
        if (kernel_database.find(name) != kernel_database.end()) {
            kernel_database[name].measured_performance = execution_time;
            kernel_database[name].last_update = std::chrono::high_resolution_clock::now();
        }
    }
    
    void print_summary() {
        printf("\nKernel Performance Summary:\n");
        printf("%-20s %-12s %-12s %-15s\n", "Kernel", "Block Size", "Occupancy%", "Time (ms)");
        printf("----------------------------------------------------------\n");
        
        for (const auto& [name, stats] : kernel_database) {
            printf("%-20s %-12d %-12.1f %-15.3f\n", 
                   name.c_str(), stats.optimal_block_size, 
                   stats.theoretical_occupancy * 100, stats.measured_performance);
        }
    }
    
    void analyze_performance_vs_occupancy() {
        printf("\nPerformance vs Occupancy Analysis:\n");
        
        std::vector<std::pair<float, float>> data;  // (occupancy, performance)
        
        for (const auto& [name, stats] : kernel_database) {
            if (stats.measured_performance > 0) {
                data.push_back({stats.theoretical_occupancy, stats.measured_performance});
            }
        }
        
        if (data.size() >= 2) {
            // 简单的相关性分析
            float avg_occupancy = 0, avg_performance = 0;
            for (const auto& [occ, perf] : data) {
                avg_occupancy += occ;
                avg_performance += perf;
            }
            avg_occupancy /= data.size();
            avg_performance /= data.size();
            
            float correlation = 0;
            float var_occ = 0, var_perf = 0;
            
            for (const auto& [occ, perf] : data) {
                correlation += (occ - avg_occupancy) * (perf - avg_performance);
                var_occ += (occ - avg_occupancy) * (occ - avg_occupancy);
                var_perf += (perf - avg_performance) * (perf - avg_performance);
            }
            
            if (var_occ > 0 && var_perf > 0) {
                correlation /= sqrt(var_occ * var_perf);
                printf("Occupancy-Performance correlation: %.3f\n", correlation);
                
                if (correlation > 0.5) {
                    printf("Strong positive correlation: Higher occupancy → Better performance\n");
                } else if (correlation < -0.5) {
                    printf("Strong negative correlation: Higher occupancy → Worse performance\n");
                } else {
                    printf("Weak correlation: Occupancy is not the main performance factor\n");
                }
            }
        }
    }
};

// 使用示例
void demonstrate_occupancy_monitoring() {
    OccupancyMonitor monitor;
    
    // 注册多个kernel
    monitor.register_kernel("vector_add", example_kernel);
    monitor.register_kernel("matrix_mul", example_kernel);  // 示例中使用同一kernel
    
    // 模拟性能测试
    monitor.update_performance("vector_add", 1.5f);
    monitor.update_performance("matrix_mul", 3.2f);
    
    monitor.print_summary();
    monitor.analyze_performance_vs_occupancy();
}
```

### 7. 面试重点和实践建议

#### 7.1 常见面试问题
**Q1: cudaOccupancyMaxPotentialBlockSize和手动选择块大小有什么区别？**
- API自动考虑所有硬件限制因素
- 手动选择可能遗漏某些限制
- API结果是理论最优，实际性能需要测试验证

**Q2: 为什么不能完全依赖Occupancy Calculator的结果？**
- 只考虑静态因素，不考虑内存访问模式
- 不考虑算法特性和数据局部性
- 需要结合实际性能测试

**Q3: 如何在生产环境中应用占用率优化？**
- 建立性能基准测试
- 定期监控关键kernel的占用率
- 结合profiler进行深度分析

#### 7.2 最佳实践总结
```
1. 开发阶段：
   - 使用API获取推荐配置作为起点
   - 进行性能基准测试验证
   - 分析占用率限制因素

2. 优化阶段：
   - 针对瓶颈进行专项优化
   - 测试多种配置找到最优解
   - 考虑不同数据规模的适应性

3. 部署阶段：
   - 建立监控体系
   - 定期性能回归测试
   - 适配新硬件架构
```

---

## 相关笔记
<!-- 自动生成 -->

暂无相关笔记

