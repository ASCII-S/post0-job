---
created: '2025-10-19'
last_reviewed: null
next_review: '2025-10-19'
review_count: 0
difficulty: medium
mastery_level: 0.0
tags:
- 熟悉大语言模型推理优化-技术层次
- 熟悉大语言模型推理优化-技术层次/灰度发布和A_B测试如何实现？.md
related_outlines: []
---
# 灰度发布和A/B测试如何实现？

## 面试标准答案

灰度发布通过逐步扩大新版本流量比例来降低风险，典型流程是5%→25%→50%→100%，每阶段监控指标决定是否继续或回滚。A/B测试同时运行两个版本，将流量随机分配并对比效果。实现方式：1)基于路由层控制流量分配比例；2)为每个副本打版本标签；3)记录请求和版本的映射关系；4)收集并分析各版本的性能指标；5)根据统计显著性决定采用哪个版本。关键是流量控制、标签追踪和指标对比。

---

## 详细讲解

### 1. 灰度发布实现

```python
class GradualRollout:
    def __init__(self, replicas):
        self.replicas = replicas
        self.stages = [
            {'ratio': 0.05, 'duration': 1800},  # 5%, 30分钟
            {'ratio': 0.25, 'duration': 1800},  # 25%, 30分钟
            {'ratio': 0.50, 'duration': 3600},  # 50%, 1小时
            {'ratio': 1.00, 'duration': 0},     # 100%
        ]
    
    def gradual_rollout(self, new_model):
        updated_count = 0
        total = len(self.replicas)
        
        for stage in self.stages:
            # 计算需要更新的副本数
            target_count = int(total * stage['ratio'])
            
            # 更新差额
            while updated_count < target_count:
                self.replicas[updated_count].load_model(new_model)
                updated_count += 1
            
            # 监控
            logger.info(f"Rollout to {stage['ratio']*100}%")
            
            if not self.monitor_stage(stage['duration']):
                # 问题检测，回滚
                self.rollback(updated_count)
                return False
        
        return True
    
    def monitor_stage(self, duration):
        start = time.time()
        while time.time() - start < duration:
            metrics = self.collect_metrics()
            if not self.validate_metrics(metrics):
                return False
            time.sleep(10)
        return True
```

### 2. A/B测试实现

```python
class ABTester:
    def __init__(self, replicas):
        # 分配副本到A/B组
        mid = len(replicas) // 2
        self.group_a = replicas[:mid]
        self.group_b = replicas[mid:]
        
        # 加载不同模型
        for r in self.group_a:
            r.load_model(model_a_path)
            r.group = 'A'
        
        for r in self.group_b:
            r.load_model(model_b_path)
            r.group = 'B'
        
        self.results = {'A': [], 'B': []}
    
    def route_request(self, request):
        # 随机分配到A或B
        if random.random() < 0.5:
            replicas = self.group_a
            group = 'A'
        else:
            replicas = self.group_b
            group = 'B'
        
        # 选择具体副本
        replica = self.select_replica(replicas)
        
        # 执行并记录
        start = time.time()
        result = replica.infer(request)
        latency = time.time() - start
        
        # 记录结果
        self.results[group].append({
            'latency': latency,
            'success': result.success,
            'user_id': request.user_id,
            'timestamp': time.time()
        })
        
        return result
    
    def analyze_results(self):
        # 统计分析
        stats_a = self.compute_stats(self.results['A'])
        stats_b = self.compute_stats(self.results['B'])
        
        # T检验
        t_stat, p_value = stats.ttest_ind(
            [r['latency'] for r in self.results['A']],
            [r['latency'] for r in self.results['B']]
        )
        
        return {
            'group_a': stats_a,
            'group_b': stats_b,
            'p_value': p_value,
            'significant': p_value < 0.05,
            'winner': 'A' if stats_a['latency'] < stats_b['latency'] else 'B'
        }
```

### 3. 流量控制

```python
class TrafficController:
    def __init__(self):
        self.traffic_split = {'A': 0.5, 'B': 0.5}
        self.version_replicas = {'A': [], 'B': []}
    
    def set_traffic_split(self, split):
        """split: {'A': 0.3, 'B': 0.7}"""
        assert sum(split.values()) == 1.0
        self.traffic_split = split
    
    def route(self, request):
        # 根据流量比例选择组
        r = random.random()
        cumsum = 0
        
        for version, ratio in self.traffic_split.items():
            cumsum += ratio
            if r < cumsum:
                selected_version = version
                break
        
        # 在选定组内负载均衡
        replicas = self.version_replicas[selected_version]
        return self.select_replica(replicas)
```

### 4. 指标收集

```python
class MetricsCollector:
    def __init__(self):
        self.metrics = defaultdict(lambda: defaultdict(list))
    
    def record(self, version, metric_name, value):
        self.metrics[version][metric_name].append(value)
    
    def get_summary(self, version):
        version_metrics = self.metrics[version]
        
        summary = {}
        for metric, values in version_metrics.items():
            summary[metric] = {
                'mean': np.mean(values),
                'median': np.median(values),
                'p50': np.percentile(values, 50),
                'p95': np.percentile(values, 95),
                'p99': np.percentile(values, 99),
                'count': len(values)
            }
        
        return summary
    
    def compare_versions(self, version_a, version_b):
        summary_a = self.get_summary(version_a)
        summary_b = self.get_summary(version_b)
        
        comparison = {}
        for metric in summary_a.keys():
            a_val = summary_a[metric]['mean']
            b_val = summary_b[metric]['mean']
            
            comparison[metric] = {
                f'{version_a}_mean': a_val,
                f'{version_b}_mean': b_val,
                'diff_pct': (b_val - a_val) / a_val * 100,
                'better': version_a if a_val < b_val else version_b
            }
        
        return comparison
```

### 5. 统计显著性检验

```python
def statistical_test(results_a, results_b, metric='latency'):
    # 提取数据
    data_a = [r[metric] for r in results_a]
    data_b = [r[metric] for r in results_b]
    
    # T检验
    t_stat, p_value = stats.ttest_ind(data_a, data_b)
    
    # 效应量(Cohen's d)
    mean_a, std_a = np.mean(data_a), np.std(data_a)
    mean_b, std_b = np.mean(data_b), np.std(data_b)
    pooled_std = np.sqrt((std_a**2 + std_b**2) / 2)
    cohens_d = (mean_b - mean_a) / pooled_std
    
    # 置信区间
    ci = stats.t.interval(
        0.95,
        len(data_a) + len(data_b) - 2,
        loc=mean_b - mean_a,
        scale=pooled_std * np.sqrt(1/len(data_a) + 1/len(data_b))
    )
    
    return {
        'p_value': p_value,
        'significant': p_value < 0.05,
        'cohens_d': cohens_d,
        'effect_size': 'large' if abs(cohens_d) > 0.8 else 'medium' if abs(cohens_d) > 0.5 else 'small',
        'confidence_interval_95': ci,
        'mean_difference': mean_b - mean_a
    }
```

### 6. 自动决策

```python
class AutoDecisionMaker:
    def __init__(self, min_samples=1000):
        self.min_samples = min_samples
    
    def should_promote(self, ab_results):
        # 检查样本量
        if len(ab_results['A']) < self.min_samples:
            return None, "Insufficient samples"
        
        # 统计检验
        test = statistical_test(ab_results['A'], ab_results['B'])
        
        if not test['significant']:
            return None, "No significant difference"
        
        # 检查改进方向
        if test['mean_difference'] > 0:  # B更差
            return 'A', "B shows regression"
        else:  # B更好
            # 检查改进幅度
            improvement = abs(test['mean_difference']) / np.mean([r['latency'] for r in ab_results['A']])
            
            if improvement > 0.05:  # 5%改进
                return 'B', f"B shows {improvement*100:.1f}% improvement"
            else:
                return None, "Improvement too small"
```

### 7. 完整灰度流程

```python
class ProductionGradualRollout:
    def execute(self, new_model):
        # 阶段0: Staging验证
        if not self.validate_in_staging(new_model):
            return False
        
        # 阶段1: 单副本金丝雀(5%)
        canary_results = self.canary_test(new_model, ratio=0.05, duration=1800)
        if not canary_results['pass']:
            return False
        
        # 阶段2: A/B测试(25%)
        ab_results = self.ab_test(new_model, ratio=0.25, duration=3600)
        decision = self.auto_decide(ab_results)
        
        if decision != 'promote':
            self.rollback()
            return False
        
        # 阶段3: 灰度扩大(50%)
        if not self.gradual_expand(new_model, ratio=0.50, duration=3600):
            self.rollback()
            return False
        
        # 阶段4: 全量(100%)
        self.full_rollout(new_model)
        
        logger.info("Gradual rollout completed successfully")
        return True
```

### 8. 监控面板

```python
# 实时监控指标
dashboard_metrics = {
    'traffic_split': {'A': 0.25, 'B': 0.75},
    
    'latency': {
        'A': {'p50': 120, 'p95': 250, 'p99': 400},
        'B': {'p50': 110, 'p95': 230, 'p99': 380},  # B更好
    },
    
    'error_rate': {
        'A': 0.01,
        'B': 0.008,  # B更好
    },
    
    'qps': {
        'A': 25,
        'B': 75,
    },
    
    'user_satisfaction': {
        'A': 4.2,
        'B': 4.4,  # B更好
    }
}
```

灰度发布和A/B测试是安全部署新模型的最佳实践，可显著降低上线风险。


---

## 相关笔记
<!-- 自动生成 -->

暂无相关笔记

