# 🚀 快速开始指南

## 📚 知识库模板化已完成！

您的知识库现在采用清晰的模板化结构，系统和内容完全分离。

## 💡 日常使用

```bash
# 早上：生成复习清单
./start.sh

# 复习：打开 reviewsToday.md，边复习边打勾 [x]

# 晚上：同步进度并提交
./end.sh
```


## 🌱 种子笔记

遇到零散知识点？使用**种子笔记**快速记录！

```bash
# 创建种子笔记（碎片知识）
./kb seed

# 特点：
# ✅ 快速记录，不纠结分类
# ✅ 按潜在主题组织
# ✅ 积累后升级为正式主题
```

**详见**：[种子笔记指南](SEEDS_GUIDE.md) 或 `notes/_seeds/README.md`

## ⚙️ 自定义配置

编辑 `config/kb_config.yaml` 来自定义设置：

```yaml
# 调整每日复习数量
daily_review:
  max_today: 30         # 改为30篇（默认20）
  
# 调整排序权重
  sort_weights:
    difficulty_easy: 4.0   # 更强调难度
```

## 📂 目录结构

```
├── system/              # 脚本系统（不要修改）
│   ├── scripts/        # 所有功能脚本
│   ├── config/         # 默认配置
│   ├── docs/           # 文档
│   └── init.sh         # 初始化脚本
│
├── notes/              # 您的笔记
├── outlines/            # 您的大纲
├── config/             # 您的配置（覆盖默认）
│
├── start.sh            # 快捷启动
├── end.sh              # 快捷结束
```

## 🎁 分享给他人

如果要分享给其他人使用：

1. **分享整个项目**（推荐）
   ```bash
   # 其他人克隆后直接运行
   ./system/init.sh    # 初始化
   ./start.sh          # 开始使用
   ```

2. **只分享系统部分**
   ```bash
   # 复制system目录到新项目
   cp -r system /path/to/new-project/
   ```

---

**开始使用吧！** 🎉

一切都已准备就绪，您可以：
- ✅ 继续日常使用（体验完全相同）
- ✅ 自定义配置（在config/中）
- ✅ 分享给他人（system/可独立使用）

