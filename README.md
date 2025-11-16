# 📚 知识库模板快速开始

欢迎使用终身知识库模板！本指南帮助新用户在 5 分钟内完成环境准备、了解项目结构，并按照“大纲 → 知识点”流程启动个人知识库。

---

## 🎯 模板目标

- 开箱即用的自动化脚本、配置与模板全部集中在 `system/`
- 通过 `outlines/` 管理主题大纲，与 `notes/` 中的知识点一一对应
- 借助 `system/start.sh`、`system/end.sh` 与 Git 自动化脚本记录每日进展
- 支持用户在 `config/` 中覆盖默认配置，无需修改模板源码

---

## 🧭 仓库结构速览

```
.
├── system/                 # 模板核心系统（可随模板更新）
│   ├── init.sh             # 首次使用时的一键初始化
│   ├── start.sh / end.sh   # 日常开始 / 结束脚本
│   ├── scripts/            # Python 自动化工具
│   ├── config/             # 默认配置（可被用户覆盖）
│   ├── templates/          # Markdown 模板
│   └── docs/               # 文档中心（本页所在地）
├── outlines/               # 个人主题大纲（用户创建）
├── notes/                  # 知识点笔记（用户创建）
├── examples/               # 示例大纲与笔记（可选）
├── config/                 # 用户自定义配置（覆盖 system/config）
├── reviewsToday.md             # 自动生成的复习任务清单
└── README.md               # 仓库首页说明
```

---

## ⚡ 五步完成首次配置

1. **克隆模板**
   ```bash
   git clone https://github.com/<your-account>/knowledge-base-template.git my-lib
   cd my-lib
   ```
2. **运行初始化脚本**
   ```bash
   ./system/init.sh
   ```
   - 自动检测依赖
   - 创建 `notes/`、`outlines/`、`reviewsArchived/` 等目录
   - 复制默认配置到 `config/kb_config.yaml`
3. **阅读核心文档**
   - `system/docs/INSTALLATION.md`：环境准备与依赖
   - `system/docs/USER_GUIDE.md`：日常工作流与自动化脚本
   - `system/docs/CUSTOMIZATION.md`：配置项与高级玩法
4. **创建第一个主题**
   - 在 `outlines/` 下新建主题文件，例如 `outlines/AI面试路线.md`
   - 按约定结构整理子主题与知识点链接
5. **准备配套知识点笔记**
   - 在 `notes/<主题名称>/` 下创建笔记，例如 `notes/AI面试路线/Transformer注意力机制.md`
   - 使用 `system/templates/note_template.md` 初始化内容与元数据

---

## 🔁 推荐工作流

- **每日开始**：运行 `./system/start.sh` 生成reviewsToday清单、同步远程
- **学习记录**：依据大纲更新知识点笔记，可使用脚本保持元数据一致
- **每日结束**：运行 `./system/end.sh` 汇总复习结果并执行 Git 提交
- **周期复盘**：使用图谱与统计脚本了解整体进度（详见用户指南）

---

## ⚙️ 配置覆盖机制

- 模板默认配置：`system/config/kb_config.yaml`
- 用户覆盖配置：`config/kb_config.yaml`
- 加载策略：先读模板默认值，再应用用户配置覆盖

详细字段说明与示例请查阅 `system/docs/CUSTOMIZATION.md`。

---

## 📚 延伸阅读

- `system/docs/INSTALLATION.md` — 系统要求、依赖与初始化细节
- `system/docs/USER_GUIDE.md` — 大纲与笔记协同、脚本命令、每日节奏
- `system/docs/CUSTOMIZATION.md` — 配置项说明、脚本参数与高级玩法
- `system/docs/DEVELOPMENT.md` — 希望二次开发或贡献代码的必读说明

欢迎将本模板分享给更多同样坚持终身学习的伙伴，并在你的仓库 `README.md` 中记录使用心得！

