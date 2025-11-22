---
created: '2025-11-23'
last_reviewed: '2025-11-23'
next_review: '2025-11-23'
review_count: 0
difficulty: medium
mastery_level: 0.0
tags:
- _seeds
- _seeds/待分类
related_outlines: []
---

# 如何使用ZCF配置Claude Code

## 面试标准答案（可背诵）

**ZCF (Zero-Config Code Flow)** 是一个一键配置Claude Code开发环境的CLI工具。通过 `npx zcf i` 命令可以自动完成Claude Code的安装、API密钥配置、工作流导入和MCP服务配置。ZCF支持多平台(Windows/macOS/Linux)，提供交互式菜单和非交互式CI/CD模式，并自动创建 `~/.claude/` 目录结构存储配置文件、agents和自定义命令。

---

## 详细讲解

### 1. ZCF概述

ZCF (Zero-Config Code Flow) 是一个综合性的命令行工具，旨在简化AI编程环境的设置、配置和管理。其核心设计理念是"零配置"，让开发者能够开箱即用地进入AI辅助编程。

**主要特性：**
- 一键完成Claude Code安装和配置
- 支持双代码工具架构（Claude Code和Codex）
- 智能配置管理和多语言支持
- 集成CCR（Claude Code Router）、MCP服务等外部工具
- 自动创建带时间戳的配置备份

### 2. 安装和基本使用

#### 2.1 快速初始化

```bash
# 完整初始化（推荐首次使用）
npx zcf i

# 或通过交互式菜单
npx zcf
# 然后选择 1
```

完整初始化包含：
- 自动检测并安装Claude Code
- 配置API密钥
- 导入预设AI工作流
- 配置MCP服务

#### 2.2 常用命令

```bash
# 仅更新工作流
npx zcf u

# 检查并更新Claude Code
npx zcf check

# 指定语言（中文）
npx zcf --lang zh-CN
npx zcf init --lang zh-CN
```

### 3. 交互式菜单选项

运行 `npx zcf` 后，会显示以下菜单选项：

| 选项 | 功能描述 |
|------|----------|
| 1 | **完整初始化** - 安装Claude Code + 导入工作流 + 配置API/CCR + 配置MCP |
| 2 | **导入工作流** - 仅导入/更新工作流相关文件 |
| 3 | **配置API** - 配置API URL和认证（支持CCR代理） |
| 4 | **配置MCP** - 配置MCP服务（包含Windows修复） |
| 5 | **配置默认模型** - 设置默认模型（opus/sonnet等） |
| 6 | **配置全局记忆** - 配置AI输出语言和输出风格 |
| 7 | **导入推荐环境变量和权限** |
| R | **CCR管理** - Claude Code Router管理 |
| U | **ccusage** - Claude Code使用量分析 |

### 4. 配置文件结构

ZCF会在 `~/.claude/` 目录下创建以下结构：

```
~/.claude/
├── CLAUDE.md          # Claude Code的系统提示词
├── settings.json      # API密钥、模型、权限配置
├── agents/            # 预置的AI代理（planner、ui-ux-designer等）
├── commands/          # 自定义命令（如/feat、/workflow）
└── backup/            # 每次更新自动创建的时间戳备份
```

### 5. 工作流详解

ZCF中的**工作流(Workflow)**是指预配置的Claude Code开发流程模板和自定义命令集，用于将AI辅助开发**结构化**，让Claude Code按照预设流程有序地完成复杂项目。

#### 5.1 自定义命令 (Commands)

安装到 `~/.claude/commands/` 目录，可通过 `/命令名` 调用：

| 命令 | 功能 |
|------|------|
| `/feat` | 功能规划 |
| `/workflow` | 工作流管理 |
| `/commit` | Git提交 |
| `/rollback` | Git回滚 |
| `/init-project` | 项目初始化 |
| `/init-architect` | 架构初始化 |

#### 5.2 AI代理 (Agents)

安装到 `~/.claude/agents/` 目录，提供专业领域能力：

- **planner** - 项目规划代理，负责任务拆分和进度管理
- **ui-ux-designer** - UI/UX设计代理，负责界面设计建议

#### 5.3 可选工作流模板

| 工作流 | 功能描述 |
|--------|----------|
| **Common Tools** | 基础工具集：init-project + init-architect + get-current-datetime |
| **Six Steps Workflow** | 完整6阶段开发流程 |
| **Feature Planning** | 功能规划套件：feat + planner + ui-ux-designer |
| **Git Commands** | Git操作集：commit, rollback, cleanBranches, worktree |
| **BMad-Method** | 敏捷AI驱动开发方法论，包含19个专业代理（PM、架构师、开发者、QA等） |
| **Spec Workflow** | 规范驱动开发流程：需求分析→架构设计→任务拆分→代码实现 |

#### 5.4 BMad-Method简介

BMad (Breakthrough Method for Agile AI-Driven Development) 是一个企业级工作流系统：

- **19个专业代理**：产品经理、架构师、开发者、UX设计师、QA等
- **完整开发生命周期**：分析→规划→架构→实现→测试
- **规模自适应智能**：自动调整规划深度
- **使用方式**：在Claude Code中输入 `/agent-name` 激活对应代理

#### 5.5 工作流的核心价值

工作流将AI辅助开发从"问答模式"升级为"流程模式"：

```
传统模式：用户提问 → AI回答 → 用户再提问 → ...

工作流模式：
需求分析 → 架构设计 → 任务拆分 → 代码实现 → 测试验证
   ↑          ↑          ↑          ↑          ↑
 专业代理   专业代理   专业代理   专业代理   专业代理
```

### 6. MCP服务配置

#### 6.1 什么是MCP？

**MCP (Model Context Protocol)** 是Claude Code的扩展协议，允许Claude Code连接外部工具和服务，扩展其能力范围。通过MCP，Claude Code可以：
- 访问实时网络信息
- 查询最新的技术文档
- 控制浏览器进行自动化操作
- 与外部知识库交互

#### 6.2 ZCF提供的MCP服务

| 服务 | 功能 | 典型使用场景 | 是否需要API密钥 |
|------|------|-------------|----------------|
| **Context7** | 文档查询 | 查询库/框架的最新文档（如React、Vue的API） | 否 |
| **DeepWiki** | 知识库服务 | 查询GitHub仓库的深度文档和代码解析 | 否 |
| **Playwright** | 浏览器自动化 | 网页截图、表单填写、E2E测试 | 否 |
| **Exa AI Search** | AI智能搜索 | 语义化搜索，理解意图的搜索结果 | **是** |
| **Open Web Search** | 多引擎搜索 | 普通网页搜索（DuckDuckGo/Bing/Brave） | 否 |

#### 6.3 如何配置MCP服务

```bash
# 方式1：在完整初始化时配置
npx zcf i
# 交互式选择需要的MCP服务

# 方式2：单独配置MCP
npx zcf
# 选择 4 - 配置MCP
```

#### 6.4 配置后的使用方式

配置完成后，MCP服务会自动注入到Claude Code中。你可以直接在对话中使用：

```bash
# Context7 示例：查询最新文档
"请用Context7查询React 19的新hooks文档"

# Playwright 示例：浏览器操作
"用Playwright截取 example.com 的首页截图"

# Open Web Search 示例：网页搜索
"搜索2024年最新的TypeScript最佳实践"
```

#### 6.5 配置存储位置

MCP配置存储在 `~/.claude/settings.json` 文件中，可以手动开关：

```json
{
  "mcpServers": {
    "context7": { "command": "npx", "args": ["@anthropic-ai/context7-mcp"] },
    "open-websearch": { "command": "npx", "args": ["open-websearch-mcp"] }
  }
}
```

**推荐配置**：个人用户建议启用免费的MCP服务（Context7、Open Web Search、DeepWiki），可以显著提升Claude Code的信息获取能力。

### 7. CCR（Claude Code Router）配置

#### 7.1 什么是CCR？

**CCR (Claude Code Router)** 是一个API代理路由器，位于Claude Code和API服务之间，起到"智能调度"的作用：

```
传统模式：Claude Code → Anthropic API (所有请求都用高价模型)

CCR模式：Claude Code → CCR路由器 → 根据请求类型智能选择模型
                                  ├── 简单任务 → 便宜模型
                                  └── 复杂任务 → 高端模型
```

#### 7.2 CCR的核心优势

- **成本优化**：根据任务复杂度自动选择合适的模型，简单任务不浪费昂贵的算力
- **透明代理**：对Claude Code来说是透明的，无需修改工作流程
- **使用量追踪**：可以更好地监控和分析API使用情况
- **保持效率**：复杂任务仍使用高端模型，不影响开发体验

#### 7.3 如何配置CCR

```bash
# 通过ZCF菜单配置
npx zcf
# 选择 R 进入CCR管理

# 或者在完整初始化时选择配置CCR
npx zcf i
# 在API配置环节选择使用CCR
```

#### 7.4 CCR配置流程

1. 运行 `npx zcf` 选择 `R` 进入CCR管理
2. 输入CCR服务的URL（如 `https://your-ccr-server.com`）
3. 输入CCR提供的API密钥
4. ZCF自动更新 `settings.json` 中的API端点

#### 7.5 CCR vs 直连API对比

| 维度 | 直连Anthropic API | 使用CCR |
|------|------------------|---------|
| **成本** | 所有请求统一定价 | 智能路由降低成本 |
| **配置** | 简单 | 需要额外部署/订阅CCR |
| **适用场景** | 个人轻度使用 | 团队/高频使用 |
| **延迟** | 最低 | 略有增加（经过路由） |

#### 7.6 查看使用量统计

配置CCR后，可以通过ZCF查看使用量统计：

```bash
npx zcf
# 选择 U - ccusage (Claude Code使用量分析)
```

**适用建议**：
- 个人轻度使用：直连API即可
- 团队协作/高频开发：推荐配置CCR以优化成本
- 预算敏感场景：CCR可显著降低API支出

### 8. 非交互式/CI/CD使用

在自动化环境中，可通过 `--skip-prompt` 或 `-s` 参数跳过所有交互式提问：

```bash
# 完整示例
npx zcf i -s -g zh-CN -t api_key -k "sk-xxx" -u "https://api.example.com"
```

参数说明：
- `-s` / `--skip-prompt` - 跳过交互式提问
- `-g` - 设置语言
- `-t` - 认证类型
- `-k` - API密钥
- `-u` - API URL

### 9. 平台兼容性

ZCF支持多平台：

| 平台 | 特性 |
|------|------|
| **Windows** | 标准cmd支持，正确的路径转义 |
| **WSL** | Windows子系统Linux，支持发行版检测 |
| **macOS/Linux** | 标准Unix环境 |
| **Android Termux** | 特别适配的终端模拟器支持 |

平台检测会影响MCP服务命令生成：
- Windows使用 `cmd /c npx`
- Unix系统使用 `npx`

### 10. 系统要求

- Claude Code版本需 **≥ 1.0.81**（以支持output-style功能）
- Node.js环境（用于npx命令）

---

## 总结

- **ZCF** 是Claude Code的一键配置工具，核心命令是 `npx zcf i`
- 支持**交互式菜单**和**非交互式CI/CD**两种模式
- 自动配置 `~/.claude/` 目录，包含系统提示词、API设置、agents和自定义命令
- **工作流**将AI开发结构化：包含自定义命令（/feat、/commit等）、AI代理（planner、ui-ux-designer）和方法论（BMad-Method、Spec Workflow）
- 集成**MCP服务**（搜索、浏览器控制等）和**CCR代理**（降低API成本）
- 每次配置自动创建**时间戳备份**，支持配置恢复
- **跨平台支持**：Windows、macOS、Linux、WSL、Android Termux

---

## 参考文献

1. **ZCF GitHub仓库**
   - https://github.com/UfoMiao/zcf
   - 官方源码仓库，包含完整文档和更新日志

2. **BMad-Method GitHub仓库**
   - https://github.com/bmad-code-org/BMAD-METHOD
   - 敏捷AI驱动开发方法论的官方仓库

3. **Claude Code官方文档**
   - https://docs.claude.com/en/docs/claude-code/setup
   - Claude Code的官方设置指南

4. **Claude Code MCP配置文档**
   - https://docs.claude.com/en/docs/claude-code/mcp
   - MCP服务的官方配置说明
