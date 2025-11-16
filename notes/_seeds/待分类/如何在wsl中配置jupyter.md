---
created: '2025-11-16'
last_reviewed: '2025-11-16'
next_review: '2025-11-16'
review_count: 0
difficulty: medium
mastery_level: 0.0
tags:
- _seeds
- _seeds/待分类
related_outlines: []
---

# WSL 如何配置 Jupyter 在本地使用

## 面试标准答案（可背诵）

在 WSL 中配置 Jupyter 供本地浏览器访问主要有两种方式：**直接访问方式**是在 WSL 中启动 Jupyter 后，复制生成的 token URL 直接在 Windows 浏览器中打开（WSL2 支持 localhost 转发）；**配置方式**是通过设置 Jupyter 配置文件，指定监听地址为 `0.0.0.0`，禁用自动打开浏览器，并可选配置密码或 token 认证。推荐使用 **Jupyter Lab** 替代传统 Notebook，体验更好。关键命令是 `jupyter lab --no-browser --port=8888`。

---

## 详细讲解

### 1. 环境准备

#### 1.1 安装 Python 和 Jupyter

```bash
# 更新包管理器
sudo apt update && sudo apt upgrade -y

# 安装 Python 和 pip（如果未安装）
sudo apt install python3 python3-pip -y

# 安装 Jupyter Lab（推荐）
pip3 install jupyterlab

# 或安装传统 Jupyter Notebook
pip3 install notebook

# 验证安装
jupyter --version
# 或
jupyter lab --version
```

#### 1.2 配置环境变量

```bash
# 将 pip 安装路径添加到 PATH
echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.bashrc
source ~/.bashrc

# 验证 jupyter 命令可用
which jupyter
```

### 2. 快速启动方式（推荐新手）

#### 2.1 最简单的启动方法

```bash
# 方法 1：直接启动（WSL2 自动支持 localhost 转发）
jupyter lab

# 方法 2：指定不自动打开浏览器
jupyter lab --no-browser

# 方法 3：指定端口
jupyter lab --no-browser --port=8888
```

**启动后的输出示例：**
```
[I 2025-11-16 10:30:00.123 ServerApp] Jupyter Server 2.x.x is running at:
[I 2025-11-16 10:30:00.123 ServerApp] http://localhost:8888/lab?token=abc123def456...
[I 2025-11-16 10:30:00.123 ServerApp]     http://127.0.0.1:8888/lab?token=abc123def456...
```

#### 2.2 在 Windows 浏览器中访问

```bash
# 1. 复制终端中显示的完整 URL（包含 token）
# 例如：http://localhost:8888/lab?token=abc123def456...

# 2. 在 Windows 浏览器中直接粘贴访问
# Chrome、Edge、Firefox 等都可以

# 3. 如果 localhost 不工作，尝试使用 127.0.0.1
# http://127.0.0.1:8888/lab?token=abc123def456...
```

### 3. 配置文件方式（推荐进阶用户）

#### 3.1 生成配置文件

```bash
# 生成 Jupyter 配置文件
jupyter lab --generate-config

# 配置文件位置
# ~/.jupyter/jupyter_lab_config.py

# 或者为 Notebook 生成配置
jupyter notebook --generate-config
# ~/.jupyter/jupyter_notebook_config.py
```

#### 3.2 编辑配置文件

```bash
# 使用你喜欢的编辑器打开配置文件
nano ~/.jupyter/jupyter_lab_config.py
# 或
vim ~/.jupyter/jupyter_lab_config.py
# 或
code ~/.jupyter/jupyter_lab_config.py  # 如果安装了 VS Code
```

**关键配置项：**

```python
# ~/.jupyter/jupyter_lab_config.py

# 1. 监听所有 IP 地址（重要！）
c.ServerApp.ip = '0.0.0.0'
# 或者只监听本地
# c.ServerApp.ip = 'localhost'

# 2. 指定端口
c.ServerApp.port = 8888

# 3. 禁止自动打开浏览器
c.ServerApp.open_browser = False

# 4. 允许远程访问（如果需要）
c.ServerApp.allow_remote_access = True

# 5. 设置工作目录
c.ServerApp.root_dir = '/home/your_username/notebooks'

# 6. 禁用 token（不推荐，安全风险）
# c.ServerApp.token = ''

# 7. 设置密码（推荐）
# 先生成密码哈希，然后配置
# c.ServerApp.password = 'argon2:...'
```

#### 3.3 设置密码（推荐）

```bash
# 方法 1：使用命令行设置密码
jupyter lab password

# 输入密码后，会自动生成配置文件
# ~/.jupyter/jupyter_server_config.json

# 方法 2：手动生成密码哈希
python3 -c "from jupyter_server.auth import passwd; print(passwd())"
# 输入密码后，复制生成的哈希值
# 例如：argon2:$argon2id$v=19$m=10240,t=10,p=8$...

# 将哈希值添加到配置文件
# c.ServerApp.password = 'argon2:$argon2id$v=19$m=10240,t=10,p=8$...'
```

### 4. 完整配置示例

#### 4.1 推荐配置（安全 + 便捷）

```python
# ~/.jupyter/jupyter_lab_config.py

# ============ 基础配置 ============
# 监听地址
c.ServerApp.ip = '0.0.0.0'

# 端口
c.ServerApp.port = 8888

# 不自动打开浏览器
c.ServerApp.open_browser = False

# 允许远程访问
c.ServerApp.allow_remote_access = True

# ============ 安全配置 ============
# 使用密码（推荐）
# 运行 jupyter lab password 生成
c.ServerApp.password = 'argon2:$argon2id$v=19$m=10240,t=10,p=8$...'

# 或使用 token（默认）
# c.ServerApp.token = 'your-custom-token'

# ============ 路径配置 ============
# 工作目录
c.ServerApp.root_dir = '/home/your_username/projects'

# ============ 性能配置 ============
# 最大缓冲区大小（MB）
c.ServerApp.max_buffer_size = 536870912  # 512MB

# ============ 日志配置 ============
# 日志级别
c.ServerApp.log_level = 'INFO'
```

#### 4.2 启动脚本

```bash
# 创建启动脚本
nano ~/start_jupyter.sh

# 添加以下内容：
#!/bin/bash
cd ~/projects  # 切换到工作目录
jupyter lab --no-browser --port=8888

# 赋予执行权限
chmod +x ~/start_jupyter.sh

# 使用脚本启动
~/start_jupyter.sh
```

### 5. WSL 网络配置

#### 5.1 WSL2 网络特性

```bash
# WSL2 使用虚拟网络，但支持 localhost 转发
# Windows 10 版本 18945 及以上自动支持

# 查看 WSL IP 地址
ip addr show eth0

# 查看 Windows 主机 IP（从 WSL 访问）
cat /etc/resolv.conf | grep nameserver | awk '{print $2}'
```

#### 5.2 防火墙配置（如果无法访问）

```powershell
# 在 Windows PowerShell（管理员）中运行

# 允许 WSL 端口通过防火墙
New-NetFirewallRule -DisplayName "WSL Jupyter" -Direction Inbound -LocalPort 8888 -Protocol TCP -Action Allow

# 或者临时关闭防火墙测试（不推荐）
# Set-NetFirewallProfile -Profile Domain,Public,Private -Enabled False
```

#### 5.3 端口转发（WSL1 或特殊情况）

```powershell
# 如果使用 WSL1 或 localhost 不工作，需要端口转发
# 在 Windows PowerShell（管理员）中运行

# 获取 WSL IP
wsl hostname -I

# 设置端口转发（假设 WSL IP 是 172.x.x.x）
netsh interface portproxy add v4tov4 listenport=8888 listenaddress=0.0.0.0 connectport=8888 connectaddress=172.x.x.x

# 查看端口转发规则
netsh interface portproxy show all

# 删除端口转发
netsh interface portproxy delete v4tov4 listenport=8888 listenaddress=0.0.0.0
```

### 6. 高级配置

#### 6.1 使用虚拟环境

```bash
# 创建虚拟环境
python3 -m venv ~/jupyter_env

# 激活虚拟环境
source ~/jupyter_env/bin/activate

# 在虚拟环境中安装 Jupyter
pip install jupyterlab

# 安装常用包
pip install numpy pandas matplotlib scikit-learn

# 启动 Jupyter
jupyter lab --no-browser
```

#### 6.2 配置多个 Kernel

```bash
# 安装 ipykernel
pip install ipykernel

# 将当前环境添加为 kernel
python -m ipykernel install --user --name=my_env --display-name "Python (my_env)"

# 查看已安装的 kernel
jupyter kernelspec list

# 删除 kernel
jupyter kernelspec uninstall my_env
```

#### 6.3 使用 Conda 环境

```bash
# 安装 Miniconda（如果未安装）
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh

# 创建 conda 环境
conda create -n jupyter_env python=3.10

# 激活环境
conda activate jupyter_env

# 安装 Jupyter
conda install jupyterlab

# 安装数据科学包
conda install numpy pandas matplotlib scikit-learn

# 启动
jupyter lab --no-browser
```

#### 6.4 后台运行 Jupyter

```bash
# 方法 1：使用 nohup
nohup jupyter lab --no-browser --port=8888 > ~/jupyter.log 2>&1 &

# 查看进程
ps aux | grep jupyter

# 停止进程
kill <PID>

# 方法 2：使用 tmux（推荐）
# 安装 tmux
sudo apt install tmux

# 创建新会话
tmux new -s jupyter

# 启动 Jupyter
jupyter lab --no-browser

# 分离会话：按 Ctrl+B，然后按 D

# 重新连接
tmux attach -t jupyter

# 方法 3：使用 systemd 服务
# 创建服务文件
sudo nano /etc/systemd/system/jupyter.service
```

**systemd 服务配置：**

```ini
[Unit]
Description=Jupyter Lab
After=network.target

[Service]
Type=simple
User=your_username
WorkingDirectory=/home/your_username
ExecStart=/home/your_username/.local/bin/jupyter lab --no-browser --port=8888
Restart=on-failure

[Install]
WantedBy=multi-user.target
```

```bash
# 启用并启动服务
sudo systemctl enable jupyter
sudo systemctl start jupyter

# 查看状态
sudo systemctl status jupyter

# 查看日志
sudo journalctl -u jupyter -f
```

### 7. 常见问题排查

#### 7.1 无法访问 localhost:8888

```bash
# 问题 1：端口被占用
# 检查端口占用
netstat -tuln | grep 8888
# 或
lsof -i :8888

# 更换端口
jupyter lab --no-browser --port=8889

# 问题 2：防火墙阻止
# 参考 5.2 节配置防火墙

# 问题 3：WSL 网络问题
# 重启 WSL
wsl --shutdown
# 然后重新打开 WSL
```

#### 7.2 Token 或密码问题

```bash
# 重置 token
jupyter lab --no-browser --ServerApp.token=''

# 重置密码
rm ~/.jupyter/jupyter_server_config.json
jupyter lab password

# 查看当前 token
jupyter lab list
```

#### 7.3 Kernel 连接问题

```bash
# 重新安装 ipykernel
pip install --upgrade ipykernel

# 清除 Jupyter 缓存
jupyter lab clean

# 重启 Jupyter
```

### 8. 最佳实践

#### 8.1 推荐工作流

```bash
# 1. 创建项目目录结构
mkdir -p ~/projects/notebooks
mkdir -p ~/projects/data
mkdir -p ~/projects/scripts

# 2. 使用虚拟环境
python3 -m venv ~/projects/.venv
source ~/projects/.venv/bin/activate

# 3. 安装依赖
pip install jupyterlab numpy pandas matplotlib

# 4. 配置 Jupyter
jupyter lab --generate-config
# 编辑配置文件设置工作目录

# 5. 启动
cd ~/projects
jupyter lab --no-browser
```

#### 8.2 安全建议

```bash
# 1. 始终使用密码或 token
jupyter lab password

# 2. 不要在公网暴露 Jupyter
# 只监听 localhost 或使用 VPN

# 3. 定期更新
pip install --upgrade jupyterlab

# 4. 使用 HTTPS（生产环境）
# 生成自签名证书
openssl req -x509 -nodes -days 365 -newkey rsa:2048 \
  -keyout ~/.jupyter/mykey.key -out ~/.jupyter/mycert.pem

# 配置文件中添加
# c.ServerApp.certfile = '/home/username/.jupyter/mycert.pem'
# c.ServerApp.keyfile = '/home/username/.jupyter/mykey.key'
```

#### 8.3 性能优化

```python
# 配置文件优化
# ~/.jupyter/jupyter_lab_config.py

# 增加最大缓冲区
c.ServerApp.max_buffer_size = 1073741824  # 1GB

# 禁用不需要的扩展
c.LabApp.disabled_extensions = ['@jupyterlab/some-extension']

# 配置资源限制
c.MappingKernelManager.cull_idle_timeout = 3600  # 1小时后关闭空闲 kernel
c.MappingKernelManager.cull_interval = 300  # 每5分钟检查一次
```

### 9. VS Code 集成（额外福利）

```bash
# 在 WSL 中安装 VS Code Server
# Windows 中安装 VS Code 和 Remote-WSL 扩展

# 在 WSL 中打开项目
code ~/projects

# VS Code 会自动检测 Jupyter Notebook
# 无需单独启动 Jupyter Server
```

---

## 总结

### 核心要点

1. **快速启动**：
   ```bash
   jupyter lab --no-browser --port=8888
   ```
   复制 token URL 在 Windows 浏览器中打开

2. **配置文件**：
   - 生成：`jupyter lab --generate-config`
   - 位置：`~/.jupyter/jupyter_lab_config.py`
   - 关键配置：`ip='0.0.0.0'`, `open_browser=False`

3. **安全设置**：
   - 使用密码：`jupyter lab password`
   - 或保留 token 认证（默认）

4. **网络访问**：
   - WSL2 自动支持 `localhost:8888`
   - 如有问题检查防火墙和端口转发

5. **后台运行**：
   - 使用 `tmux` 或 `nohup`
   - 或配置 systemd 服务

6. **最佳实践**：
   - 使用虚拟环境隔离依赖
   - 配置工作目录
   - 定期更新和备份

---

## 参考文献

1. **Jupyter 官方文档**
   - https://jupyter-notebook.readthedocs.io/en/stable/
   - Jupyter Notebook 完整配置指南

2. **JupyterLab 文档**
   - https://jupyterlab.readthedocs.io/en/stable/
   - JupyterLab 使用和配置文档

3. **WSL 官方文档**
   - https://docs.microsoft.com/en-us/windows/wsl/
   - WSL 网络和配置说明

4. **Jupyter Server 配置**
   - https://jupyter-server.readthedocs.io/en/latest/users/configuration.html
   - 服务器配置详细说明

5. **WSL Networking**
   - https://docs.microsoft.com/en-us/windows/wsl/networking
   - WSL 网络配置和故障排除

6. **Real Python - Jupyter Notebook Tutorial**
   - https://realpython.com/jupyter-notebook-introduction/
   - Jupyter 使用教程和最佳实践