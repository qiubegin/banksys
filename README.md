# banksys

一个基于银行营销数据的教学展示项目，包含：
- 交互式数据分析页面
- 离线模型训练流程
- 在线认购预测页面
- 基于 GitHub Actions 的完整 CI/CD 部署链路

## 项目目标

本项目用于演示如何围绕一份银行营销数据集，完成从数据分析、模型训练、在线预测到自动化部署的完整工程闭环。

当前已经实现：
- 数据分析页面：支持多条件筛选、统计指标与图表展示
- 离线训练：支持训练模型、输出指标并保存模型产物
- 在线预测：支持表单输入客户特征，返回是否认购及对应概率
- GitHub Flow + CI/CD：支持 feature 分支开发、PR 合并、主线自动 CI、自动 CD 到服务器

## 技术栈

- Python 3.11
- Streamlit
- scikit-learn
- pytest
- ruff
- Docker
- GitHub Actions

## 项目结构

```text
banksys/
├── app/
│   ├── app.py            # Streamlit 应用入口
│   └── training.py       # 离线训练与预测逻辑
├── data/
│   ├── train.csv         # 训练数据（含 subscribe 标签）
│   └── test.csv          # 测试数据（不含标签）
├── models/               # 模型与指标产物（默认不进 Git）
├── tests/                # pytest 测试
├── .github/workflows/    # CI/CD 工作流
├── Dockerfile
├── requirements.txt
├── requirements-dev.txt
└── standards/            # 项目上下文、需求与进度记录
```

## 本地运行

### 1. 安装依赖

```bash
pip install -r requirements.txt -r requirements-dev.txt
```

### 2. 启动应用

```bash
streamlit run app/app.py --server.port=8004 --server.address=0.0.0.0
```

访问：
- http://localhost:8004

## 离线训练

运行训练命令：

```bash
python -m app.training
```

训练完成后会生成：
- `models/subscription_model.joblib`
- `models/training_metrics.json`

首版训练指标（当前实现）：
- Accuracy: `0.8756`
- Precision: `0.5824`
- Recall: `0.1797`
- F1: `0.2746`

说明：
- 训练产物默认不提交到 Git
- 在线预测页面依赖这些训练产物
- 如果页面提示“当前没有可用模型”，先执行一次训练命令即可

## 在线预测

在线预测页面会：
- 自动读取离线训练产物
- 按训练特征生成输入控件
- 点击按钮后给出：
  - 是否认购
  - 认购概率

如果模型不存在，页面会提示先执行：

```bash
python -m app.training
```

## 测试与代码质量检查

```bash
ruff format --check .
ruff check .
pytest
```

当前测试覆盖率要求：
- 核心代码覆盖率 >= 80%

## Docker 运行

### 本地构建镜像

```bash
docker build -t banksys:latest .
```

### 本地启动容器

```bash
docker run -d --name banksys --restart unless-stopped -p 8004:8004 banksys:latest
```

访问：
- http://localhost:8004

## CI/CD 说明

### CI

PR 或分支 push 时自动执行：
- Ruff format check
- Ruff lint
- Pytest
- Coverage 校验
- Docker build

### CD

合并到 `main` 后自动执行：
- 读取 GitHub Secrets
- SSH 登录服务器
- 同步项目到部署目录
- 远程 Docker build
- 启动/替换容器
- 校验 8004 端口服务可用

### 需要的 GitHub Secrets

- `SSH_PRIVATE_KEY`
- `SSH_HOST`
- `SSH_USER`

## 这个项目证明了什么

这个项目可作为一个完整的 CI/CD 展示案例，已经实际跑通：
- feature 分支开发
- Pull Request 合并
- GitHub Actions CI
- GitHub Actions CD
- 远程 Docker 部署
- 线上问题排查与修复

实际解决过的部署问题包括：
- SSH 在远程 Docker 构建阶段 `Broken pipe`
- Streamlit 容器内脚本导入路径冲突
- 训练产物缺失导致预测不可用

## 后续可继续增强

- 改进模型效果（更多模型对比、调参、特征选择）
- 将训练产物持久化到宿主机挂载目录
- 拆分分析页与预测页为更清晰的多页面结构
- 增加 README 中的系统架构图与部署流程图
