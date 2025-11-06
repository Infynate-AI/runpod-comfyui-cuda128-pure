# Docker Hub 自动化构建配置指南

本文档说明如何配置 GitHub Actions 自动构建并推送镜像到 Docker Hub。

## 前置条件

1. Docker Hub 账号：`robinl9527`
2. GitHub 仓库：`ultimatech-cn/runpod-comfyui-cuda128`

## 配置步骤

### 步骤 1: 在 Docker Hub 创建仓库

1. 登录 [Docker Hub](https://hub.docker.com/)
2. 点击右上角 **"Create Repository"**
3. 填写仓库信息：
   - **Repository Name**: `comfyui-cuda128`
   - **Visibility**: Public（公开）或 Private（私有）
   - **Description**: 可选描述
4. 点击 **"Create"**

### 步骤 2: 创建 Docker Hub Access Token

1. 在 Docker Hub 右上角点击你的用户名，选择 **"Account Settings"**
2. 在左侧菜单选择 **"Security"**
3. 点击 **"New Access Token"**
4. 填写：
   - **Description**: `GitHub Actions`
   - **Access permissions**: 选择 **Read & Write**
5. 点击 **"Generate"**
6. **重要**：复制生成的 token（格式：`dckr_pat_...`），它只会显示一次

### 步骤 3: 在 GitHub 配置 Secrets

1. 进入你的 GitHub 仓库：`https://github.com/ultimatech-cn/runpod-comfyui-cuda128`
2. 点击 **Settings** > **Secrets and variables** > **Actions**
3. 点击 **"New repository secret"**，添加以下 secrets：

#### DOCKERHUB_USERNAME
- **Name**: `DOCKERHUB_USERNAME`
- **Value**: `robinl9527`

#### DOCKERHUB_TOKEN
- **Name**: `DOCKERHUB_TOKEN`
- **Value**: 粘贴步骤 2 中创建的 Access Token（`dckr_pat_...`）

### 步骤 4: 工作流触发方式

配置完成后，工作流会在以下情况自动触发：

1. **手动触发**：
   - 进入 **Actions** 标签页
   - 选择 **"Build and Push to Docker Hub"** 工作流
   - 点击 **"Run workflow"**

2. **自动触发**：
   - 推送到 `main` 分支（当 Dockerfile 或工作流文件有变化时）
   - 创建新的 Release 时

### 步骤 5: 验证构建

1. 构建完成后，可以在 GitHub Actions 页面查看构建日志
2. 在 Docker Hub 仓库页面查看推送的镜像：
   - `robinl9527/comfyui-cuda128:latest`
   - `robinl9527/comfyui-cuda128:main`（如果从 main 分支构建）

## 镜像标签说明

工作流会自动创建以下标签：

- `latest`: 从 main 分支构建时
- `main`: 从 main 分支构建时
- `<version>`: 创建 Release 时使用版本号（如 `1.0.0`）
- `<major>.<minor>`: 语义化版本的主次版本（如 `1.0`）

## 故障排查

### 构建失败

1. 检查 Secrets 是否正确配置
2. 检查 Docker Hub token 是否有效
3. 查看 GitHub Actions 日志获取详细错误信息

### 推送失败

1. 确认 Docker Hub 仓库已创建
2. 确认 token 有 **Write** 权限
3. 检查仓库名称是否匹配（`comfyui-cuda128`）

## 参考

- [GitHub Actions 文档](https://docs.github.com/en/actions)
- [Docker Hub 文档](https://docs.docker.com/docker-hub/)
- [docker/build-push-action 文档](https://github.com/docker/build-push-action)

