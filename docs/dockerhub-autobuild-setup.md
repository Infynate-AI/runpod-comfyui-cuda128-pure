# Docker Hub 自动构建配置指南

> **⚠️ 重要提示（2024年更新）**：
> 
> 根据新版 Docker Hub 界面（2024-2025），**Docker Hub 的自动构建功能可能已被移除或不再对免费用户开放**。在新版界面中，创建仓库时可能找不到 "Create Automated Build" 选项，仓库设置中也可能没有 "Builds" 标签页。
> 
> **推荐方案**：使用 **GitHub Actions** 进行自动构建和推送，这是目前最可靠和灵活的方案。详见 [GitHub Actions 自动化构建配置指南](dockerhub-setup.md)。

## 如果 Docker Hub 自动构建仍可用

如果你在 Docker Hub 界面中能够找到自动构建相关选项，可以按照以下步骤配置：

## 前置条件

- Docker Hub 账号（如：`robinl9527`）
- GitHub 仓库已公开（或 Docker Hub 已授权访问私有仓库）

## 重要提示：自动构建 vs 普通仓库

Docker Hub 有两种类型的仓库：
- **普通仓库**：只能手动推送镜像，没有自动构建功能
- **自动构建仓库**：连接 GitHub/Bitbucket 后可以自动构建

**如果你已经创建了普通仓库**（只有 Overview 和 Tags 标签，没有 Builds 标签），你有两个选择：

1. **推荐**：删除现有普通仓库，重新创建自动构建仓库（见下方步骤）
2. **备选**：保留现有仓库，使用 GitHub Actions 或本地构建推送

如果选择方案 1，请先删除现有仓库：
- 在仓库页面点击 **"Settings"** → 滚动到底部 → 点击 **"Delete Repository"**

## 步骤 1：创建自动构建仓库

**重要**：Docker Hub 的界面可能因版本而异。以下是几种可能的配置方式：

### 方式 A：创建时直接启用自动构建（新版界面）

1. 登录 [Docker Hub](https://hub.docker.com/)
2. 访问你的仓库列表：`https://hub.docker.com/repositories/robinl9527`
3. 点击 **"Create Repository"** 或 **"Create"** → **"Create Repository"** 按钮
4. 在创建仓库页面，填写基本信息：
   - **Repository Name**: `comfyui-cuda128`
   - **Visibility**: 选择 `Public` 或 `Private`
   - **Description**（可选）: 描述你的镜像
5. **查找自动构建选项**：
   - 在创建页面查找 **"Build Settings"**、**"Automated Builds"** 或 **"Enable Automated Builds"** 选项
   - 如果看到这些选项，勾选或启用它们
   - 选择 **"GitHub"** 作为代码源
   - 授权 Docker Hub 访问你的 GitHub 账号（如果还没有授权）
   - 选择要连接的仓库：`ultimatech-cn/runpod-comfyui-cuda128`
6. 点击 **"Create"** 创建仓库

### 方式 B：使用 "Create Automated Build" 选项（旧版界面）

1. 登录 [Docker Hub](https://hub.docker.com/)
2. 点击右上角的 **"Create"** 按钮
3. 在下拉菜单中查找并选择 **"Create Automated Build"** 或 **"Automated Build"**
4. 选择 **"GitHub"** 并授权访问
5. 选择要连接的仓库：`ultimatech-cn/runpod-comfyui-cuda128`
6. 配置仓库信息：
   - **Repository Name**: `comfyui-cuda128`
   - **Visibility**: 选择 `Public` 或 `Private`
7. 点击 **"Create"** 创建仓库

### 方式 C：在现有仓库中启用自动构建（如果支持）

1. 进入现有仓库页面：`https://hub.docker.com/r/robinl9527/comfyui-cuda128`
2. 点击 **"Settings"** 标签页
3. 查找 **"Build Settings"**、**"Automated Builds"** 或类似选项
4. 如果找到，启用并连接 GitHub 仓库

**注意**：
- 如果创建后仍然看不到 "Builds" 标签页，说明自动构建未成功启用
- 你可能需要删除仓库并按照方式 A 或 B 重新创建
- Docker Hub 的界面可能因地区、账号类型或版本而异

## 步骤 2：验证仓库创建

1. 创建完成后，访问仓库页面：`https://hub.docker.com/r/robinl9527/comfyui-cuda128`
2. 检查标签页：
   - **成功启用自动构建**：你应该能看到 **"Overview"**、**"Builds"** 和 **"Tags"** 三个标签页
   - **未启用自动构建**：你只能看到 **"Overview"** 和 **"Tags"** 两个标签页

3. 如果看到 **"Builds"** 标签页：
   - 点击 **"Builds"** 标签页
   - 你应该能看到构建配置界面和构建规则设置

4. 如果看不到 **"Builds"** 标签页：
   - 说明自动构建未成功启用
   - 请尝试以下操作：
     - 检查仓库设置中是否有自动构建选项
     - 或者删除仓库，按照上述步骤重新创建

## 步骤 3：配置构建规则（在 Builds 标签页）

在 Docker Hub 的构建配置页面，你需要设置以下构建规则：

### 规则 1：主分支构建（latest 标签）

1. 点击 **"Create Automated Build"** 或 **"Add Build Rule"**
2. 配置：
   - **Source Type**: `Branch`
   - **Source**: `main`（或你的主分支名）
   - **Docker Tag**: `latest`
   - **Dockerfile Location**: `Dockerfile`（默认）
   - **Build Context**: `/`（默认）
3. 点击 **"Create"** 或 **"Save"**

### 规则 2：版本标签构建（可选）

如果你想为 Git tags 创建版本标签：

1. 再次点击 **"Add Build Rule"**
2. 配置：
   - **Source Type**: `Tag`
   - **Source**: `v*`（匹配所有以 `v` 开头的标签，如 `v1.0.0`）
   - **Docker Tag**: `{sourceref}`（使用标签名作为 Docker tag）
   - **Dockerfile Location**: `Dockerfile`
   - **Build Context**: `/`
3. 点击 **"Create"** 或 **"Save"**

### 规则 3：每次提交构建（可选，不推荐）

如果你想要每次提交都构建（会产生很多构建）：

1. 再次点击 **"Add Build Rule"**
2. 配置：
   - **Source Type**: `Branch`
   - **Source**: `main`
   - **Docker Tag**: `{sourceref}`（使用 commit SHA 作为 tag）
   - **Dockerfile Location**: `Dockerfile`
   - **Build Context**: `/`
3. 点击 **"Create"** 或 **"Save"**

## 步骤 4：触发构建

配置完成后，你可以通过以下方式触发构建：

### 方式 1：自动触发（推荐）

- **推送代码到主分支**：当你推送代码到 `main` 分支时，Docker Hub 会自动检测并开始构建
- **创建 Git Tag**：如果你配置了版本标签规则，创建新 tag 时会自动构建

```bash
# 创建并推送版本标签示例
git tag v1.0.0
git push origin v1.0.0
```

### 方式 2：手动触发

1. 在 Docker Hub 仓库的 **"Builds"** 标签页
2. 找到对应的构建规则
3. 点击规则右侧的 **"Trigger"** 按钮
4. 选择要构建的分支或标签
5. 点击 **"Trigger Build"**

## 步骤 5：查看构建状态

1. 在 Docker Hub 仓库的 **"Builds"** 标签页
2. 你可以看到所有构建历史：
   - **绿色**：构建成功
   - **红色**：构建失败
   - **黄色**：构建中
   - **灰色**：已取消或等待中
3. 点击构建记录可以查看详细日志

## 构建优化建议

### 1. 使用 `.dockerignore` 文件

确保 `.dockerignore` 文件已正确配置，排除不必要的文件以加快构建速度：

```dockerignore
.DS_Store
venv
.env
data
models
simulated_uploaded
__pycache__
.specstory/
logs*.txt
*.log
```

### 2. 构建缓存

Docker Hub 会自动缓存构建层，如果 Dockerfile 的层顺序合理，构建会更快。

### 3. 构建时间

- 首次构建可能需要 **1-3 小时**（取决于镜像大小和网络速度）
- 后续构建如果只是小改动，通常需要 **30 分钟到 1 小时**

## 常见问题

### Q: 构建失败怎么办？

A: 查看构建日志：
1. 在 Docker Hub 的构建页面点击失败的构建记录
2. 查看详细日志，找出失败原因
3. 修复问题后重新推送代码或手动触发构建

### Q: 如何清理旧的构建？

A: Docker Hub 会自动清理旧的构建记录，但镜像会保留在仓库中。你可以在仓库的 **"Tags"** 标签页手动删除不需要的标签。

### Q: 构建速度慢怎么办？

A: 
- 检查 `.dockerignore` 是否排除了不必要的文件
- 优化 Dockerfile 的层顺序，将变化少的层放在前面
- 考虑使用多阶段构建（如果适用）

### Q: 如何设置构建触发器？

A: 在构建规则中，你可以设置：
- **Autobuild**: 自动构建（推送代码时自动触发）
- **Active**: 构建规则是否启用

### Q: 私有仓库如何配置？

A: 
1. 确保 Docker Hub 已授权访问你的 GitHub 私有仓库
2. 在连接仓库时选择私有仓库
3. Docker Hub 的自动构建支持私有仓库

### Q: 我在创建仓库时找不到自动构建选项，怎么办？

A: Docker Hub 的界面可能因版本而异。请尝试以下方法：

1. **检查是否在创建页面**：
   - 确保你点击的是 "Create Repository"，而不是其他选项
   - 在创建页面仔细查找 "Build Settings"、"Automated Builds" 或类似选项
   - 可能需要在页面底部或侧边栏

2. **尝试通过右上角菜单**：
   - 点击右上角的 "Create" 按钮
   - 查看下拉菜单中是否有 "Create Automated Build" 或 "Automated Build" 选项

3. **检查现有仓库设置**：
   - 进入现有仓库的 "Settings" 标签页
   - 查找是否有启用自动构建的选项

4. **如果以上都不行**：
   - Docker Hub 可能已经更改了自动构建的配置方式
   - 建议使用 GitHub Actions 作为替代方案（见 README.md）
   - 或者联系 Docker Hub 支持获取最新指南

### Q: Docker Hub 是否已经弃用自动构建功能？

A: **根据 2024-2025 年的新版 Docker Hub 界面**，自动构建功能似乎已经不再对普通用户开放：
- 新版界面中创建仓库时可能没有 "Create Automated Build" 选项
- 仓库设置中可能没有 "Builds" 标签页
- Docker Hub 可能已将自动构建功能整合到 **Docker Build Cloud**（付费服务）中

**推荐解决方案**：
- **使用 GitHub Actions**：这是目前最可靠和免费的方案，功能更强大且灵活
- 详细配置请参考：[GitHub Actions 自动化构建配置指南](dockerhub-setup.md)
- 如果确实需要 Docker Hub 原生的自动构建，可能需要联系 Docker 支持或考虑 Docker Build Cloud

## 验证构建成功

构建成功后，你可以通过以下方式验证：

```bash
# 拉取镜像
docker pull robinl9527/comfyui-cuda128:latest

# 查看镜像信息
docker images robinl9527/comfyui-cuda128:latest
```

## 下一步

配置完成后，你可以：
1. 在 RunPod 中使用 Docker Hub 镜像：`robinl9527/comfyui-cuda128:latest`
2. 更新 `.runpod/hub.json` 中的镜像配置（如果需要）
3. 删除或禁用 GitHub Actions 工作流（如果不再需要）

---

**注意**：Docker Hub 的免费账号有构建时间限制（每月 200 分钟），如果你的镜像很大，可能需要考虑升级到付费计划。

