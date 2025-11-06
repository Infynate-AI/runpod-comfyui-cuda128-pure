# 🚀 构建时间优化指南

## 问题

原始 `Dockerfile` 在构建时需要下载大量模型文件，导致：
- **构建时间**：1.5-5 小时
- **镜像大小**：~92 GB
- **推送时间**：数小时
- **灵活性**：更新模型需要重新构建镜像

## 解决方案

使用 **Network Volume** 存储模型，镜像中只包含代码和自定义节点。

### 优化效果对比

| 指标 | 原方案 | 优化方案 | 改善 |
|------|--------|----------|------|
| 构建时间 | 1.5-5 小时 | 10-30 分钟 | **减少 80-90%** |
| 镜像大小 | ~92 GB | ~5-10 GB | **减少 85-90%** |
| 推送时间 | 数小时 | 10-30 分钟 | **减少 80-90%** |
| 模型更新 | 需重建镜像 | 直接更新 Volume | **更灵活** |

## 快速开始

### 1. 使用优化版 Dockerfile 构建

```powershell
# 使用优化版 Dockerfile（不包含模型）
docker build --platform linux/amd64 -f Dockerfile.optimized -t runpod-comfyui-cuda128:optimized .
```

### 2. 创建并配置 Network Volume

详细步骤请参考：[Network Volume 配置指南](docs/network-volume-setup.md)

**简要步骤**：
1. 在 RunPod 控制台创建 Network Volume（建议 200GB+）
2. 创建临时 Pod 并附加 Network Volume
3. 使用 `scripts/upload-models-to-volume.sh` 批量下载模型
4. 在 Endpoint 配置中附加 Network Volume

### 3. 部署优化版镜像

```powershell
# 推送到 Docker Hub
docker tag runpod-comfyui-cuda128:optimized your-username/comfyui-cuda128:optimized
docker push your-username/comfyui-cuda128:optimized
```

然后在 RunPod Endpoint 配置中：
- 选择优化版镜像
- 附加 Network Volume
- 部署

## 文件说明

### 新增文件

- **`Dockerfile.optimized`**：优化版 Dockerfile，移除了所有模型下载
- **`docs/network-volume-setup.md`**：详细的 Network Volume 配置指南
- **`scripts/upload-models-to-volume.sh`**：批量上传模型到 Network Volume 的脚本

### 原有文件

- **`Dockerfile`**：原始版本（保留，用于需要内置模型的场景）
- **`src/extra_model_paths.yaml`**：配置 ComfyUI 从 Network Volume 加载模型

## 使用场景建议

### 使用优化方案（推荐）

✅ **适合以下场景**：
- 需要频繁更新模型
- 构建时间敏感
- 多个 Endpoint 共享模型
- 模型体积很大（>50GB）
- 需要快速迭代和测试

### 使用原方案

✅ **适合以下场景**：
- 模型相对固定，不常更新
- 需要完全自包含的镜像
- 不介意较长的构建时间
- 单次部署，不需要共享模型

## 迁移指南

如果您已经使用原方案，想要迁移到优化方案：

1. **备份现有模型**（如果需要）
2. **创建 Network Volume**
3. **上传模型到 Network Volume**（使用临时 Pod）
4. **构建优化版镜像**
5. **更新 Endpoint 配置**（使用优化版镜像 + Network Volume）
6. **测试验证**

## 常见问题

### Q: 两种方案可以同时使用吗？

A: 可以。ComfyUI 会同时从两个位置加载模型：
- 镜像中的模型：`/comfyui/models/...`
- Network Volume 中的模型：`/runpod-volume/models/...`

### Q: Network Volume 的成本是多少？

A: 根据 RunPod 定价，Network Volume 按存储容量和区域计费。请查看 [RunPod 定价页面](https://www.runpod.io/pricing)。

### Q: 如何知道模型是否加载成功？

A: 检查 ComfyUI 日志或使用 `/object_info` API 端点查看可用模型列表。

### Q: 可以部分模型在镜像中，部分在 Network Volume 吗？

A: 可以。这是完全可行的，您可以根据需要灵活配置。

## 相关文档

- [Network Volume 配置指南](docs/network-volume-setup.md) - 详细配置步骤
- [RunPod Network Volumes 官方文档](https://docs.runpod.io/pods/storage/create-network-volumes)
- [RunPod Serverless 概述](https://docs.runpod.io/serverless/overview)

## 总结

使用 Network Volume 方案可以：
- ✅ 大幅缩短构建时间（从数小时到 10-30 分钟）
- ✅ 减小镜像体积（从 92GB 到 5-10GB）
- ✅ 提高灵活性（无需重建镜像即可更新模型）
- ✅ 节省成本（更快的构建和部署）

**推荐在生产环境中使用优化方案！**

