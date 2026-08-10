# 📚 项目文档导航

> 文档已按用途分入 4 个子目录：`design/`（设计与实现）、`training/`（训练与部署）、`paper/`（论文）、`_archive/`（历史备份）。

---

## 📁 快速导航（按子目录）

### 🎨 design/ — 设计与实现

| 文件 | 说明 |
|------|------|
| `design/LEARNING_GUIDE.md` | ⭐ **项目完全掌握指南**：知识地图 + 一次请求的旅程 + 7 大知识块核心讲解 + 学习路径 |
| `design/design-overview.md` | 三大增强模块（语义KB/Agent/推理引擎）设计文档 |
| `design/learning-notes.md` | 项目学习笔记 |
| `design/weather-implementation.md` | 天气服务与虫害风险评估实现细节 |
| `design/comprehensive-summary.md` | 项目综合总结报告 |
| `design/K-P01.md` | K-P01 技术要点 |
| `design/project-structure.md` | ⭐ 项目目录树：六大分类 + 后端依赖图 + 可清理清单 |

### 🏋️ training/ — 训练与部署

| 文件 | 说明 |
|------|------|
| `training/model-training-journey.md` | ⭐ 训练全记录 + 12 个讨论问题 |
| `training/GPU_TRAINING_GUIDE.md` | AutoDL GPU 租用与训练指南 |
| `training/autodl-5090-env-setup.md` | RTX 5090 环境搭建（nightly cu128 排障） |
| `training/review-notes-training-deploy.md` | ⭐ 完整复习笔记：训练排障 + 部署 + 论文规划 |
| `training/train_yolov11s_colab.ipynb` | Colab 训练笔记本（含蒸馏方案） |
| `training/DOCKER_DEPLOY.md` | 🐳 Docker 部署指南 |

### 📝 paper/ — 论文

| 文件 | 说明 |
|------|------|
| `paper/paper-outline.md` | ⭐ 论文结构草案（摘要/六章大纲/公式/图表清单） |
| `paper/paper-experiment-plan.md` | ⭐ 论文实验设计（对比/消融/问答评估/行动路线） |
| `paper/paper-system-robustness.md` | ⭐ 系统工程健壮性论文素材（实测指标 + 段落模板） |
| `paper/small-object-optimization.md` | ⭐ 小目标优化方案 + 论文素材 |

### 🗄️ _archive/ — 历史备份（日常忽略）

- `_archive/design-overview.html` / `.zip` — 设计文档可视化版 + 备份
- `_archive/learning-notes.html` / `.zip` — 学习笔记可视化版 + 备份

---

## 📊 演进时间线

```
07-28 ─── design-overview（系统设计）───── 三大增强模块诞生
07-29 ─── learning-notes / weather / summary / K-P01
07-30 ─── model-training-journey ⭐ / GPU_TRAINING_GUIDE / Colab
           数据集讨论 → 模型选型 → 蒸馏训练 → 论文定位
07-31 ─── 公网部署 + review-notes-training-deploy ⭐ / 论文规划
08-04 ─── 后端健壮性优化 + Docker + 小目标优化 + 论文素材整理
```

> 💡 **小提示**：`_archive/` 是历史备份，日常使用可忽略；以各子目录中的 Markdown 源文件为准。
