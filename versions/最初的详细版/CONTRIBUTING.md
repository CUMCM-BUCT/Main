# 代码仓库协作规范

## 开始任务

1. 在任务单写明唯一负责人、允许修改的文件和验收条件。
2. 从最新 `main` 创建分支：`feat/<任务号>-<简短描述>` 或 `fix/<任务号>-<简短描述>`。
3. 分支只能由负责人推送，其他成员通过评审意见协作。

```bash
git switch main
git pull --ff-only origin main
git switch -c feat/12-q2-optimization
```

## 提交与审核

- 只暂存本任务文件，不使用无检查的 `git add .`。
- 推荐提交格式：`feat(q2): implement optimization model`。
- 合并前同步最新 `main`，运行 `scripts/check.ps1` 和本任务测试。
- PR 必须由另一位成员审核；每次只合并一个 PR。
- 使用 Squash and merge，合并后删除任务分支。

## 数据和实验

- `data/raw/` 中的原始数据视为只读；清洗结果写入其他目录。
- 不提交虚拟环境、缓存、临时结果、密钥或个人电脑绝对路径。
- 为随机算法固定种子，并记录依赖版本和复现命令。
- 不多人编辑同一个 Notebook；每个任务使用独立 Notebook，稳定逻辑再转入 `src/`。
- 最终图表进入 `artifacts/final/` 前须经过结果负责人确认。

## 冲突处理

冲突只能在任务分支解决。PR 作者同步最新 `main` 后，与文件负责人共同决定保留内容，重新运行测试，再推送任务分支。禁止使用强制覆盖来跳过人工判断。
