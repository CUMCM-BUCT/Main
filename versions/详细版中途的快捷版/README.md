# Main

数学建模代码、测试、数据处理与可复现实验仓库。

## 协作原则

- `main` 必须始终可运行，不直接推送。
- 一个任务对应一个短期分支和一个 Pull Request。
- 一个分支只有一名负责人；同一文件同一时刻只有一人修改。
- 按题目或功能划分目录，不按成员姓名划分正式代码。
- 公共模块必须单独开任务修改，并由相关问题负责人审核。
- 每个论文结果都应能追溯到本仓库的确定提交。

具体流程见 [CONTRIBUTING.md](CONTRIBUTING.md)，模块责任人见 [docs/OWNERS.md](docs/OWNERS.md)。

## 建议目录

```text
src/common/       公共读取、计算和绘图代码
src/q1/           第一问
src/q2/           第二问
src/q3/           第三问
tests/            快速检查与回归测试
scripts/          一键复现入口
data/raw/         原始数据，只读
data/processed/   可重新生成的数据
artifacts/final/  确认后供论文使用的结果
```

## A题 V4 当前进度

阶段1已固定三套参数包、单位与H1-H6假设，提供附件1/2的只读加载、严格校验、线性插值与尾部延拓，并定义四个结果工作簿的采样契约。

阶段2已实现四问共用的 cell-centered 径向有限体积核心、Backward Euler、Picard耦合、半单元 Robin 面值和中心重构。求解器在新时间层更新环境与半径，检查非线性残差，并对未收敛或负含水率状态拒步。`integrate` 只保证到达指定终时刻；正式采样点和事件试探时刻须由调用方分段传入。解析解验证、正式Q1-Q4计算与结果工作簿属后续阶段。

强非线性的半单元传质 Robin 代数方程在部分状态下存在多个数学根。程序以上一已接受时间层的表面含水率为初猜，优先用局部信赖域 Newton 连续校正同一根支；在折叠处局部根消失时，确定性的全域保障回退负责切换支路。公开 `boundary_state` 接口的 `surface_moisture_hint` 也只是局部校正初猜，不承诺返回与任意初猜欧氏距离最近的全局根。这是数值闭合选择，不能视为题给物理律唯一确定了表面支路；后续必须在边界离散和参数敏感性中报告该风险。

```powershell
powershell -File scripts/test.ps1
$env:PYTHONPATH = "src"
python scripts/check_inputs.py
```

细节见 [docs/model_v4_phase1.md](docs/model_v4_phase1.md)。

## 竞赛期间注意事项

当前仓库仅在本地配置。涉及当届赛题的内容是否可以存放到远端平台，应严格以当届竞赛规则和赛区解释为准。
