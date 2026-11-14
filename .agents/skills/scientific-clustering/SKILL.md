---
name: scientific-clustering
description: "Execute and review the AISC5030 Topic 3 air-quality knowledge-discovery project, including data preparation, clustering comparisons, stability, interpretation and reproducible reporting. Use for this repository's unsupervised project implementation; do not redirect it to prediction."
---

# Scientific Clustering

为本仓库实施 Topic 3 的科学数据知识发现。先阅读 [AGENTS.md](../../../AGENTS.md) 和 [当前交接](../../../docs/08_agent_handoff.md)，确认用户要求的阶段与真实完成状态。用户最新要求优先于项目文档。

## 按任务加载资料

- 开始实施或整理整体方案：读 [执行手册](../../../docs/03_execution_plan.md)、[架构](../../../docs/04_architecture.md) 与 `configs/project.yaml`。
- 处理来源、清洗、字段或样本：读 [数据字典](../../../docs/05_data_dictionary.md)，运行本地 `scripts/verify_source.py`，核对版本与日覆盖口径。
- 运行方法、选参数或评价：读 [实验协议](../../../docs/06_experiment_protocol.md)，不要凭常识另设评分或固定簇数。
- 解释与提交：读 [交付清单](../../../docs/07_deliverables.md)、[来源](../../../references/sources.md) 和报告模板。
- 用户要求核对题意：对照 [符合性检查](../../../docs/02_requirement_alignment.md)，明确哪些是教师要求、哪些是研究设计。

## 核心决策

研究单位是固定 Aotizhongxin 站点的合格日。主输入严格六污染物，逐列 >=18 有效小时的交集后计算日均。原始版本预核验 35,064 小时、1,461 日、1,333 合格日；版本改变时重算，不能以这些计数为目标调整数据。

主表示为 log1p+StandardScaler。主比较 K-Means 和 Ward，DBSCAN 提供密度/噪声补充。没有监督标签，不建立预测 y；日期、季节、气象、站点和 AQI 等背景不能进入主距离或当作调参真值。PCA 二维坐标只展示标签。

遵循协议的候选网格与代表设置规则。先跑全部候选，再按内部指标入围，做时间块稳定性，冻结代表设置，最后查看季节/天气背景。不要先命名污染类型再寻找支持结果。

## 实施与核验

1. 环境和锁文件来自实际成功安装。原始文件只读保留；不要使用外层附带的小 data.csv/test.csv。
2. 将数据、变换、聚类、评价、稳定性、画像和输出来源拆成架构中定义的轻量模块。Notebook 调用模块，避免重复实现。
3. 主方法用同日同六维空间评价。DBSCAN 的 -1 排除在污染簇与内部指标外，同时报告覆盖率、噪声率和无效原因；不要用子集 silhouette 赢过全样本结果。
4. 用日期对齐不同标签，以 ARI 比较，不按任意簇号相等计算正确率。既比较各代表结果，也保留同 k 的主方法对照。
5. 每次块子采样重新拟合缩放与聚类，保留日历块和实际成员清单。它是结构稳健性诊断，不是预测泛化或独立确认。
6. 从原始日均浓度计算簇中位数/IQR及大小。日期/气象只在后解释连接，不因天气缺失减少主样本。不要把 log 中心反变换称为原始算术均值。
7. 测试实质边界：17/18 小时、零/负值、六列交集、排除字段、日期连接、全噪声/单簇、簇号置换和同配置复现。保存失败候选而非改写为 0 分。

## 产出与停止条件

每个实际运行保存配置、数据/源码哈希、依赖、种子、候选表、标签、警告、稳定性和选参依据。结果进入独立 run_id，不覆盖历史。报告发现必须有原始单位画像、方法差异与局限支持，不能宣称健康等级、污染源或因果效果。

完成用户指定阶段后更新交接。若授权完整实施，以交付清单验收后停止；若仅授权规划或某个模块，完成该范围后停止。不要因 skill 存在就自行启动未要求的长期实验、发布或提交 Git 变更。
