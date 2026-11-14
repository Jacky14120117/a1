# AISC5030 Topic 3：科学数据中的知识发现

项目选题：**北京空气质量监测数据中的日污染模式发现：K-Means、Ward 层次聚类与 DBSCAN 的比较**。

本仓库当前完成的是课程资料阅读、方案设计、数据可行性核验和项目骨架。**聚类实验、研究结论、最终报告与答辩 PPT 尚未执行或生成。** 用户已明确指定 Topic 3，后续工作必须保持无监督聚类与知识发现方向。

## 阅读顺序

1. [课程资料提炼](docs/00_course_review.md)：Week 5 的方法、页码与适用边界。
2. [项目完成方案](docs/01_project_proposal.md)：科学问题、数据、方法与可行性。
3. [主次题目符合性检查](docs/02_requirement_alignment.md)：逐项对照课程要求。
4. [详细执行方案](docs/03_execution_plan.md)：从环境到交付的具体操作、命令接口和验收条件。
5. [项目架构](docs/04_architecture.md)：目录、模块、数据流和输出契约。
6. [数据字典](docs/05_data_dictionary.md) 与 [实验协议](docs/06_experiment_protocol.md)。
7. [交付清单与排期](docs/07_deliverables.md) 和 [下次 agent 交接](docs/08_agent_handoff.md)。

## 已核验的数据

使用 [UCI Beijing Multi-Site Air Quality](https://archive.ics.uci.edu/dataset/501/beijing+multi+site+air+quality+data) 的固定站点 Aotizhongxin。官方全量数据覆盖 12 个站点；本项目只分析其中一个站点，减少空间因素与计算量。引用、许可和下载入口见 [来源记录](references/sources.md)。

已下载官方原始压缩包并核验选定 CSV：35,064 条小时记录，2013-03-01 至 2017-02-28，共 1,461 个日历日。按“六种污染物各自每天至少 18 条有效小时记录”的预设规则，得到 **1,333 个可用日**，排除 128 日。这是数据结构与覆盖率核验，尚不是聚类结果。完整哈希与计数见 [source_preflight.json](data/metadata/source_preflight.json)。

```powershell
# 现在已经可运行，仅核验本地官方来源，不训练模型
python scripts/verify_source.py --extract-selected
python scripts/verify_scaffold.py
```

详细方案中的 `run_project.py` 等实验入口属于**下阶段待实现接口**，当前不可直接运行。请先完成执行方案的 T01–T02，再按阶段实现和运行。依赖范围在 [requirements.txt](requirements.txt)，尚未安装验证或生成实验环境锁文件。

## Agent 入口

先读根目录 [AGENTS.md](AGENTS.md)，再读项目内技能 [scientific-clustering/SKILL.md](.agents/skills/scientific-clustering/SKILL.md)。技能写在仓库中，由 AGENTS.md 明确指引加载；如果客户端没有自动发现仓库技能，可直接提供该路径。

`Week 5/` 保留课程原件。`data/raw/` 保留官方原始数据。生成结果按运行编号进入 `results/runs/`，不得将计划写成已经得到的实验结论。
