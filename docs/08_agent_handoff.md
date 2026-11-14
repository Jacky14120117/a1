# 下次 Agent 执行交接

规划交付日期：2026-11-15（Asia/Shanghai）。用户最新约束：**完全按照 Topic 3: Knowledge Discovery from Scientific Data，不能偏离。**

## 当前已经完成

- 阅读 Week 5 PPT 19 页、Notebook PDF 25 页、题目 PDF 1 页，并检查原始 ipynb。
- 写出课程信息、可行选题、主次题目逐项检查、所有执行细节、数据字典、实验协议、架构与交付清单。
- 创建项目各部分目录，以及根 AGENTS.md 与项目技能。
- 从 UCI 官方下载空气质量原始包，保存为 `data/raw/beijing_air_quality_source.zip`。
- 从原始 PRSA 内层包精确提取 Aotizhongxin 站点 CSV，文件与原字节一致。
- 运行 `scripts/verify_source.py --extract-selected`：35,064 小时记录、18 列、1,461 日；六列各 >=18 小时取交集得到 1,333 日，排除 128 日；没有重复小时键或负污染物浓度。
- 来源版本与全部计数在 `data/metadata/source_preflight.json`，脚本不训练聚类。

本轮结束前的骨架/技能核验记录另见 `docs/planning_validation.json`。不要将“规划文件已经准备完毕”写成“科学实验完成”。

## 尚未执行

尚未创建虚拟环境、安装项目依赖、生成真实 requirements-lock；尚未实现 src 的研究模块、统一运行入口、结果检查入口或三本 Notebook。`daily_pollutants.csv` 等日表尚未正式生成，1,333 日是来源预核验计数。尚未拟合聚类、确定 k、计算聚类分数或生成簇名、图表、研究结论、最终报告和答辩 PPT。

因此结果目录主要为空，模板含待填提示是正常状态。不要生成虚构的示例实验结果来填满目录。

## 下次执行入口

用户要求开始实施后，先读 `AGENTS.md` 和 `.agents/skills/scientific-clustering/SKILL.md`，再依次实施 `docs/03_execution_plan.md` 的 T01–T12。

首批实际任务：

1. 建立独立环境，验证依赖并保存真实锁文件。
2. 复核本地来源哈希，确认数据版本。
3. 按 `docs/04_architecture.md` 的模块契约实现 prepare、配置读取和日期/特征边界测试。
4. 生成正式日表、coverage 与主矩阵，核对日期集合及 1,333 日口径。
5. 连续推进 EDA、两主方法和 DBSCAN，再进行协议规定的选择、稳定性、解释与交付。

若用户只要求实现某阶段，遵守其范围；若明确要求做完整项目，则按整个方案自主推进，不在每个可逆步骤重复请求确认。

## 不得遗忘的约束

- 次题目已由用户明确选择，不能再按回归方案工作。
- 主输入六污染物；季节、天气不输入主聚类，也不能当作真值挑 k。
- 样本日聚合，NA 不补零，不强制训练/测试预测划分。
- K-Means 与 Ward 是核心，DBSCAN 允许无合格划分但应如实比较参数与噪声。
- 必须讨论簇画像与方法差异，不只给分数。
- 源数据外层有附加小 CSV，不能误用。

## 后续执行记录

从这里追加真实日期、完成阶段、运行 ID、测试结果、发现的问题和下一步。参数修订说明应同时写入对应运行 manifest，不能只留在聊天中。
