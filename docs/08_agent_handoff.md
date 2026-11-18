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

### 2026-11-16（Asia/Shanghai）：T01–T03 完成，prepare 实际运行通过

- 环境：本机原无 Python，经 winget 安装 Python 3.12.10；项目内创建 `.venv`，按 `requirements.txt` 实际安装成功，`requirements-lock.txt` 来自 `pip freeze` 真实导出（numpy 2.5.4 / pandas 3.1.1 / scipy 1.18.1 / scikit-learn 1.9.1 / matplotlib 3.11.2 / seaborn 0.13.2 / PyYAML 6.0.3 / pytest 9.2.0）。
- T02：`verify_source.py --extract-selected` 复核通过，35,064 行 / 1,461 日 / 1,333 合格日 / 128 排除日，CSV SHA256 与预核验一致。
- T03：已实现 `src/air_quality_clustering/` 的 `config.py`、`data.py`、`features.py`、`provenance.py` 与 `scripts/run_project.py`（当前仅 prepare 阶段，其余阶段按执行方案待实现）。
- 测试：`tests/test_prepare.py` 13 项边界测试全部通过（17/18 小时阈值、零/负值/NA、各自有效小时均值、冲突小时报错、相同重复去重、非法 token 报错、六列固定与排序、气象不删日、RAIN 需 24 小时、季节约定、配置校验）。修复过两处问题：pandas 3.x groupby 组名兼容、一处测试断言计算错误。
- 实际运行：run_id `20261115T180625Z-0fafa468`，状态 completed，无警告。coverage 1,461 行；主日表 1,333×6（2013-03-01 至 2017-02-28，无缺失）；排除 128 日；主矩阵有限、日期唯一升序；各年合格日 2013:234 / 2014:335 / 2015:351 / 2016:356 / 2017:57，与预核验一致。派生表在 `data/interim/` 与 `data/processed/`，缩放参数与摘要在该运行目录。
- 尚未执行：T04 EDA、T05–T07 三方法候选、T08 选择与稳定性、T09–T10 解释与敏感性、T11–T12 报告与复现。聚类仍未拟合，不得把 prepare 产物写成聚类结论。
- 下一步：实现 `plots.py` 与 Notebook 01（EDA），随后 `clustering.py` / `evaluation.py` 按协议跑 89 个基础候选。

### 2026-11-16（Asia/Shanghai）：去冗余重构 + T04–T07 完成，89 候选实际运行通过

- 用户反馈数据处理步骤冗余，已重构 `data.py`：数值解析向量化、按日分组不再整表拷贝（`_by_day` 直接用归一化时间戳）、`daily_pollutants` 去掉与 coverage 重复的阈值参数；删除未使用的 `timedelta` 导入。口径不变，测试保驾。
- 新增模块：`evaluation.py`（内部指标有效性规则：单簇/全噪声返回 null+原因，DBSCAN 去噪后计分并记录 n_scored；ARI 置换不变）、`clustering.py`（E01 70 候选、E02 一棵树 7 切分、E03 12 组含自身 k-distance eps，重复 eps 复用拟合并留记录，eps<=0 记 invalid_eps）、`plots.py`（F01–F07，300dpi）。
- `run_project.py` 重写：支持多阶段 `--stage prepare eda cluster`，一次调用一个 run 目录一份 manifest；跨运行复用上游表时按 SHA256 校验；`stages_completed` 记录在 manifest 顶层（读取兼容旧结构）。新增 `eda` 配置段（图清单/dpi/spearman）。
- 测试：`tests/test_clustering.py` 新增 10 项（全噪声/单簇+噪声、ARI 置换、eps 含自身、12 行记录、Ward 单树嵌套一致性、同配置复现等），合计 23 项全部通过。期间修复：matplotlib 3.11 箱线图 `labels`→`tick_labels`；两处测试构造错误（对称序列反转等价置换、日均断言）；manifest 阶段字段位置不一致。
- 实际运行：run_id `20261115T184245Z-75ddd759`，prepare+eda+cluster 全部 completed，无警告。experiments.csv 89 行（kmeans 70 ok / ward 7 ok / dbscan 12 ok），89 份带 date 的标签（1333 行），7 张图（F01–F07），Ward linkage 与 DBSCAN eps 网格存 artifacts。
- 真实结果概况（仅候选，未选代表 k）：K-Means silhouette 中位数 k=2 最高 0.3607，随 k 增大下降；种子 ARI 中位数 k≤7 均 ≥0.96，k=8 降至 0.843（最小 0.524）。Ward k=2 silhouette 0.3575。同 k ARI：k=2 0.892，k=3 0.391，k=4 0.634。DBSCAN 仅 m=5/q70 得到 3 个非噪声簇（噪声 14.2%，子集 silhouette 0.10），其余 11 组均为单簇+噪声（silhouette null，原因 single_cluster）——数据更接近连续梯度，密度假设弱，如实保留。
- Notebook：`scripts/build_notebooks.py` 用 nbformat 生成 01/02（只调用 src 与运行产物，不复制实现）；`scripts/run_notebooks.py` 用 nbclient 实际从头执行两本均成功，执行副本在 `notebooks/executed/`。
- 尚未执行：T08 代表设置选择与时间块稳定性、T09–T10 解释与敏感性、T11–T12 报告与干净复现。不得把候选指标写成最终结论。
- 下一步：实现 `stability.py` 与 T08 选择规则（最小簇 14 日 → silhouette 差 ≤0.02 入围 → 20 次七日块 ARI 选代表），冻结后才进入季节/气象解释。

### 2026-11-17（Asia/Shanghai）：T08–T11 完成，全流水线与报告交付

- 新增模块：`stability.py`（七日块 209 块、20 次抽 167 块、子集重拟合缩放与模型、入围/代表/弱稳定性规则、DBSCAN 代表规则与噪声 Jaccard）、`discovery.py`（原始单位画像、列联、季节/年份/气象背景、PCA 仅展示、raw+scaler 敏感性）、`report.py`（全部数字取自运行产物的英文报告生成器）；`plots.py` 增 F08–F12。
- 入口扩展为七阶段 `prepare eda cluster stability compare sensitivity export`；同一次调用内状态复用，跨调用按 manifest+SHA256 校验；export 会把各来源运行图表补齐到本运行目录保证报告相对引用完整。
- 测试合计 30 项全部通过（新增 stability 7 项：块锚点/末块不足七日、子集确定性与无放回、最小簇排除、入围与并列取小 k、无合格代表、弱稳定性标志、DBSCAN 代表规则、失败次数不掩盖）。
- 完整运行 `20261116T193443Z-75ddd759` 七阶段 completed 无警告；收尾与报告最终运行 `20261116T193840Z-75ddd759`（export，含 13 张图与 report.md）。
- 真实结论：两主方法代表均 k=2（K-Means 中位块 ARI 0.970；Ward 0.707，刚过 0.70 警示线，IQR 0.554–0.799）；代表间 ARI 0.892，列联 799/13/24/497；同 k ARI 0.359–0.892。DBSCAN 无合格代表（11/12 单簇+噪声；唯一多簇设置覆盖率 0.858 但最小簇 <14 日）。敏感性：raw+scaler 下 K-Means ARI 0.392（划分明显重组）、Ward 0.719——log1p 是显式建模决定，报告如实区分。画像：C1 低负荷富臭氧（PM2.5 中位 39.0，O3 66.0，n=812），C2 高颗粒物/CO（PM2.5 119.5，O3 22.8，n=521）；C2 冬季占 36.1%、夏季 6.3%，温度中位 6.7°C vs C1 20.5°C（描述性，非因果）。
- Notebook 03 新增并三本全部 nbclient 从头执行成功。报告 Word 版经 md2docx 转换（13 图、10 表嵌入），交付副本在会话 outputs；项目内权威版本为运行目录 report.md。
- 环境注意：助理运行时 PYTHONPATH 会注入 cp314 依赖包，与项目 .venv(3.12) 冲突；运行项目命令前需在该进程清空 PYTHONPATH（未改系统配置）。
- 尚未做：T12 干净环境复现（新 venv 重装锁文件后重跑 all）与答辩 PPT；GitHub 推送已由用户自行通过 SSH 完成（远程已改为用户个人仓库 a1）。

### 2026-11-18（Asia/Shanghai）：应用户要求复核代码与结果，报告补全双方向季节表

- 新增 `scripts/verify_results.py`：独立复核漏斗口径（从 hourly_clean 重算 coverage/日均）、89 标签日期完整性、Ward 切分嵌套性、画像与原始中位数一致性、选参不变量、报告数字抽查；`--baseline` 模式跨运行对比指标/标签/选参/稳定性/敏感性。
- 报告完善：第 9 节补 Ward 季节构成表与"季节内簇占比"反方向表（协议要求双方向、分母写明）；Word 版同步重转（12 表 13 图）。
- 从头重跑全流水线：run `20261117T225450Z-75ddd759` 七阶段 completed 无警告；30 项测试通过；三本 Notebook 再次从头执行成功。
- 复核结论：17 项校验 ALL CHECKS PASSED；与基线运行 `20261116T193443Z` 对比，89 组指标逐值一致、89 份划分 ARI=1、代表/稳定性/敏感性完全相同——结果可复现、无计算错误。唯一曾 FAIL 项为校验脚本整表 dtype 严格比较的误报（CSV 往返布尔列变 object），已改为逐列语义比较。
- 权威报告现为 `results/runs/20261117T225450Z-75ddd759/report.md`；交付 Word 原地更新。

### 2026-11-18（Asia/Shanghai）：T12 干净环境复现完成，模型测试验收闭环

- 新建独立环境 `.venv-clean`（Python 3.12.10），仅按 `requirements-lock.txt` 安装（numpy 2.5.4 / pandas 3.1.1 / scikit-learn 1.9.1 等，与锁文件逐一对应）；`.venv-clean` 已加入 .gitignore，不入库。
- 干净环境中 30 项测试全部通过；`--stage all` 七阶段 completed 无警告，run `20261118T150843Z-75ddd759`。
- `verify_results.py --baseline 20261117T225450Z` 17 项 ALL CHECKS PASSED：89 组指标逐值一致、89 份划分 ARI=1、代表/稳定性/敏感性完全相同。至此三次独立运行（基线 193443Z、复核 225450Z、干净环境 150843Z）结果完全一致。
- 交付清单验收项全部满足：无监督口径、来源哈希、六列/日期一致、多参数真实比较、DBSCAN 退化披露、同空间评价、簇差异证据、三类稳健性、PCA 仅展示、簇名来自观测、报告数字与运行一致、测试通过+干净复现+交接更新。
- 剩余可选项：答辩 PPT（presentations/storyboard.md 已有 12 页结构建议），用户未要求前不启动。
