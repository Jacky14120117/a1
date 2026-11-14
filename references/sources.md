# 来源与引用记录

核查日期：2026-11-14。来源分为课程要求、数据说明、原始科学背景和软件方法。文档中的研究阈值与参数范围属于项目设计，不归因给这些来源。

## 课程原件

- [Week 5 PPT.pdf](../Week%205/Week%205%20PPT.pdf)：19 页。无监督方法见第 9 页，工作流见第 13 页。
- [Week 5 Notebook.pdf](../Week%205/Week%205%20Notebook.pdf)：25 页，California Housing 回归示例。
- [Week 5 Notebook.ipynb](../Week%205/Week%205%20Notebook.ipynb)：132 个单元格、59 个代码单元格。
- [Week 5 Project Topics.pdf](../Week%205/Week%205%20Project%20Topics.pdf)：第 1 页，主题目及 Topic 3 的硬要求。

## 数据与科学背景

1. Chen, S. (2017). *Beijing Multi-Site Air Quality* [Dataset]. UCI Machine Learning Repository. [DOI](https://doi.org/10.24432/C5RK5G)，[官方说明](https://archive.ics.uci.edu/dataset/501/beijing+multi+site+air+quality+data)。用于来源、全量大小、日期范围、字段单位与许可。官方全量 420,768 小时记录、12 个站点；本项目的单站点与清洗后规模来自本地核验。
2. [官方原始下载入口](https://archive.ics.uci.edu/static/public/501/beijing+multi+site+air+quality+data.zip)。原始 PRSA ZIP 内选定站点，禁止用外层附加小型 data.csv/test.csv 替代全站点 CSV。
3. Zhang, S., Guo, B., Dong, A., He, J., Xu, Z., & Chen, S. X. (2017). *Cautionary tales on air-quality improvement in Beijing*. Proceedings of the Royal Society A, 473, 20170457. [原始论文全文](https://pmc.ncbi.nlm.nih.gov/articles/PMC5627385/)，[DOI](https://doi.org/10.1098/rspa.2017.0457)。用于理解气象混杂与观测数据解释边界。该论文研究范围比本项目及 UCI 子集更大，不把论文的站点数或分析结论当成本项目数据规模或复现结果。
4. 数据许可：[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)。保留数据作者、来源及变换说明。本仓库仅记录数据许可，不据此给课程课件或所有代码自动重新授权。

## 标准方法文档

- [scikit-learn Clustering](https://scikit-learn.org/stable/modules/clustering.html)：方法假设、聚类评价与局限。
- [KMeans](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html)：参数与显式 n_init。
- [DBSCAN](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.DBSCAN.html)：eps、min_samples、-1 噪声标记。min_samples 包含样本本身。
- [SciPy linkage](https://docs.scipy.org/doc/scipy/reference/generated/scipy.cluster.hierarchy.linkage.html) 与 [cut_tree](https://docs.scipy.org/doc/scipy/reference/generated/scipy.cluster.hierarchy.cut_tree.html)：Ward 欧氏层次结构与精确指定簇数的切分。
- [silhouette_score](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.silhouette_score.html)：有效标签数须满足 2 <= n_labels <= n_samples-1。
- [adjusted_rand_score](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.adjusted_rand_score.html)：划分比较对簇编号置换不敏感，不要求类别真值。
- [StandardScaler](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html)：缩放、异常值敏感性。
- [Common pitfalls](https://scikit-learn.org/stable/common_pitfalls.html)：学习型预处理与复现注意事项；本项目不把探索性稳定性冒称为独立预测验证。

实施时记录实际安装版本。这里的 stable 文档可能更新，必要时改查与锁定依赖版本对应的文档。
