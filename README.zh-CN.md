# SEES 论文代码发布准备包

本包已按 2026-11-02 的中英文审阅稿更新。主模型统一为 **G2 + typed
(`G2__dep_typed`)**；词级条件化、局部否定和联合结构作为配置对照列出。
英文首页见 [README.md](README.md)。

SEES 利用目标说话人的预测历史情绪指导词级和话语级证据聚合，通过类型化依存
关系增强词表示，并与对话上下文、角色交互和历史情绪分支结合，预测下一轮情绪。
未来话语不作为输入。G1 仍保留普通词级注意力，不等同于去掉证据路径。

本包包含模型、训练和评价代码、GLP-text 与 DialogInfer-S 适配基线、消融入口、
论文结果 CSV、框架图与可编辑 PPT，以及数据准备、命令和来源说明。
论文中的 DialogInfer-S 对应运行记录中的 `DialogInfer-S-adapted`。

请优先审阅：

1. [英文首页与主结果](README.md)：主模型、方法和基线比较。
2. [模型配置对照](docs/MODEL.md)：论文名称与代码标识一一对应。
3. [论文表格索引](results/manuscript/README.md)：15 张数字表，包含主结果、消融及附录数据。
4. [结果解释](docs/RESULTS.md)：支持哪些论点、哪些结果仍有限制。

SEES 在 MELD 和 DailyDialog 的 weighted-F1 高于两个基线，但没有在三个数据集的
所有指标上领先。主结果与整条证据路径的匹配重训结果分开保存；没有混用均值和标准差。
模型、训练和评价源码保持原样，这次只更新发布文档与结果导出。

运行 `python scripts/verify_release.py` 可核验代码包。训练前阅读
[DATA.md](docs/DATA.md) 和 [REPRODUCING.md](docs/REPRODUCING.md)。
数据集、权重、特征缓存和检查点不在 Git 仓库内；目前依赖已有预处理资产，
尚不能宣称从原始数据一键复现。DailyDialog 匹配实验的本机回传包没有对应 typed 权重，
不能以旧 G2 检查点替代。作者、论文链接和新代码许可仍待补齐。仓库目前用于私有审阅。
