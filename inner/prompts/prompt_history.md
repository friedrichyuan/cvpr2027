我计划延续我们此前的工作AoE（AoE: Always-on Egocentric Human Video Collection for Embodied AI），做一个数据开源和数据工具的工作Open-AoE，其中包含3大块内容：
  1. 数据开源：数据量2000小时，标注包含原子动作切分和caption、手部mano及keypoint标注
  2. 数据工具：
    2.1. 包含数据可视化工具AoE-visualization（目标是方便用户直观地理解AoE数据）
    2.2. 数据重映射工具AoE-retarget-replay（目标是提供一个AoE数据 human-to-robot的转换范例，打通人类数据到机器人本体的通路，用可真机replay体现数据的准确性/稳定性/可学习）、
    2.3. 数据格式转换工具AoE-training-ready（目标是使得AoE数据可以直接用于行业主流模型的训练，打通人类数据到模型算法的通路，比如pi-0.5/gr00t/fastWAM/以及后续持续跟进打通爆款开源模型）
  3. 数据报告：包含对开源数据分布的洞察（场景/动作/时间/人员/采集机型等数据分布）等内容，需要依靠后续调研结果进行调整和补充
  
参考资料：
  - 已发布的AoE论文：/Users/woxue/Documents/卧雪的研究/Paper_Notes/AoE_CVPRW_2026
  - AoE样例数据：/Users/woxue/Documents/卧雪的研究/Open-AoE/sample_data

我需要你完成的工作：
  1. 使用deep-research技能，调研具身智能数据综述、具身智能开源数据集、及ego-centric方向的开源数据集（比如egodex/egolive），产出3方面内容，形成一个报告：
    1.1. 我想知道开源数据集工作，尤其是类似AoE数据的ego-centric开源数据集，在技术报告中应该包含哪些内容（比如数据分布可视化/数据质量筛选方法/数据标注方法/实验）？
    1.2. 我记得数据集工作通常会列一个表格，直观呈现自己的数据集与此前的开源数据集的对比，我想知道在具身智能数据/ego-centric数据上，其他工作是怎么列的表格？关注了哪些重要维度？AoE数据可能在哪些维度上体现优势/独特性/社区贡献？
    1.3. 根据以上两方面工作，顺便写一个可以放在技术报告中的relate work
  2. 使用deep-research技能，调研人类数据到机器人本体的转换方法（human-to-robot），根据论文引用数/github star数量来评估这些方法的质量，产出一个报告来说明AoE有可能集成哪些方法来构建数据重映射工具AoE-retarget-replay  
  3. 使用deep-research技能，调研具身智能行业主流开源模型（比如pi-0.5/gr00t），根据论文引用数/github star数量来评估这些模型的质量，产出一个报告来说明AoE有可能集成哪些模型来构建数据格式转换工具AoE-training-ready

补充说明：所有调研的中间信息搜集/下载的论文/代码/数据等，都归类放在/Users/woxue/Documents/卧雪的研究/Open-AoE文件夹下


在/Users/woxue/Documents/卧雪的研究/Open-AoE路径下，创建一个用于推动这个项目不断细化的脚手架，目标是实现多人协作贡献，以及AI协作，避免内容碎片化和复杂度爆炸。我认为这个脚手架需要包含：
- CLAUDE.md：用于记录和沟通这个项目的AI协作
- STORY.md：用于迭代这个项目的行业叙事，保障和实际的代码/数据实现不脱钩
- PROJECT.md：用于记录和沟通这个项目的多人分工和协作
- release文件夹：用于迭代这个项目未来会发布出去的代码/数据
- report文件夹：用于迭代这个项目的报告，latex格式
- inner文件夹：用于迭代这个项目的内部代码/数据
请结合你现在调研的内容，设计并实现这个脚手架，重新组织文件夹中已有的内容，并添加README.md

现在，我需要形成一份模块分工文档输出到：/Users/woxue/Documents/卧雪的研究/Open-AoE/inner/drafts，为所有Open-AoE项目的参与者呈现这个项目的最终目标发布内容（数据集/报告/代码库），这些内容中预计包含的模块，以及这些模块的简要介绍，这个文档的主要内容是让所有参与者可以看到项目的工作量，然后进行分工和进度追踪。可以调用huamei-draw技能来绘制框架图更直观呈现