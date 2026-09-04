# 世界模型研究笔记

由 [Austin-152](https://github.com/Austin-152) 整理的个人研究笔记，主题涵盖世界模型、VLA 系统、记忆模块以及机器人部署。

## 仓库内容

这里收录的是我在具身智能方向上的自学总结、技术笔记和面向实现的思考，主要来源于论文、工程文档和开源项目。

## 核心主题

- 用于机器人控制的 Vision-Language-Action（VLA）模型
- 世界模型，包括潜在动力学模型和扩散式模型
- 点云、场景图、NeRF、Gaussian Splatting 等三维场景表示
- 面向长时序机器人任务的记忆机制
- 训练策略、仿真到现实迁移与部署权衡

## 主要阅读入口

主文档：

- [世界模型调研汇总 -- Austin](./世界模型调研汇总%20--%20Austin.md)

同目录相关笔记：

- [Multical：核心算法与逻辑详解](./multical%20核心算法与逻辑详解.md)
- [Multical：技术分析报告](./multical%20技术分析报告.md)
- [VLAW 技术参考文档](./VLAW%20技术参考文档.md)
- [OpenVLA 论文工程技术参考文档](./OpenVLA%20论文工程技术参考文档.md)
- [PointWorld 论文总结报告](./论文总结报告：《PointWorld_%20Scaling%203D%20World%20Models%20for%20In-The-Wild%20Robotic%20Manipulation》.md)

## 参考来源

### 机器人策略模型

- [RT-2](https://robotics-transformer2.github.io/)
- [RT-X / Open X-Embodiment](https://robotics-transformer-x.github.io/)
- [OpenVLA](https://openvla.github.io/)
- [Octo](https://octo-models.github.io/)

### 世界模型与规划

- [DreamerV3](https://danijar.com/project/dreamerv3/)
- [TD-MPC2](https://tdmpc2.github.io/)
- [CTRL-World](https://arxiv.org/abs/2409.01306)
- [Gaussian World Model](https://arxiv.org/abs/2503.09751)

### 记忆与三维表示

- [NeRF](https://www.matthewtancik.com/nerf)
- [3D Gaussian Splatting](https://repo-supplied-link-placeholder.invalid)
- [Point World](https://qcnxc2ih90pn.feishu.cn/wiki/Gv8OwF8Luin4xgkAFt3cTQzJn9c)
- [VLAW](https://qcnxc2ih90pn.feishu.cn/wiki/WO3xwXX8XibmvdkOohncIx2an1b)
- [OpenVLA 笔记](./OpenVLA%20论文工程技术参考文档.md)

## 引用说明

部分材料最初来自内部整理或阅读平台链接。为了便于公开展示，我保留了摘要内容，并尽量补充可公开访问的原始出处；若此页尚未列出完整论文链接，对应细节可在各篇笔记中继续查看。

## 说明

这些内容属于个人学习型作品集，不是正式产品文档。它们主要展示我如何理解模型选型、系统设计以及研究到工程落地之间的关系。

## 版权

© 2026 [Austin-152](https://github.com/Austin-152). All rights reserved.
