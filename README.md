# WeLM Blog

微信 WeLM 模型的技术解析博客系列。

## 博客列表

- [WeLM：以适度资源构建高效稀疏 MoE 模型 - 技术总结](./WeLM_1.md)
- [WeLM-258B MoE：后训练 (Post-Training) 技术与实战总结](./WeLM_2.md)

## 关于 WeLM

WeLM 是微信 AI 团队开发的稀疏 MoE（混合专家）大语言模型，在计算资源受限（不足 14T tokens）的条件下，成功训练出性能极具竞争力的 80B 及 130B 模型。

### 核心亮点

- **Loss-free Balance Routing**：无损均衡路由
- **KV-Mirror**：KV 镜像，降低推理成本
- **Over-encoding**：2-head 超编码
- **Depth Up-Scaling**：深度扩展技术
- **极致的系统优化**：通信掩盖、算子融合、激活显存优化

## 评测表现

| 评测基准 | WeLM 80B-A3B | WeLM 130B-A4.9B |
| :--- | :--- | :--- |
| MMLU | 85.65 | 86.91 |
| BBH | 85.47 | 88.05 |
| MATH | 60.04 | 58.78 |
| GSM8K | 87.72 | 90.90 |

## 仓库链接

https://github.com/Dominic789654/WeLM-blog
