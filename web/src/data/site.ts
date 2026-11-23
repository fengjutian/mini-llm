export const SITE = {
  name: 'mini-llm',
  title: 'mini-llm · 用 Rust 理解大语言模型',
  description:
    'mini-llm 是一个教学优先的 Rust 项目：从 CPU 张量、Softmax、注意力，到一个小型 Transformer。可运行的代码、可推导的数学、可以亲手拖动的概念演示。',
  lang: 'zh-CN',
  license: 'Apache-2.0',
  version: 'v0.1',
} as const;

export const NAV = [
  { href: '/', label: '概览' },
  { href: '/learn', label: '概念图解' },
  { href: '/docs', label: '文档' },
] as const;

export type LessonVisual = 'tensor' | 'softmax' | 'attention';

export type Lesson = {
  href: string;
  index: string;
  tag: string;
  title: string;
  summary: string;
  visual: LessonVisual;
  time: string;
};

export const LESSONS: Lesson[] = [
  {
    href: '/learn/tensor-shape',
    index: '01',
    tag: '数据结构',
    title: '张量的形状、步长与偏移',
    summary:
      '一句话在模型里就是一个 [B, S, D] 张量。改一改 shape，看元素在一条线性内存里到底落在哪。',
    visual: 'tensor',
    time: '6 分钟',
  },
  {
    href: '/learn/softmax',
    index: '02',
    tag: '概率',
    title: 'Logits 怎样变成概率',
    summary:
      '模型吐出的是没有约束的分数，Softmax 把它们压成一个分布。拖动分数和温度，观察概率与熵的变化。',
    visual: 'softmax',
    time: '8 分钟',
  },
  {
    href: '/learn/attention',
    index: '03',
    tag: '注意力',
    title: '一次注意力究竟算了什么',
    summary:
      'Q 与 K 点积得到“该看谁”，除以 √d 之后做 Softmax，再用权重去混合 V。整条链路只有三步。',
    visual: 'attention',
    time: '10 分钟',
  },
];

export type RoadmapState = 'done' | 'current' | 'planned';

export type RoadmapItem = {
  id: string;
  title: string;
  state: RoadmapState;
  text: string;
};

export const ROADMAP: RoadmapItem[] = [
  {
    id: 'M0',
    title: '可运行骨架',
    state: 'done',
    text: 'Cargo workspace、CPU Tensor、Add / MatMul / ReLU / Softmax、单头注意力、CLI 与测试。',
  },
  {
    id: 'M1',
    title: 'Autograd',
    state: 'current',
    text: '计算图与 backward、基础算子梯度、finite-difference 梯度检查。',
  },
  {
    id: 'M2',
    title: '神经网络基础',
    state: 'planned',
    text: 'Linear、Embedding、LayerNorm、SGD / AdamW，以及 XOR 等小型训练任务。',
  },
  {
    id: 'M3',
    title: 'Tiny GPT',
    state: 'planned',
    text: '字符级 Tokenizer、Causal Attention、Transformer Decoder、训练循环与文本生成。',
  },
  {
    id: 'M4',
    title: '现代 LLM 组件',
    state: 'planned',
    text: 'RoPE、RMSNorm、SwiGLU、GQA、KV Cache 与 MoE。',
  },
  {
    id: 'M5',
    title: '训练配方',
    state: 'planned',
    text: 'Pretraining、SFT / Loss Masking、LoRA、DPO 与 GRPO 玩具环境。',
  },
  {
    id: 'M6',
    title: '加速与可观测',
    state: 'planned',
    text: 'Candle 后端、CPU/GPU 数值对照、训练事件协议与内部状态可视化。',
  },
];

export const PRINCIPLES = [
  {
    title: '从第一性原理出发',
    text: '先写透明的 CPU 张量运算，再往上搭网络。每一层都能读、能改、能测试。',
  },
  {
    title: '教学实现与加速分离',
    text: '保留一个可读的 CPU 参考后端，之后再引入 Candle 等加速实现。',
  },
  {
    title: '实验可复现',
    text: '配置、随机种子、数据集、指标与 checkpoint 都应该可追溯。',
  },
  {
    title: '内部状态可观察',
    text: '张量、激活、注意力权重、logits 与梯度都应能被取出并对照。',
  },
  {
    title: '架构保持模块化',
    text: '模型定义不绑定某个计算后端，crate 之间的依赖方向清晰。',
  },
  {
    title: '先正确，再快',
    text: '每个里程碑都要有可运行示例、数值测试与明确的限制说明。',
  },
];
