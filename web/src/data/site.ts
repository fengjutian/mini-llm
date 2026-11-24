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
  { href: '/studio', label: '研究台' },
  { href: '/docs', label: '知识地图' },
] as const;

export type LessonVisual = 'tensor' | 'softmax' | 'attention' | 'autograd' | 'rope' | 'sampling';

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
  {
    href: '/learn/autograd',
    index: '04',
    tag: '训练',
    title: '梯度是怎样流回参数的',
    summary:
      '反向传播不是另一套公式，而是把每个算子的局部导数沿计算图倒着乘一遍。一次 backward，所有参数同时拿到梯度。',
    visual: 'autograd',
    time: '7 分钟',
  },
  {
    href: '/learn/rope',
    index: '05',
    tag: '位置编码',
    title: '位置信息怎样进入注意力',
    summary:
      'RoPE 不往 embedding 里加位置向量，而是按位置旋转 Q 和 K。旋转之后，点积只和两者的相对距离有关。',
    visual: 'rope',
    time: '8 分钟',
  },
  {
    href: '/learn/sampling',
    index: '06',
    tag: '生成',
    title: '同一个模型，为什么每次说得不一样',
    summary:
      '模型给的是分布，不是答案。温度缩放分布，采样再从里面掷骰子；temperature = 0 时才退化成贪心。',
    visual: 'sampling',
    time: '6 分钟',
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
    state: 'done',
    text: '计算图与 backward、基础算子梯度、finite-difference 梯度检查。',
  },
  {
    id: 'M2',
    title: '神经网络基础',
    state: 'done',
    text: 'Linear、Embedding、LayerNorm、SGD / AdamW，以及 XOR 等小型训练任务。',
  },
  {
    id: 'M3',
    title: 'Tiny GPT',
    state: 'done',
    text: '字符级 Tokenizer、Causal Attention、Transformer Decoder、训练循环与文本生成。',
  },
  {
    id: 'M4',
    title: '现代 LLM 组件',
    state: 'done',
    text: 'RoPE、RMSNorm、SwiGLU、GQA、KV Cache 与 MoE。',
  },
  {
    id: 'M5',
    title: '训练配方',
    state: 'done',
    text: 'Pretraining、SFT / Loss Masking、LoRA、DPO 与 GRPO 玩具环境。',
  },
  {
    id: 'M6',
    title: '加速与可观测',
    state: 'done',
    text: 'Candle 后端、CPU/GPU 数值对照、训练事件协议与内部状态可视化。',
  },
];

export type DocTrack = {
  section: string;
  anchor: string;
  blurb: string;
};

/**
 * Knowledge tracks for the docs. The `section` matches the frontmatter written
 * by `scripts/sync-docs.mjs`; `anchor` is used by the docs index headings.
 */
export const DOC_TRACKS: DocTrack[] = [
  {
    section: '实现引擎',
    anchor: 'track-implementation',
    blurb: '从第一性原理实现一个能训练、能生成的模型。',
  },
  {
    section: '认知与知识',
    anchor: 'track-cognition',
    blurb: '模型如何表示世界、保存记忆、组织知识并理解因果。',
  },
  {
    section: '规划与学习',
    anchor: 'track-planning',
    blurb: '在不确定性下规划与决策，并从经验中持续学习。',
  },
  {
    section: '智能体',
    anchor: 'track-agent',
    blurb: '把模型变成能调用工具、循环执行并自我改进的 Agent。',
  },
  {
    section: '项目设计',
    anchor: 'track-design',
    blurb: 'crate 边界、MVP 范围与工程里程碑。',
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
