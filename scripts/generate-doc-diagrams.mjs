#!/usr/bin/env node

import { mkdir, readFile, writeFile } from 'node:fs/promises';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const docsDir = path.join(root, 'docs');
const outDir = path.join(docsDir, 'assets', 'diagrams');

const specs = {
  'MATH_FOUNDATIONS.md': ['数学基础到神经网络', ['线性代数', '微积分', '概率统计', '优化方法', '神经网络']],
  'TENSOR_ENGINE.md': ['Tensor 的逻辑视图与物理计算', ['Storage', 'Shape / Stride', 'Tensor View', 'Operators', 'MatMul']],
  'AUTOGRAD_ENGINE.md': ['反向模式自动微分', ['Forward Ops', '计算图', 'Loss', '链式法则', 'Gradients']],
  'NEURAL_NETWORK_ENGINE.md': ['神经网络训练组件', ['Parameters', 'Linear / MLP', 'Loss', 'Backward', 'Optimizer']],
  'TRANSFORMER_ENGINE.md': ['Transformer 表征流水线', ['Token IDs', 'Embedding', 'Position', 'Self-Attention', 'Context']],
  'TRANSFORMER_BLOCK.md': ['Transformer Block 数据流', ['Input', 'Norm + Attention', 'Residual', 'Norm + FFN', 'Output']],
  'DECODER_ENGINE.md': ['自回归解码循环', ['Hidden State', 'LM Head', 'Logits', 'Sampling', 'Next Token']],
  'TRAINING_ENGINE.md': ['端到端训练闭环', ['Dataset', 'Batch', 'Forward', 'Loss + Backward', 'AdamW']],
  'EVALUATION_ENGINE.md': ['模型评估链路', ['Validation Set', 'Loss', 'Perplexity', 'Benchmarks', '报告与比较']],
  'REPRESENTATION_ENGINE.md': ['从向量到概念表征', ['Embedding Space', 'Features', 'Residual Stream', 'Concepts', 'World Representation']],
  'MEMORY_ARCHITECTURE.md': ['分层记忆系统', ['Working Memory', 'Episodic Memory', 'Semantic Memory', 'Consolidation', 'Long-term Memory']],
  'KNOWLEDGE_ARCHITECTURE.md': ['知识组织与推理', ['Observations', 'Entities', 'Ontology', 'Knowledge Graph', 'Reasoning']],
  'COGNITIVE_ARCHITECTURE.md': ['认知循环', ['Perception', 'Working Memory', 'Beliefs', 'Executive Control', 'Metacognition']],
  'WORLD_MODEL_ENGINE.md': ['世界模型的预测与行动', ['Observation', 'State', 'Transition', 'Prediction', 'Action']],
  'WORLD_MODEL_ENGINE_CAUSAL_REASONING.md': ['动力学与因果规划', ['State', 'Dynamics Model', 'Intervention', 'Rollout', 'Plan']],
  'WORLD_MODEL_ENGINE_LATENT_DYNAMICS.md': ['潜在空间世界模型', ['Observation', 'Encoder', 'Latent State', 'Dynamics', 'Prediction']],
  'CAUSAL_REASONING_ENGINE.md': ['因果推理阶梯', ['Causal Graph', 'SCM', 'Intervention', 'Counterfactual', 'Discovery']],
  'DECISION_ENGINE.md': ['不确定性下的决策', ['Belief State', 'Actions', 'Expected Utility', 'Risk', 'Decision']],
  'PLANNING_ENGINE.md': ['从目标到可执行计划', ['Goal', 'Decomposition', 'Search / MCTS', 'Schedule', 'Execution']],
  'SIMULATION_ENGINE.md': ['仿真与策略评估', ['Environment', 'Initial State', 'Transitions', 'Monte Carlo', 'Outcomes']],
  'EMBODIED_AI_ENGINE.md': ['具身智能感知行动闭环', ['Sensors', 'Sensor Fusion', 'World State', 'Motor Control', 'Physical Action']],
  'AGENT_ENGINE.md': ['目标驱动的 Agent Loop', ['Goal', 'Task Plan', 'Tool Call', 'Observation', 'Reflection']],
  'AGENT_ENGINE_RUNTIME.md': ['Agent Runtime 执行闭环', ['Request', 'Agent Loop', 'Runtime', 'Tools / Agents', 'Result']],
  'AGENT_LEARNING_ENGINE.md': ['Agent 从经验中学习', ['Trajectory', 'Replay', 'Credit Assignment', 'Policy Update', 'Improved Agent']],
  'LEARNING_ENGINE.md': ['持续学习闭环', ['Feedback', 'Experience', 'Replay Buffer', 'Learning Update', 'New Policy']],
  'META_CONTROLLER.md': ['自适应推理控制', ['Task', 'Difficulty', 'Model Routing', 'Compute Budget', 'Answer']],
  'RUNTIME_ENGINE.md': ['可靠执行运行时', ['Execution Graph', 'Scheduler', 'Async Workers', 'State Store', 'Recovery']],
  'PROTOCOL_ENGINE.md': ['Agent 通信协议栈', ['Function Schema', 'Tool Calling', 'MCP', 'Message Bus', 'A2A']],
  'OBSERVABILITY_ENGINE.md': ['Agent 可观测性闭环', ['Events', 'Logs / Metrics', 'Traces', 'Evaluation', 'Replay']],
  'SECURITY_ENGINE.md': ['纵深安全控制链', ['Identity', 'Permission', 'Policy', 'Sandbox', 'Audit']],
  'MULTI_AGENT_ENGINE.md': ['多智能体协作拓扑', ['Coordinator', 'Delegation', 'Specialist Agents', 'Shared Memory', 'Consensus']],
  'ARCHITECTURE.md': ['mini-llm 分层架构', ['CLI / Studio', 'Training / Inference', 'Model', 'Tensor API', 'CPU / Accelerated']],
  'PRD.md': ['产品能力演进', ['Educational Core', 'Autograd', 'Tiny GPT', 'Training', 'Research Studio']],
  'ROADMAP.md': ['mini-llm 里程碑', ['Tensor', 'Autograd', 'Tiny GPT', 'Modern LLM', 'Observable AI']],
};

const escapeXml = (value) => value.replaceAll('&', '&amp;').replaceAll('<', '&lt;').replaceAll('>', '&gt;');
const slug = (file) => file.replace(/\.md$/i, '').toLowerCase().replaceAll('_', '-');

function svg(title, nodes) {
  const width = 1400;
  const height = 430;
  const boxWidth = 210;
  const boxHeight = 94;
  const gap = 42;
  const startX = (width - (nodes.length * boxWidth + (nodes.length - 1) * gap)) / 2;
  const y = 198;
  const boxes = nodes.map((node, index) => {
    const x = startX + index * (boxWidth + gap);
    const arrow = index === nodes.length - 1 ? '' : `
    <path class="arrow" d="M ${x + boxWidth + 10} ${y + boxHeight / 2} H ${x + boxWidth + gap - 12}"/>
    <path class="arrow-head" d="M ${x + boxWidth + gap - 22} ${y + boxHeight / 2 - 8} L ${x + boxWidth + gap - 10} ${y + boxHeight / 2} L ${x + boxWidth + gap - 22} ${y + boxHeight / 2 + 8}"/>`;
    return `<g>
    <rect class="box box-${index}" x="${x}" y="${y}" width="${boxWidth}" height="${boxHeight}" rx="18"/>
    <text class="node" x="${x + boxWidth / 2}" y="${y + 56}" text-anchor="middle">${escapeXml(node)}</text>${arrow}
  </g>`;
  }).join('\n  ');

  return `<svg xmlns="http://www.w3.org/2000/svg" width="${width}" height="${height}" viewBox="0 0 ${width} ${height}" role="img" aria-labelledby="title desc">
  <title id="title">${escapeXml(title)}</title>
  <desc id="desc">${nodes.map(escapeXml).join('，')} 的流程图</desc>
  <style>
    :root { color-scheme: light dark; }
    .bg { fill: #f7f4ed; }
    .grid { stroke: #ded8cc; stroke-width: 1; opacity: .42; }
    .eyebrow { fill: #bc4b28; font: 700 17px ui-monospace, SFMono-Regular, Consolas, monospace; letter-spacing: 2px; }
    .heading { fill: #172126; font: 700 31px Inter, "Noto Sans SC", "Microsoft YaHei", sans-serif; }
    .box { fill: #fffdf8; stroke: #b8b1a5; stroke-width: 2; }
    .box-0, .box-${nodes.length - 1} { fill: #fff0e8; stroke: #d6653f; }
    .node { fill: #263238; font: 650 19px Inter, "Noto Sans SC", "Microsoft YaHei", sans-serif; }
    .arrow, .arrow-head { fill: none; stroke: #68777d; stroke-width: 3; stroke-linecap: round; stroke-linejoin: round; }
    @media (prefers-color-scheme: dark) {
      .bg { fill: #101416; }
      .grid { stroke: #334047; opacity: .42; }
      .heading { fill: #edf2f3; }
      .box { fill: #182024; stroke: #516068; }
      .box-0, .box-${nodes.length - 1} { fill: #32211c; stroke: #e17750; }
      .node { fill: #e7edef; }
      .arrow, .arrow-head { stroke: #91a1a8; }
      .eyebrow { fill: #ef855d; }
    }
  </style>
  <rect class="bg" width="${width}" height="${height}" rx="24"/>
  <path class="grid" d="M 70 130 H 1330 M 70 342 H 1330"/>
  <text class="eyebrow" x="70" y="70">MINI-LLM · CONCEPT MAP</text>
  <text class="heading" x="70" y="116">${escapeXml(title)}</text>
  ${boxes}
</svg>
`;
}

await mkdir(outDir, { recursive: true });

for (const [file, [title, nodes]] of Object.entries(specs)) {
  const diagramName = `${slug(file)}.svg`;
  await writeFile(path.join(outDir, diagramName), svg(title, nodes), 'utf8');

  const filePath = path.join(docsDir, file);
  const raw = await readFile(filePath, 'utf8');
  const marker = `](assets/diagrams/${diagramName})`;
  if (raw.includes(marker)) continue;

  const eol = raw.includes('\r\n') ? '\r\n' : '\n';
  const lines = raw.split(/\r?\n/);
  const headingIndex = lines.findIndex((line) => /^#\s+/.test(line));
  if (headingIndex < 0) throw new Error(`${file} has no H1 heading`);
  const insertion = [
    '',
    `![${title}](assets/diagrams/${diagramName})`,
    '',
    `_图：${title}。从左到右展示本章核心概念之间的数据流与依赖关系。_`,
  ];
  lines.splice(headingIndex + 1, 0, ...insertion);
  await writeFile(filePath, lines.join(eol), 'utf8');
}

console.log(`Generated and linked ${Object.keys(specs).length} document diagrams.`);
