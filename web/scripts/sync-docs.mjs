#!/usr/bin/env node
/**
 * Sync the repository's `docs/*.md` into the Astro content collection.
 *
 * The documents under `docs/` are the single source of truth. New files are
 * picked up automatically; `KNOWN` only supplies the nicer sidebar metadata
 * (track, position, description). The script also fixes three quirks:
 *
 *   1. the documents carry no frontmatter, so title/description/order are
 *      derived (or taken from `KNOWN`) and written as YAML frontmatter;
 *   2. LaTeX is written with escaped delimiters (`\\[ ... \\]`, `\\( ... \\)`)
 *      and `\\\` row separators, which are normalized to `$$ ... $$` so that
 *      remark-math can parse them;
 *   3. each document starts with an H1 that duplicates the page title, which
 *      is removed because the layout renders the title itself.
 *
 * It also extracts a few "knowledge points" (structural H2 headings) per
 * document so the docs index can show and search them.
 *
 * Run through `pnpm run sync` (also invoked by `dev` and `build`).
 */
import { cp, mkdir, readFile, readdir, rm, writeFile } from 'node:fs/promises';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const here = path.dirname(fileURLToPath(import.meta.url));
const webRoot = path.resolve(here, '..');
const repoRoot = path.resolve(webRoot, '..');
const docsDir = path.join(repoRoot, 'docs');
const outDir = path.join(webRoot, 'src', 'content', 'docs');
const docsAssetsDir = path.join(docsDir, 'assets');
const publicDocsAssetsDir = path.join(webRoot, 'public', 'docs-assets');

/**
 * The docs are organised into tracks. `section` groups them in the sidebar and
 * the docs index; `part` numbers them inside the track. `order` is global and
 * leaves room for new documents in each track.
 */
const TRACKS = {
  implementation: '实现引擎',
  cognition: '认知与知识',
  planning: '规划与学习',
  agent: '智能体',
  design: '项目设计',
  extra: '扩展阅读',
};

const KNOWN = [
  // 01 · 实现引擎 -----------------------------------------------------------
  {
    file: 'MATH_FOUNDATIONS.md',
    slug: 'math-foundations',
    title: '数学基础与神经网络原理',
    short: '数学基础',
    description: '线性代数、微积分、概率、优化与神经网络数学：LLM 背后那套可推导的直觉。',
    section: TRACKS.implementation,
    series: TRACKS.implementation,
    part: 1,
    order: 10,
  },
  {
    file: 'TENSOR_ENGINE.md',
    slug: 'tensor-engine',
    title: 'Tensor Engine 深度解析',
    short: 'Tensor Engine',
    description: '从内存布局、Shape、Stride 到矩阵乘法，拆开一个教学用张量引擎的内部结构。',
    section: TRACKS.implementation,
    series: TRACKS.implementation,
    part: 2,
    order: 20,
  },
  {
    file: 'AUTOGRAD_ENGINE.md',
    slug: 'autograd-engine',
    title: 'Autograd Engine 深度解析',
    short: 'Autograd Engine',
    description: '从计算图、链式法则到反向传播：梯度是怎样被自动算出来的。',
    section: TRACKS.implementation,
    series: TRACKS.implementation,
    part: 3,
    order: 30,
  },
  {
    file: 'NEURAL_NETWORK_ENGINE.md',
    slug: 'neural-network-engine',
    title: 'Neural Network Engine 深度解析',
    short: 'Neural Network',
    description: '从 Linear、MLP 到 Loss、Optimizer：把算子、梯度与损失串成可训练的网络。',
    section: TRACKS.implementation,
    series: TRACKS.implementation,
    part: 4,
    order: 40,
  },
  {
    file: 'TRANSFORMER_ENGINE.md',
    slug: 'transformer-engine',
    title: 'Transformer Engine 深度解析',
    short: 'Transformer Engine',
    description: '从 Embedding、位置编码到 Self-Attention 与 Multi-Head：Transformer 的计算核心。',
    section: TRACKS.implementation,
    series: TRACKS.implementation,
    part: 5,
    order: 50,
  },
  {
    file: 'TRANSFORMER_BLOCK.md',
    slug: 'transformer-block',
    title: 'Transformer Block 深度解析',
    short: 'Transformer Block',
    description: '从 FFN、残差连接、LayerNorm 到 Pre-Norm 与 GQA：一个 Block 里到底有什么。',
    section: TRACKS.implementation,
    series: TRACKS.implementation,
    part: 6,
    order: 60,
  },
  {
    file: 'DECODER_ENGINE.md',
    slug: 'decoder-engine',
    title: 'Decoder Engine 深度解析',
    short: 'Decoder Engine',
    description: '从 Transformer Stack、LM Head 到采样与 KV Cache：GPT 如何逐 token 生成文本。',
    section: TRACKS.implementation,
    series: TRACKS.implementation,
    part: 7,
    order: 70,
  },
  {
    file: 'TRAINING_ENGINE.md',
    slug: 'training-engine',
    title: 'Training Engine 深度解析',
    short: 'Training Engine',
    description: '从 Dataset、滑动窗口到 Cross Entropy、AdamW 与 checkpoint：把模型真正训练起来。',
    section: TRACKS.implementation,
    series: TRACKS.implementation,
    part: 8,
    order: 80,
  },
  {
    file: 'EVALUATION_ENGINE.md',
    slug: 'evaluation-engine',
    title: 'Evaluation Engine 深度解析',
    short: 'Evaluation Engine',
    description: '从 Validation Loss、Perplexity 到 Benchmark：训练之后如何判断模型真正学会了什么。',
    section: TRACKS.implementation,
    series: TRACKS.implementation,
    part: 9,
    order: 90,
  },

  // 02 · 认知与知识 ---------------------------------------------------------
  {
    file: 'REPRESENTATION_ENGINE.md',
    slug: 'representation-engine',
    title: 'Representation Engine 深度解析',
    short: 'Representation',
    description: '从 Embedding Space、语义几何到 Feature 与概念：模型内部如何表示语言与世界。',
    section: TRACKS.cognition,
    series: TRACKS.cognition,
    part: 1,
    order: 110,
  },
  {
    file: 'MEMORY_ARCHITECTURE.md',
    slug: 'memory-architecture',
    title: 'Memory Architecture 深度解析',
    short: 'Memory',
    description: '短期、情景、语义与长期记忆，以及巩固和遗忘：记忆系统如何设计。',
    section: TRACKS.cognition,
    series: TRACKS.cognition,
    part: 2,
    order: 120,
  },
  {
    file: 'KNOWLEDGE_ARCHITECTURE.md',
    slug: 'knowledge-architecture',
    title: 'Knowledge Architecture 深度解析',
    short: 'Knowledge',
    description: '知识表示、本体、知识图谱与神经符号：事实、概念和关系如何组织。',
    section: TRACKS.cognition,
    series: TRACKS.cognition,
    part: 3,
    order: 130,
  },
  {
    file: 'CAUSAL_REASONING_ENGINE.md',
    slug: 'causal-reasoning-engine',
    title: 'Causal Reasoning Engine 深度解析',
    short: 'Causal Reasoning',
    description: '因果图、结构因果模型、干预与反事实：从相关走到因果。',
    section: TRACKS.cognition,
    series: TRACKS.cognition,
    part: 4,
    order: 140,
  },
  {
    file: 'WORLD_MODEL_ENGINE.md',
    slug: 'world-model-engine',
    title: 'World Model Engine 深度解析',
    short: 'World Model',
    description: '从状态表示、状态转移到预测与规划：模型如何不只理解语言，还能推演世界的变化。',
    section: TRACKS.cognition,
    series: TRACKS.cognition,
    part: 5,
    order: 150,
  },
  {
    file: 'COGNITIVE_ARCHITECTURE.md',
    slug: 'cognitive-architecture',
    title: 'Cognitive Architecture 深度解析',
    short: 'Cognitive Arch.',
    description: '工作记忆、信念系统、元认知与执行控制：一个认知循环由什么组成。',
    section: TRACKS.cognition,
    series: TRACKS.cognition,
    part: 6,
    order: 160,
  },

  // 03 · 规划与学习 ---------------------------------------------------------
  {
    file: 'PLANNING_ENGINE.md',
    slug: 'planning-engine',
    title: 'Planning Engine 深度解析',
    short: 'Planning',
    description: '目标拆解、搜索、分层规划、MCTS 与 MPC：从目标到可执行计划。',
    section: TRACKS.planning,
    series: TRACKS.planning,
    part: 1,
    order: 210,
  },
  {
    file: 'DECISION_ENGINE.md',
    slug: 'decision-engine',
    title: 'Decision Engine 深度解析',
    short: 'Decision',
    description: '效用函数、期望效用、贝叶斯决策与风险敏感规划：如何在不确定性下做选择。',
    section: TRACKS.planning,
    series: TRACKS.planning,
    part: 2,
    order: 220,
  },
  {
    file: 'LEARNING_ENGINE.md',
    slug: 'learning-engine',
    title: 'Learning Engine 深度解析',
    short: 'Learning',
    description: '反馈、经验回放、在线学习与强化学习：系统如何从经验中持续改进。',
    section: TRACKS.planning,
    series: TRACKS.planning,
    part: 3,
    order: 230,
  },

  // 04 · 智能体 -------------------------------------------------------------
  {
    file: 'AGENT_ENGINE.md',
    slug: 'agent-engine',
    title: 'Agent Engine 深度解析',
    short: 'Agent',
    description: '目标表示、任务规划到工具调用与自主执行：模型如何变成能完成任务的 Agent。',
    section: TRACKS.agent,
    series: TRACKS.agent,
    part: 1,
    order: 310,
  },
  {
    file: 'AGENT_ENGINE_RUNTIME.md',
    slug: 'agent-engine-runtime',
    title: 'Agent Engine Runtime 深度解析',
    short: 'Agent Runtime',
    description: 'Agent Loop、工具调用、运行时与多智能体：让 Agent 稳定跑起来的工程结构。',
    section: TRACKS.agent,
    series: TRACKS.agent,
    part: 2,
    order: 320,
  },
  {
    file: 'AGENT_LEARNING_ENGINE.md',
    slug: 'agent-learning-engine',
    title: 'Agent Learning Engine 深度解析',
    short: 'Agent Learning',
    description: '经验回放、信用分配、策略优化与持续学习：Agent 如何从执行中变强。',
    section: TRACKS.agent,
    series: TRACKS.agent,
    part: 3,
    order: 330,
  },

  // 05 · 项目设计 -----------------------------------------------------------
  {
    file: 'ARCHITECTURE.md',
    slug: 'architecture',
    title: '系统架构',
    short: '系统架构',
    description: 'crate 职责、依赖方向、Tensor 不变量与训练可观测性的设计约定。',
    section: TRACKS.design,
    order: 410,
  },
  {
    file: 'PRD.md',
    slug: 'prd',
    title: '产品需求文档（PRD）',
    short: 'PRD',
    description: '目标用户、MVP 范围、验收标准与关键风险。',
    section: TRACKS.design,
    order: 420,
  },
  {
    file: 'ROADMAP.md',
    slug: 'roadmap',
    title: '开发路线图',
    short: '路线图',
    description: '从可运行骨架到 Autograd、Tiny GPT、现代 LLM 与可观测研究环境的里程碑。',
    section: TRACKS.design,
    order: 430,
  },

  // 99 · 扩展阅读 -----------------------------------------------------------
  {
    file: 'WORLD_MODEL_ENGINE_CAUSAL_REASONING.md',
    slug: 'world-model-engine-causal-reasoning',
    title: 'World Model × Causal Reasoning',
    short: 'World Model 扩展',
    description: 'World Model 的扩展版：把状态表示、动力学、规划与因果推理放在同一条链上。',
    section: TRACKS.extra,
    order: 500,
  },
];

/** Structural heading prefixes used to pick "knowledge points". */
const structuralHeading =
  /^([一二三四五六七八九十百]+、|\d+(\.\d+)*[.、]?\s|第[一二三四五六七八九十百\d]+[章部]|Milestone\s|V\d+[：:]|Step\s\d)/;
const closingHeading = /^(结语|小结|本章小结|总结|参考资料|附录|后记|Definition of Done|下一篇)/;

function cleanHeading(text) {
  return text
    .replace(/\*\*/g, '')
    .replace(/^[一二三四五六七八九十百]+、\s*/, '')
    .replace(/^第[一二三四五六七八九十百\d]+[章部][：:]?\s*/, '')
    .replace(/^\d+(\.\d+)*[.、]?\s+/, '')
    .replace(/^[（(]\d+[)）]\s*/, '')
    .replace(/^[：:]\s*/, '')
    .replace(/[？?！!。.]+$/, '')
    .trim();
}

/** Pull a handful of structural headings to show as knowledge-point chips. */
function extractConcepts(markdown, limit = 5) {
  const lines = markdown.split('\n');
  let fence = null;
  const levelTwo = [];
  const levelThree = [];

  for (const line of lines) {
    const fenceMatch = /^\s*(```|~~~)/.exec(line);
    if (fenceMatch) {
      if (!fence) fence = fenceMatch[1];
      else if (fenceMatch[1] === fence) fence = null;
      continue;
    }
    if (fence) continue;

    const match = /^(#{2,3})\s+(.*)$/.exec(line.trim());
    if (!match) continue;
    const text = match[2].trim().replace(/\*\*/g, '').trim();
    if (!structuralHeading.test(text) || closingHeading.test(text)) continue;
    const depth = match[1].length;
    (depth === 2 ? levelTwo : levelThree).push(text);
  }

  const source = levelTwo.length >= 3 ? levelTwo : [...levelTwo, ...levelThree];
  const seen = new Set();
  const concepts = [];
  for (const heading of source) {
    if (/^(引言|前言|副标题)[：:]?/.test(heading)) continue;
    const concept = cleanHeading(heading);
    if (concept.length < 3 || seen.has(concept)) continue;
    seen.add(concept);
    concepts.push(concept.length > 26 ? `${concept.slice(0, 26)}…` : concept);
    if (concepts.length >= limit) break;
  }
  return concepts;
}

/**
 * Rough reading time for technical Chinese prose: characters and Latin words at
 * different speeds, plus a per-display-equation allowance. It intentionally runs
 * on the raw source: after math normalization, `$$` blocks can span lines and
 * make a naive text strip eat whole paragraphs.
 */
function estimateMinutes(raw) {
  const lines = raw.split(/\r?\n/);
  let fence = null;
  const kept = [];
  for (const line of lines) {
    const fenceMatch = /^\s*(```|~~~)/.exec(line);
    if (fenceMatch) {
      if (!fence) fence = fenceMatch[1];
      else if (fenceMatch[1] === fence) fence = null;
      continue;
    }
    if (fence) continue;
    if (/^\\\\\[/.test(line.trim()) || /^\\\\\]/.test(line.trim())) continue;
    kept.push(line);
  }

  const text = kept.join('\n').replace(/\\\\[a-zA-Z]+/g, ' ');
  const cjk = (text.match(/[\u4e00-\u9fff]/g) ?? []).length;
  const words = (text.replace(/[\u4e00-\u9fff]/g, ' ').match(/[A-Za-z0-9]+/g) ?? []).length;
  const equations = (raw.match(/\\\\\[/g) ?? []).length;
  return Math.max(1, Math.round(cjk / 380 + words / 240 + equations * 0.06));
}

const slugify = (value) =>
  value
    .replace(/\.md$/i, '')
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, '-')
    .replace(/^-+|-+$/g, '') || 'document';

const fencePattern = /^\s*(`{3,}|~{3,})/;

/**
 * Normalize the escaped LaTeX dialect used in `docs/` into the `$`/`$$`
 * delimiters understood by remark-math. Fenced code blocks are left untouched.
 */
function normalizeMath(markdown) {
  const lines = markdown.split(/\r?\n/);
  let fence = null;
  let converted = 0;

  const output = lines.flatMap((line) => {
    const fenceMatch = fencePattern.exec(line);
    if (fenceMatch) {
      if (fence === null) {
        fence = fenceMatch[1][0];
      } else if (fenceMatch[1][0] === fence) {
        fence = null;
      }
      return [line];
    }
    if (fence !== null) return [line];

    const next = line
      .replace(/\\\\\[/g, () => '$$')
      .replace(/\\\\\]/g, () => '$$')
      .replace(/\\\\\(/g, () => '$')
      .replace(/\\\\\)/g, () => '$')
      // Three backslashes are bmatrix row separators (`\\\` -> `\\`).
      .replace(/\\\\\\/g, () => '\\\\')
      // `\\|` is a mangled norm delimiter (`\|`).
      .replace(/\\\\\|/g, () => '\\|');

    if (next !== line) converted += 1;

    // `\\[ ... \\]` always occupies its own line in these documents. Put the
    // `$$` fences on their own lines so remark-math emits display math
    // (`katex-display`) instead of inline math.
    const trimmed = next.trim();
    if (next === trimmed && trimmed.startsWith('$$') && trimmed.endsWith('$$') && trimmed.length > 4) {
      const inner = trimmed.slice(2, -2).trim();
      return ['$$', inner, '$$'];
    }

    return [next];
  });

  return { markdown: output.join('\n'), converted };
}

/** Remove the leading H1, returning its text so metadata can be derived. */
function stripLeadingHeading(markdown) {
  const lines = markdown.split('\n');
  let index = 0;
  while (index < lines.length && lines[index].trim() === '') index += 1;
  if (index >= lines.length || !/^#\s+/.test(lines[index])) {
    return { markdown, heading: null };
  }
  const heading = lines[index].replace(/^#\s+/, '').replace(/\*\*/g, '').trim();
  lines.splice(index, 1);
  while (index < lines.length && lines[index].trim() === '') lines.splice(index, 1);
  return { markdown: lines.join('\n'), heading };
}

const yamlString = (value) => JSON.stringify(value);

await mkdir(outDir, { recursive: true });

// Keep Markdown image paths portable in the repository while exposing the
// same assets from a stable absolute URL in the generated Astro site.
await rm(publicDocsAssetsDir, { recursive: true, force: true });
await cp(docsAssetsDir, publicDocsAssetsDir, { recursive: true, force: true }).catch((error) => {
  if (error?.code !== 'ENOENT') throw error;
});

const files = (await readdir(docsDir, { withFileTypes: true }))
  .filter((entry) => entry.isFile() && entry.name.endsWith('.md'))
  .map((entry) => entry.name)
  .sort();

if (files.length === 0) {
  throw new Error(`No Markdown documents found in ${docsDir}`);
}

const knownByFile = new Map(KNOWN.map((doc) => [doc.file, doc]));
for (const doc of KNOWN) {
  if (!files.includes(doc.file)) {
    console.warn(`[sync-docs] warning: ${doc.file} is configured but missing from docs/`);
  }
}

const documents = files.map((file, index) => {
  const known = knownByFile.get(file);
  return (
    known ?? {
      file,
      slug: slugify(file),
      title: null,
      description: `来自 docs/${file} 的仓库文档。`,
      section: TRACKS.extra,
      series: null,
      part: null,
      order: 900 + index,
    }
  );
});

let written = 0;
const report = [];

for (const doc of documents) {
  const sourcePath = path.join(docsDir, doc.file);
  const raw = await readFile(sourcePath, 'utf8');

  const normalized = normalizeMath(raw);
  const siteMarkdown = normalized.markdown.replaceAll('](assets/', '](/docs-assets/');
  const { markdown, heading } = stripLeadingHeading(siteMarkdown);

  const title = doc.title ?? heading ?? doc.slug;
  const minutes = estimateMinutes(raw);
  const concepts = extractConcepts(markdown);
  const frontmatter = [
    '---',
    `title: ${yamlString(title)}`,
    `description: ${yamlString(doc.description)}`,
    `section: ${yamlString(doc.section)}`,
    ...(doc.series ? [`series: ${yamlString(doc.series)}`] : []),
    ...(doc.part ? [`part: ${doc.part}`] : []),
    ...(doc.short ? [`short: ${yamlString(doc.short)}`] : []),
    `concepts: ${yamlString(concepts)}`,
    `minutes: ${minutes}`,
    `order: ${doc.order}`,
    `source: ${yamlString(`docs/${doc.file}`)}`,
    '---',
    '',
  ].join('\n');

  await writeFile(path.join(outDir, `${doc.slug}.md`), `${frontmatter}${markdown.trimEnd()}\n`, 'utf8');
  written += 1;
  report.push(
    `  sync  docs/${doc.file} -> src/content/docs/${doc.slug}.md (${normalized.converted} math lines, ${concepts.length} concepts)`,
  );
}

// Drop pages whose source document disappeared, so stale routes do not linger.
const knownSlugs = new Set(documents.map((doc) => `${doc.slug}.md`));
for (const entry of await readdir(outDir)) {
  if (entry.endsWith('.md') && !knownSlugs.has(entry)) {
    await rm(path.join(outDir, entry), { force: true });
    report.push(`  prune src/content/docs/${entry}`);
  }
}

console.log(`[sync-docs] ${written} document(s) synced from docs/`);
console.log(report.join('\n'));
