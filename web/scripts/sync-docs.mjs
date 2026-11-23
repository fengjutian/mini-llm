#!/usr/bin/env node
/**
 * Sync the repository's `docs/*.md` into the Astro content collection.
 *
 * The documents under `docs/` are the single source of truth. New files are
 * picked up automatically; `KNOWN` only supplies the nicer sidebar metadata.
 * Three quirks are fixed up here:
 *
 *   1. the documents carry no frontmatter, so title/description/order are
 *      derived (or taken from `KNOWN`) and written as YAML frontmatter;
 *   2. LaTeX is written with escaped delimiters (`\\[ ... \\]`, `\\( ... \\)`)
 *      and `\\\` row separators, which are normalized to `$$ ... $$` so that
 *      remark-math can parse them;
 *   3. each document starts with an H1 that duplicates the page title, which
 *      is removed because the layout renders the title itself.
 *
 * Run through `pnpm run sync` (also invoked by `dev` and `build`).
 */
import { mkdir, readFile, readdir, rm, writeFile } from 'node:fs/promises';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const here = path.dirname(fileURLToPath(import.meta.url));
const webRoot = path.resolve(here, '..');
const repoRoot = path.resolve(webRoot, '..');
const docsDir = path.join(repoRoot, 'docs');
const outDir = path.join(webRoot, 'src', 'content', 'docs');

/**
 * Curated sidebar metadata. `order` controls the navigation order; `part`
 * numbers the "engine" series so the site can show progress through it.
 */
const ENGINE_SERIES = '引擎系列';
const DESIGN_SERIES = '项目设计';

const KNOWN = [
  {
    file: 'MATH_FOUNDATIONS.md',
    slug: 'math-foundations',
    title: '数学基础与神经网络原理',
    short: '数学基础',
    description: '线性代数、微积分、概率、优化与神经网络数学：LLM 背后那套可推导的直觉。',
    section: ENGINE_SERIES,
    series: ENGINE_SERIES,
    part: 1,
    order: 10,
  },
  {
    file: 'TENSOR_ENGINE.md',
    slug: 'tensor-engine',
    title: 'Tensor Engine 深度解析',
    short: 'Tensor Engine',
    description: '从内存布局、Shape、Stride 到矩阵乘法，拆开一个教学用张量引擎的内部结构。',
    section: ENGINE_SERIES,
    series: ENGINE_SERIES,
    part: 2,
    order: 20,
  },
  {
    file: 'AUTOGRAD_ENGINE.md',
    slug: 'autograd-engine',
    title: 'Autograd Engine 深度解析',
    short: 'Autograd Engine',
    description: '从计算图、链式法则到反向传播：梯度是怎样被自动算出来的。',
    section: ENGINE_SERIES,
    series: ENGINE_SERIES,
    part: 3,
    order: 30,
  },
  {
    file: 'NEURAL_NETWORK_ENGINE.md',
    slug: 'neural-network-engine',
    title: 'Neural Network Engine 深度解析',
    short: 'Neural Network',
    description: '从 Linear、MLP 到 Loss、Optimizer：把算子、梯度与损失串成可训练的网络。',
    section: ENGINE_SERIES,
    series: ENGINE_SERIES,
    part: 4,
    order: 40,
  },
  {
    file: 'TRANSFORMER_ENGINE.md',
    slug: 'transformer-engine',
    title: 'Transformer Engine 深度解析',
    short: 'Transformer Engine',
    description: '从 Embedding、位置编码到 Self-Attention 与 Multi-Head：Transformer 的计算核心。',
    section: ENGINE_SERIES,
    series: ENGINE_SERIES,
    part: 5,
    order: 50,
  },
  {
    file: 'TRANSFORMER_BLOCK.md',
    slug: 'transformer-block',
    title: 'Transformer Block 深度解析',
    short: 'Transformer Block',
    description: '从 FFN、残差连接、LayerNorm 到 Pre-Norm 与 GQA：一个 Block 里到底有什么。',
    section: ENGINE_SERIES,
    series: ENGINE_SERIES,
    part: 6,
    order: 60,
  },
  {
    file: 'DECODER_ENGINE.md',
    slug: 'decoder-engine',
    title: 'Decoder Engine 深度解析',
    short: 'Decoder Engine',
    description: '从 Transformer Stack、LM Head 到采样与 KV Cache：GPT 如何逐 token 生成文本。',
    section: ENGINE_SERIES,
    series: ENGINE_SERIES,
    part: 7,
    order: 70,
  },
  {
    file: 'TRAINING_ENGINE.md',
    slug: 'training-engine',
    title: 'Training Engine 深度解析',
    short: 'Training Engine',
    description: '从 Dataset、滑动窗口到 Cross Entropy、AdamW 与 checkpoint：把模型真正训练起来。',
    section: ENGINE_SERIES,
    series: ENGINE_SERIES,
    part: 8,
    order: 80,
  },
  {
    file: 'EVALUATION_ENGINE.md',
    slug: 'evaluation-engine',
    title: 'Evaluation Engine 深度解析',
    short: 'Evaluation Engine',
    description: '从 Validation Loss、Perplexity 到 Benchmark：训练之后如何判断模型真正学会了什么。',
    section: ENGINE_SERIES,
    series: ENGINE_SERIES,
    part: 9,
    order: 90,
  },
  {
    file: 'ARCHITECTURE.md',
    slug: 'architecture',
    title: '系统架构',
    short: '系统架构',
    description: 'crate 职责、依赖方向、Tensor 不变量与训练可观测性的设计约定。',
    section: DESIGN_SERIES,
    order: 100,
  },
  {
    file: 'PRD.md',
    slug: 'prd',
    title: '产品需求文档（PRD）',
    short: 'PRD',
    description: '目标用户、MVP 范围、验收标准与关键风险。',
    section: DESIGN_SERIES,
    order: 110,
  },
  {
    file: 'ROADMAP.md',
    slug: 'roadmap',
    title: '开发路线图',
    short: '路线图',
    description: '从可运行骨架到 Autograd、Tiny GPT、现代 LLM 与可观测研究环境的里程碑。',
    section: DESIGN_SERIES,
    order: 120,
  },
];

/** Rough reading time: CJK characters and Latin words at different speeds. */
function estimateMinutes(markdown) {
  const text = markdown
    .replace(/```[\s\S]*?```/g, ' ')
    .replace(/\$\$[\s\S]*?\$\$/g, ' ')
    .replace(/[#>*_`|~-]/g, ' ');
  const cjk = (text.match(/[\u4e00-\u9fff]/g) ?? []).length;
  const words = (text.replace(/[\u4e00-\u9fff]/g, ' ').match(/[A-Za-z0-9]+/g) ?? []).length;
  return Math.max(1, Math.round(cjk / 420 + words / 220));
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
      section: '其他',
      series: null,
      part: null,
      order: 200 + index,
    }
  );
});

let written = 0;
const report = [];

for (const doc of documents) {
  const sourcePath = path.join(docsDir, doc.file);
  const raw = await readFile(sourcePath, 'utf8');

  const normalized = normalizeMath(raw);
  const { markdown, heading } = stripLeadingHeading(normalized.markdown);

  const title = doc.title ?? heading ?? doc.slug;
  const minutes = estimateMinutes(markdown);
  const frontmatter = [
    '---',
    `title: ${yamlString(title)}`,
    `description: ${yamlString(doc.description)}`,
    `section: ${yamlString(doc.section)}`,
    ...(doc.series ? [`series: ${yamlString(doc.series)}`] : []),
    ...(doc.part ? [`part: ${doc.part}`] : []),
    ...(doc.short ? [`short: ${yamlString(doc.short)}`] : []),
    `minutes: ${minutes}`,
    `order: ${doc.order}`,
    `source: ${yamlString(`docs/${doc.file}`)}`,
    '---',
    '',
  ].join('\n');

  await writeFile(path.join(outDir, `${doc.slug}.md`), `${frontmatter}${markdown.trimEnd()}\n`, 'utf8');
  written += 1;
  report.push(
    `  sync  docs/${doc.file} -> src/content/docs/${doc.slug}.md (${normalized.converted} lines with math)`,
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
