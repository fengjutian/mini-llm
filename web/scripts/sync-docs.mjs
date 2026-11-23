#!/usr/bin/env node
/**
 * Sync the repository's `docs/*.md` into the Astro content collection.
 *
 * The documents under `docs/` are the single source of truth. They are plain
 * Markdown with three quirks this script fixes up:
 *
 *   1. they carry no frontmatter, so title/description/order come from DOCS;
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

/** Sidebar metadata. `order` controls the docs navigation. */
const DOCS = [
  {
    file: 'TENSOR_ENGINE.md',
    slug: 'tensor-engine',
    title: 'Tensor Engine 深度解析',
    description: '从内存布局、Shape、Stride 到矩阵乘法，拆开一个教学用张量引擎的内部结构。',
    section: '内核',
    order: 10,
  },
  {
    file: 'AUTOGRAD_ENGINE.md',
    slug: 'autograd-engine',
    title: 'Autograd Engine 深度解析',
    description: '从计算图、链式法则到反向传播：梯度是怎样被自动算出来的。',
    section: '内核',
    order: 15,
  },
  {
    file: 'MATH_FOUNDATIONS.md',
    slug: 'math-foundations',
    title: '数学基础与神经网络原理',
    description: '线性代数、微积分、概率、优化与神经网络数学：LLM 背后那套可推导的直觉。',
    section: '内核',
    order: 20,
  },
  {
    file: 'ARCHITECTURE.md',
    slug: 'architecture',
    title: '系统架构',
    description: 'crate 职责、依赖方向、Tensor 不变量与训练可观测性的设计约定。',
    section: '设计',
    order: 30,
  },
  {
    file: 'PRD.md',
    slug: 'prd',
    title: '产品需求文档（PRD）',
    description: '目标用户、MVP 范围、验收标准与关键风险。',
    section: '设计',
    order: 40,
  },
  {
    file: 'ROADMAP.md',
    slug: 'roadmap',
    title: '开发路线图',
    description: '从可运行骨架到 Autograd、Tiny GPT、现代 LLM 与可观测研究环境的里程碑。',
    section: '设计',
    order: 50,
  },
];

const fencePattern = /^\s*(`{3,}|~{3,})/;

/**
 * Normalize the escaped LaTeX dialect used in `docs/` into the `$`/`$$`
 * delimiters understood by remark-math. Fenced code blocks are left untouched.
 */
function normalizeMath(markdown, file) {
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

/** Remove the leading H1, returning its text so metadata can be validated. */
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

let written = 0;
const report = [];

await mkdir(outDir, { recursive: true });

for (const doc of DOCS) {
  const sourcePath = path.join(docsDir, doc.file);
  let raw;
  try {
    raw = await readFile(sourcePath, 'utf8');
  } catch (error) {
    throw new Error(`Cannot read ${path.relative(repoRoot, sourcePath)}: ${error.message}`);
  }

  const normalized = normalizeMath(raw, doc.file);
  const { markdown, heading } = stripLeadingHeading(normalized.markdown);

  if (heading && !heading.includes(doc.title.slice(0, 6))) {
    report.push(`  note  ${doc.file}: source heading "${heading}" differs from configured title`);
  }

  const frontmatter = [
    '---',
    `title: ${yamlString(doc.title)}`,
    `description: ${yamlString(doc.description)}`,
    `section: ${yamlString(doc.section)}`,
    `order: ${doc.order}`,
    `source: ${yamlString(`docs/${doc.file}`)}`,
    '---',
    '',
  ].join('\n');

  await writeFile(path.join(outDir, `${doc.slug}.md`), `${frontmatter}${markdown.trimEnd()}\n`, 'utf8');
  written += 1;
  report.push(`  sync  docs/${doc.file} -> src/content/docs/${doc.slug}.md (${normalized.converted} lines with math)`);
}

// Drop pages whose source document disappeared, so stale routes do not linger.
const known = new Set(DOCS.map((doc) => `${doc.slug}.md`));
for (const entry of await readdir(outDir)) {
  if (entry.endsWith('.md') && !known.has(entry)) {
    await rm(path.join(outDir, entry), { force: true });
    report.push(`  prune src/content/docs/${entry}`);
  }
}

console.log(`[sync-docs] ${written} document(s) synced from docs/`);
console.log(report.join('\n'));
