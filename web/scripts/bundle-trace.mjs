// Bundles the newest traces/*.jsonl into the site:
//   - src/data/sample-trace.json  (imported at build time by /studio)
//   - public/sample-trace.jsonl   (downloadable copy / fetch fallback)
//
// Usage:
//   node web/scripts/bundle-trace.mjs            # fail when there is no trace
//   node web/scripts/bundle-trace.mjs --optional # warn and keep the old sample
//   TRACES_DIR=/tmp/traces node web/scripts/bundle-trace.mjs
//
// `--optional` is what `pnpm build` and `pnpm dev` run: `traces/` is git-ignored,
// so a fresh clone or a CI build has nothing to bundle and must still build.
import { copyFileSync, readdirSync, readFileSync, writeFileSync } from 'node:fs';
import { dirname, join, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';

const here = dirname(fileURLToPath(import.meta.url));
const root = join(here, '..', '..');
const tracesDir = resolve(process.env.TRACES_DIR ?? join(root, 'traces'));
const optional = process.argv.includes('--optional');
const sampleJson = join(root, 'web', 'src', 'data', 'sample-trace.json');
const sampleJsonl = join(root, 'web', 'public', 'sample-trace.jsonl');

function skip(message) {
  if (!optional) {
    console.error(message);
    process.exit(1);
  }
  console.warn(`skipped: ${message}; keeping the committed sample`);
  process.exit(0);
}

let names = [];
try {
  names = readdirSync(tracesDir).filter((name) => name.endsWith('.jsonl'));
} catch {
  skip(`no ${tracesDir} yet - run \`cargo run -p mini-llm-cli -- trace\` first`);
}

// Trace files are named `run-<unix millis>.jsonl`, so name order is time order.
names.sort();
const latest = names.at(-1);
if (!latest) skip(`no .jsonl trace in ${tracesDir} - run \`cargo run -p mini-llm-cli -- trace\` first`);

const source = join(tracesDir, latest);
const events = readFileSync(source, 'utf8')
  .split('\n')
  .filter((line) => line.trim().length > 0)
  .map((line) => JSON.parse(line));

writeFileSync(sampleJson, JSON.stringify(events));
copyFileSync(source, sampleJsonl);
console.log(`bundled ${latest}: ${events.length} events`);
