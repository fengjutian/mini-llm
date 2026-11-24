// Bundles the newest traces/*.jsonl into the site:
//   - src/data/sample-trace.json  (imported at build time by /studio)
//   - public/sample-trace.jsonl   (downloadable copy / fetch fallback)
//
// Usage: node web/scripts/bundle-trace.mjs
import { copyFileSync, readdirSync, readFileSync, writeFileSync } from 'node:fs';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';

const here = dirname(fileURLToPath(import.meta.url));
const root = join(here, '..', '..');
const tracesDir = join(root, 'traces');

let latest;
try {
  latest = readdirSync(tracesDir)
    .filter((name) => name.endsWith('.jsonl'))
    .sort()
    .pop();
} catch {
  console.error(`no ${tracesDir} yet - run \`cargo run -p mini-llm-cli -- trace\` first`);
  process.exit(1);
}

if (!latest) {
  console.error('no .jsonl trace found - run `cargo run -p mini-llm-cli -- trace` first');
  process.exit(1);
}

const source = join(tracesDir, latest);
const events = readFileSync(source, 'utf8')
  .split('\n')
  .filter((line) => line.trim().length > 0)
  .map((line) => JSON.parse(line));

writeFileSync(join(root, 'web', 'src', 'data', 'sample-trace.json'), JSON.stringify(events));
copyFileSync(source, join(root, 'web', 'public', 'sample-trace.jsonl'));
console.log(`bundled ${latest}: ${events.length} events`);
