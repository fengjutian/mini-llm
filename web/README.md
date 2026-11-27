# mini-llm web

The Astro site for mini-llm. Its job is to make the concepts behind the Rust code
easier to understand, not to run the model: the interactive lessons recompute
small examples in the browser, while the docs are the repository's Markdown
rendered as static pages.

## Stack

- **Astro 7** with the static output adapter (no SSR)
- **React 19** island for `/studio` (`@astrojs/react`); the three lessons remain
  plain `<script>` blocks so they ship as small vanilla TypeScript modules
- **Unified** Markdown pipeline (`@astrojs/markdown-remark`) for `remark-math` +
  `rehype-katex`, because the repository docs contain a lot of LaTeX
- Shiki for syntax highlighting, with separate light and dark themes

## Research Studio

`/studio` loads the JSONL trace produced by
`cargo run -p mini-llm-cli -- trace` and renders the loss curve, every attention
matrix (layer × head) and the final-position logits. The bundled sample lives at
`src/data/sample-trace.json`; `pnpm dev` and `pnpm build` regenerate it from the
newest trace automatically:

```bash
cargo run -p mini-llm-cli -- trace   # writes traces/run-<millis>.jsonl
pnpm build                           # bundles the newest trace, then builds
```

`traces/` is git-ignored, so a fresh clone or a CI build has nothing to bundle:
`bundle-trace.mjs --optional` (what the scripts run) prints a warning and keeps
the committed sample instead of failing. Run `pnpm run sample` for the strict
version that exits non-zero when no trace exists.

The page also accepts any `.jsonl` through its file picker, so a new run can be
inspected without rebuilding the site.

### Live trace (SSE)

To watch a run while it is being written, start the watcher in one terminal and
train in another:

```bash
cargo run -p mini-llm-cli -- watch-trace            # tail newest traces/*.jsonl
cargo run -p mini-llm-cli -- trace                  # the run it streams
```

`watch-trace` serves `GET /trace` as `text/event-stream` on
`127.0.0.1:8090`: every JSONL line becomes one `data:` frame, and a subscriber
that joins mid-run first receives everything already on disk. The server sends
`Access-Control-Allow-Origin: *`, so the static site can subscribe from another
port. In the browser, click “连接实时”, or open
`/studio?live=http://127.0.0.1:8090/trace` to connect on load (an empty
`?live=` means the default URL). The browser reconnects on its own and the
server replays from the start, so a dropped connection never duplicates events:
a `run_started` frame always resets the view.

Useful flags: `--addr 127.0.0.1:PORT`, `--file traces/<run>.jsonl` (default: the
newest file by name). `GET /health` reports the watched file plus frame and
subscriber counters.

## Scripts

```bash
pnpm install
pnpm dev       # sync docs + bundle newest trace, then http://localhost:4321
pnpm build     # sync docs + bundle newest trace, then static output in dist/
pnpm preview   # serve dist/
pnpm sync      # only regenerate src/content/docs
pnpm sample    # only bundle traces/*.jsonl into the studio sample
```

`dev` / `build` run `scripts/sync-docs.mjs` and `scripts/bundle-trace.mjs
--optional` first. The bundling step is best-effort: with no `traces/` directory
it keeps the committed sample and the build still succeeds.

## How the docs are synced

`docs/*.md` at the repository root is the single source of truth. Every run:

1. discovers `docs/*.md` automatically, so a new document becomes a route without
   editing the site;
2. writes frontmatter: curated `title` / `description` / `section` / `series` /
   `part` / `short` from the `KNOWN` table, plus a computed reading time
   (`minutes`) and the source path;
3. drops the document's leading H1, because the page layout renders the title;
4. converts the escaped LaTeX dialect used in these documents
   (`\\[ ... \\]`, `\\( ... \\)`, `\\\` matrix row separators, `\\|`) into the
   `$` / `$$` delimiters that remark-math understands, and puts `$$` fences on
   their own lines so equations become display math;
5. writes the result to `src/content/docs/`, which is git-ignored and pruned of
   stale pages on every run.

`KNOWN` is optional metadata. Documents missing from it still render (title from
the leading H1, `section: '扩展阅读'`), but they get no part number and appear
after the curated ones.

The docs are grouped into knowledge tracks: `实现引擎`, `认知与知识`,
`规划与学习`, `智能体`, `项目设计`, and `扩展阅读`. `part` numbers a document
inside its track. The sync script also extracts up to five structural H2 headings
as `concepts`, which the docs index shows as knowledge-point chips and uses for
client-side search. Bump `part`/`order` to slot a new document into a track.

## Structure

```text
web/
├── astro.config.mjs        # markdown processor, Shiki themes, SITE_URL
├── scripts/sync-docs.mjs   # docs/ -> src/content/docs/
├── public/                 # favicon, og image, robots.txt
└── src/
    ├── content.config.ts   # docs collection schema
    ├── data/site.ts        # navigation, lessons, roadmap, principles
    ├── styles/global.css   # design tokens and shared styles
    ├── components/         # header, footer, lesson card, docs sidebar, TOC
    ├── layouts/            # base, docs, lesson
    └── pages/
        ├── index.astro
        ├── learn/{index,tensor-shape,softmax,attention}.astro
        └── docs/{index,[...slug]}.astro
```

## Design notes

- Colors, type and spacing live as CSS custom properties at the top of
  `src/styles/global.css`; the theme is a `data-theme` attribute on `<html>`.
- The theme follows the OS by default and can be toggled in the header, or forced
  with `?theme=dark` / `?theme=light` (the choice is persisted).
- The docs index is a searchable knowledge map: six tracks, numbered parts,
  reading time per document, and concept chips extracted from each document.
- The docs table of contents is generated from the Markdown headings by
  `render()`. Numbered headings are kept and figure captions are filtered out;
  documents with more than ten level-2 headings fall back to the chapter level so
  the list stays usable.
- Each docs page renders a thin reading-progress bar and turns section headings
  into hover/focus anchor links.
- The docs index shows the engine series as a numbered path plus cards; the
  landing page reuses the same collection to render the series.
- Equations for the lessons are written as inline HTML (`<sub>`, `<sup>`) on
  purpose: they are short, and it keeps KaTeX CSS off those pages.

## Deployment

The build is fully static. Any static host works:

```bash
pnpm build
# upload dist/
```

Set `SITE_URL` (for example `SITE_URL=https://mini-llm.example.com pnpm build`)
to emit canonical and Open Graph URLs. Without it those tags are omitted rather
than pointing at localhost.

### Netlify

The repository root has a `netlify.toml` for this package:

```toml
[build]
  base = "web"
  command = "pnpm run build"
  publish = "dist"
```

`publish` is relative to `base`, so Netlify publishes `web/dist`. The file also
sets `NODE_VERSION=22` and long-lived caching for `/_astro/*`. Because the
committed configuration overrides the UI, leave the Netlify build fields empty
when connecting the repository; a deploy of the repo root without a build is what
produces Netlify's default 404 page.

Manual alternatives:

```bash
pnpm build
npx netlify deploy --prod --dir=dist   # from web/
```

or drag `web/dist` onto <https://app.netlify.com/drop>.

During a Netlify build the `URL` environment variable is available, and
`astro.config.mjs` uses it as the `site` fallback so canonical URLs point at the
deployed domain without extra configuration.
