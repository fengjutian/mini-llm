# mini-llm web

The Astro site for mini-llm. Its job is to make the concepts behind the Rust code
easier to understand, not to run the model: the interactive lessons recompute
small examples in the browser, while the docs are the repository's Markdown
rendered as static pages.

## Stack

- **Astro 7** with the static output adapter (no SSR)
- **Unified** Markdown pipeline (`@astrojs/markdown-remark`) for `remark-math` +
  `rehype-katex`, because the repository docs contain a lot of LaTeX
- **No UI framework**: the three lessons are plain `<script>` blocks, so they ship
  as small vanilla TypeScript modules instead of hydrated islands
- Shiki for syntax highlighting, with separate light and dark themes

## Scripts

```bash
pnpm install
pnpm dev       # sync docs, then http://localhost:4321
pnpm build     # sync docs, then static output in dist/
pnpm preview   # serve dist/
pnpm sync      # only regenerate src/content/docs
```

`sync` / `dev` / `build` run `scripts/sync-docs.mjs` first.

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
the leading H1, `section: '其他'`), but they get no part number and appear after
the curated ones. The engine series uses `part` to show `Part 01 … 09` in the
sidebar, the docs index, and the previous/next links.

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
