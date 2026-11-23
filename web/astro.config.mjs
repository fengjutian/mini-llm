// @ts-check
import { defineConfig } from 'astro/config';
import { unified } from '@astrojs/markdown-remark';
import remarkMath from 'remark-math';
import rehypeKatex from 'rehype-katex';

// The repository docs contain a fair amount of LaTeX, so the site opts back
// into the unified pipeline (Astro 7 ships Sätteri by default) to use
// remark-math + rehype-katex. This also keeps Shiki highlighting available.
export default defineConfig({
  // Set SITE_URL (e.g. https://example.com) to emit canonical and Open Graph
  // absolute URLs; without it the build stays portable.
  site: process.env.SITE_URL || undefined,
  markdown: {
    processor: unified({
      remarkPlugins: [remarkMath],
      rehypePlugins: [
        [
          rehypeKatex,
          {
            throwOnError: false,
            strict: 'ignore',
            output: 'htmlAndMathml',
          },
        ],
      ],
    }),
    shikiConfig: {
      themes: {
        light: 'github-light',
        dark: 'github-dark-dimmed',
      },
      wrap: true,
    },
  },
  // Astro 7 strips whitespace between inline elements by default; the docs
  // rely on ordinary HTML whitespace semantics, so keep the older behavior.
  compressHTML: true,
  devToolbar: { enabled: false },
});
