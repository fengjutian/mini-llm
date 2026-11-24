// @ts-check
import { defineConfig } from 'astro/config';
import react from '@astrojs/react';
import { unified } from '@astrojs/markdown-remark';
import remarkMath from 'remark-math';
import rehypeKatex from 'rehype-katex';

// The repository docs contain a fair amount of LaTeX, so the site opts back
// into the unified pipeline (Astro 7 ships Sätteri by default) to use
// remark-math + rehype-katex. This also keeps Shiki highlighting available.
export default defineConfig({
  // Set SITE_URL (e.g. https://example.com) to emit canonical and Open Graph
  // absolute URLs. Netlify exposes the primary site URL as `URL` during builds,
  // so Netlify deployments get canonical tags without extra configuration.
  site: process.env.SITE_URL || process.env.URL || undefined,
  integrations: [react()],
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
