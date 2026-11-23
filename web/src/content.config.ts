import { defineCollection, z } from 'astro:content';
import { glob } from 'astro/loaders';

/**
 * Pages in `src/content/docs` are generated from the repository's `docs/*.md`
 * by `scripts/sync-docs.mjs`, which also adds the frontmatter validated here.
 */
const docs = defineCollection({
  loader: glob({ pattern: '*.md', base: './src/content/docs' }),
  schema: z.object({
    title: z.string(),
    description: z.string(),
    section: z.string(),
    order: z.number(),
    source: z.string(),
  }),
});

export const collections = { docs };
