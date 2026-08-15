/**
 * SEC-45 projection, asserted from the inside.
 *
 * The black-box probe ingests a capture and greps the listing. This checks the
 * same rule against the shapes that probe cannot easily produce: unknown
 * metadata keys a future capture path might add, and unknown top-level keys.
 */

import { describe, expect, it } from 'vitest';
import { PROJECTORS } from './projections.ts';

const project = PROJECTORS.url_listing;

describe('url_listing projection', () => {
  it('drops captured page content and every unreviewed metadata key', () => {
    const projected = project([
      {
        original: 'http://example.com/a',
        normalized: 'http://example.com/a',
        status: 'scraped',
        metadata: {
          title: 'A title',
          status_code: 200,
          content: 'THE ENTIRE PAGE BODY',
          description: 'meta description, also page-derived',
          og_image: 'https://cdn.example.com/x.png',
          some_field_invented_next_year: 'sensitive',
        },
      },
    ]);

    expect(projected).toEqual([
      {
        original: 'http://example.com/a',
        normalized: 'http://example.com/a',
        status: 'scraped',
        metadata: { title: 'A title', status_code: 200 },
      },
    ]);
  });

  it('drops unreviewed TOP-LEVEL keys too, not just metadata ones', () => {
    const projected = project([
      { original: 'u', normalized: 'u', status: 'scraped', metadata: {}, content: 'BODY' },
    ]) as Array<Record<string, unknown>>;
    expect(Object.hasOwn(projected[0] as object, 'content')).toBe(false);
  });

  it('keeps the reviewed metadata whitelist', () => {
    const projected = project([
      {
        original: 'u',
        normalized: 'u',
        status: 'scraped',
        metadata: {
          title: 't',
          status_code: 200,
          auth_type: 'form',
          auth_used: true,
          capture_id: 'cap-1',
          credential_scope_drop: ['other.example'],
        },
      },
    ]) as Array<{ metadata: Record<string, unknown> }>;

    expect(Object.keys(projected[0]!.metadata).sort()).toEqual([
      'auth_type',
      'auth_used',
      'capture_id',
      'credential_scope_drop',
      'status_code',
      'title',
    ]);
  });

  it('substitutes an empty object when metadata is absent or not an object', () => {
    const projected = project([
      { original: 'u', normalized: 'u', status: 'pending' },
      { original: 'v', normalized: 'v', status: 'pending', metadata: 'not-an-object' },
    ]) as Array<{ metadata: unknown }>;
    expect(projected[0]!.metadata).toEqual({});
    expect(projected[1]!.metadata).toEqual({});
  });

  it('passes a non-array payload through untouched (404 bodies, error objects)', () => {
    expect(project({ detail: 'Session not found' })).toEqual({ detail: 'Session not found' });
  });
});
