/** Tiny URL-keyed ImageBitmap LRU.
 *
 * Used by the 3D viewer's PreviewBorderPlane so that dragging the AP slider
 * back-and-forth across recently-visited sections doesn't re-fetch and
 * re-decode the PNG every tick. Closes bitmaps on eviction so we don't leak
 * GPU/CPU memory.
 *
 * Bitmaps returned from `getCachedBitmap` are OWNED BY THE CACHE — callers
 * must NOT call `bmp.close()`. The cache will close them when they fall off
 * the LRU.
 */

const CACHE_MAX = 50;
const cache = new Map<string, ImageBitmap>(); // Map preserves insertion order.

export async function getCachedBitmap(url: string): Promise<ImageBitmap> {
  const hit = cache.get(url);
  if (hit !== undefined) {
    // Re-insert to mark as most-recently-used.
    cache.delete(url);
    cache.set(url, hit);
    return hit;
  }
  const blob = await (await fetch(url, { cache: "force-cache" })).blob();
  const bmp = await createImageBitmap(blob);
  cache.set(url, bmp);
  while (cache.size > CACHE_MAX) {
    const oldestKey = cache.keys().next().value as string | undefined;
    if (oldestKey === undefined) break;
    const old = cache.get(oldestKey);
    cache.delete(oldestKey);
    old?.close();
  }
  return bmp;
}
