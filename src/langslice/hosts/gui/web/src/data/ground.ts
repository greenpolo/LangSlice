// The slides' own ground (the strip's `ground`: the glass of the job's middle section, as
// pictures/parts.py reads it). The window takes it for its own ground, as ABBA's
// brightfield and darkfield modes do, so the pictures are shown as their files hold them,
// never cut out of a ground of ours: bright glass makes the light look, dark glass the dark.
export type Ground = { key: string; rgb: [number, number, number]; bright: boolean };

const BRIGHT = 0.55;

export function groundOf(rgb: [number, number, number] | null): Ground | null {
  if (!rgb) return null;
  const [r, g, b] = rgb;
  const lum = (0.2126 * r + 0.7152 * g + 0.0722 * b) / 255;
  return { key: rgb.join(','), rgb, bright: lum > BRIGHT };
}
