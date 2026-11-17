// The window's motion, one family: strong curves (ease-out for what enters or leaves,
// ease-in-out for what moves on screen) and a few durations, in seconds. The same values
// are tokens.css's `--ease-*` and `--dur-*`, for what CSS animates.
export const EASE_OUT = [0.23, 1, 0.32, 1] as const;
export const EASE_IN_OUT = [0.77, 0, 0.175, 1] as const;
export const DUR = {
  quick: 0.2,   // a line of text swapped, something leaving
  enter: 0.42,  // a picture or the agent's words arriving
  morph: 0.48,  // a section's picture redrawn after the agent moved it
  layout: 0.56, // the stage rearranging its pictures
  glide: 0.82,  // a dot moving along the front-to-back line
};
// The stage's rearrangement: a spring that settles without a bounce, so pictures read as
// moved rather than thrown.
export const LAYOUT_SPRING = { type: 'spring', duration: DUR.layout, bounce: 0.08 } as const;
