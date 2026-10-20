# Refusals and gates

## exploring

### `look('section', sections=[0, 'nope.png'])`

```json
{
 "error": "UNKNOWN_SLICE_IDS",
 "filenames": [
  "s0.png",
  "s1.png",
  "s2.png"
 ],
 "message": "No section is named '0', 'nope.png'. Sections are named by filename (or the filename without its extension, when no other section shares it). Sections have no numbers to name them by. Filenames: s0.png, s1.png, s2.png.",
 "status": "error",
 "unknown": [
  "0",
  "nope.png"
 ]
}
```

### `look('section', sections=['s0.png'], view={'mode': 'section'})`

```json
{
 "error": "UNKNOWN_ARGUMENTS",
 "message": "look: Unknown key(s) in (top level): view; accepted: mode, sections, positions_mm, channels, atlas_layers, atlas_opacity, warp, layers, depth. Nothing was done.",
 "problems": [
  {
   "accepted": [
    "mode",
    "sections",
    "positions_mm",
    "channels",
    "atlas_layers",
    "atlas_opacity",
    "warp",
    "layers",
    "depth"
   ],
   "argument": "(top level)",
   "unknown": [
    "view"
   ]
  }
 ],
 "status": "error",
 "tool": "look"
}
```

### `look('section', sections=['s0.png'], channels=['purple'])`

```json
{
 "error": "UNKNOWN_CHANNEL",
 "message": "s0.png has no channel 'purple'; its channels: gray, or 'preprocessed'",
 "status": "error"
}
```

### `zoom([0, 0, 50, 50], picture=999)`

```json
{
 "error": "UNKNOWN_PICTURE",
 "message": "There is no picture 999.",
 "picture": 999,
 "status": "error"
}
```

### `read_file('../outside.txt')`

```json
{
 "error": "BAD_PATH",
 "message": "'../outside.txt' is outside the job folder; only files inside it can be read.",
 "status": "error"
}
```

### `undo()`

```json
{
 "error": "NOTHING_TO_UNDO",
 "status": "error"
}
```

### `submit('Nothing placed.', [], [])`

```json
{
 "error": "MISSING_POSITIONS",
 "message": "3 of 3 section(s) have no position; every section needs one, damaged ones included.",
 "missing_ids": [
  "s0.png",
  "s1.png",
  "s2.png"
 ],
 "status": "error"
}
```

## placing

### `position_sections([{'id': 's0.png', 'position_mm': 0.1}])`

```json
{
 "changed": [
  {
   "damaged": false,
   "flip": false,
   "id": "s0.png",
   "position_mm": 0.1,
   "rotation_deg": 0
  }
 ],
 "cutting_angles_deg": {
  "pitch": 0.0,
  "yaw": 0.0
 },
 "interval_breaks": [],
 "n_sections": 3,
 "order": [
  "s0.png",
  "s1.png",
  "s2.png"
 ],
 "pictures": [
  {
   "caption": "positioning: 1 section below in position order at 0.10 mm, raw gray auto; atlas above at 1 position; ruler 0.0-0.2 mm from anterior (left) to posterior; all at 31.2 um/px. Sections: s0.png 0.10 mm",
   "id": 1
  }
 ],
 "status": "ok",
 "written": [
  {
   "id": "s0.png",
   "position_mm": 0.1
  }
 ]
}
```

### `position_sections([{'id': 's0.png', 'position_mm': 9.0}])`

```json
{
 "atlas_range_mm": [
  0.0,
  0.25
 ],
 "changed": [
  {
   "damaged": false,
   "flip": false,
   "id": "s0.png",
   "position_mm": 0.25,
   "rotation_deg": 0
  }
 ],
 "clamped": [
  {
   "clamped_to_mm": 0.25,
   "id": "s0.png",
   "requested_mm": 9.0
  }
 ],
 "cutting_angles_deg": {
  "pitch": 0.0,
  "yaw": 0.0
 },
 "interval_breaks": [],
 "n_sections": 3,
 "order": [
  "s0.png",
  "s1.png",
  "s2.png"
 ],
 "pictures": [
  {
   "caption": "positioning: 1 section below in position order at 0.25 mm, raw gray auto; atlas above at 1 position; ruler 0.0-0.3 mm from anterior (left) to posterior; all at 31.2 um/px. Sections: s0.png 0.25 mm",
   "id": 2
  }
 ],
 "status": "ok",
 "written": [
  {
   "id": "s0.png",
   "position_mm": 0.25
  }
 ]
}
```

### `position_sections([{'id': 's0.png', 'position_mm': 0.1, 'mm': 1}])`

```json
{
 "error": "UNKNOWN_ARGUMENTS",
 "message": "position_sections: Unknown key(s) in sections[0]: mm; accepted: id, position_mm. Nothing was done.",
 "problems": [
  {
   "accepted": [
    "id",
    "position_mm"
   ],
   "argument": "sections[0]",
   "unknown": [
    "mm"
   ]
  }
 ],
 "status": "error",
 "tool": "position_sections"
}
```

### `position_sections(cutting_angles={'pitch_deg': 1.0, 'wobble': 2})`

```json
{
 "error": "UNKNOWN_ARGUMENTS",
 "message": "position_sections: Unknown key(s) in cutting_angles: wobble; accepted: pitch_deg, yaw_deg. Nothing was done.",
 "problems": [
  {
   "accepted": [
    "pitch_deg",
    "yaw_deg"
   ],
   "argument": "cutting_angles",
   "unknown": [
    "wobble"
   ]
  }
 ],
 "status": "error",
 "tool": "position_sections"
}
```

### `mark_damage('s1.png', ['NOPE'])`

```json
{
 "error": "UNKNOWN_REGIONS",
 "message": "Unknown atlas structures: NOPE",
 "status": "error"
}
```

### `interactive_transform([{'id': 's0.png', 'rotate_quarter': 45}])`

```json
{
 "error": "NOTHING_WRITTEN",
 "results": [
  {
   "allowed": [
    0,
    90,
    180,
    270
   ],
   "error": "BAD_ROTATION",
   "id": "s0.png",
   "status": "error"
  }
 ],
 "status": "error"
}
```

### `submit('Not yet.', [], [])`

```json
{
 "error": "MISSING_POSITIONS",
 "message": "2 of 3 section(s) have no position; every section needs one, damaged ones included.",
 "missing_ids": [
  "s1.png",
  "s2.png"
 ],
 "status": "error"
}
```

## frozen

### `position_sections([{'id': 's0.png', 'position_mm': 0.3}])`

```json
{
 "error": "FROZEN",
 "ids": [
  "s0.png"
 ],
 "message": "Frozen on s0.png: position. The job keeps the value it gave there.",
 "properties": [
  "position"
 ],
 "status": "error"
}
```

### `position_sections([{'id': 's1.png', 'position_mm': 0.3}])`

```json
{
 "atlas_range_mm": [
  0.0,
  0.25
 ],
 "changed": [
  {
   "damaged": false,
   "flip": false,
   "frozen": [
    "angles",
    "affine"
   ],
   "id": "s1.png",
   "position_mm": 0.25,
   "rotation_deg": 0
  }
 ],
 "clamped": [
  {
   "clamped_to_mm": 0.25,
   "id": "s1.png",
   "requested_mm": 0.3
  }
 ],
 "cutting_angles_deg": {
  "pitch": 0.0,
  "yaw": 0.0
 },
 "interval_breaks": [],
 "n_sections": 3,
 "order": [
  "s0.png",
  "s1.png",
  "s2.png"
 ],
 "pictures": [
  {
   "caption": "positioning: 1 section below in position order at 0.25 mm, raw red, green, blue auto; atlas above at 1 position; ruler 0.0-0.3 mm from anterior (left) to posterior; all at 31.2 um/px. Sections: s1.png 0.25 mm",
   "id": 3
  }
 ],
 "status": "ok",
 "written": [
  {
   "id": "s1.png",
   "position_mm": 0.25
  }
 ]
}
```

### `position_sections(cutting_angles={'pitch_deg': 1.0, 'yaw_deg': 0.0})`

```json
{
 "error": "FROZEN",
 "ids": [
  "s0.png",
  "s1.png",
  "s2.png"
 ],
 "message": "Frozen on every section: cutting angles. The job keeps the value it gave there.",
 "properties": [
  "angles"
 ],
 "status": "error"
}
```

### `interactive_transform([{'id': 's1.png', 'rotation_deg': 3.0}])`

```json
{
 "error": "NOTHING_WRITTEN",
 "results": [
  {
   "error": "FROZEN",
   "id": "s1.png",
   "message": "Frozen on s1.png: affine (in-plane transform). The job keeps the value it gave there.",
   "properties": [
    "affine"
   ],
   "status": "error"
  }
 ],
 "status": "error"
}
```
