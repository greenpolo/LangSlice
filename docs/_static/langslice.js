// Front page: point at the section and the page names the atlas region under the pointer, and lights it in both
// halves: its outline on the photograph (left) and its characters in the typed half (right).
// The hero (docs/_includes/ascii_hero.html) carries a region map, two base-62 characters per character cell:
// 0 for no region, else 2 * area + side + 1, where side tells the two hemispheres apart. The outlines on the photo
// carry the same area number (data-a). The lit characters are typed again in off-white on a layer above the
// section; the rest of the section dims.
(() => {
  const brain = document.querySelector(".ls-brain");
  const dataNode = document.getElementById("ls-brain-data");
  if (!brain || !dataNode) return;

  const data = JSON.parse(dataNode.textContent);
  const stage = brain.querySelector(".ls-brain__stage");
  const lit = brain.querySelector(".ls-brain__lit");
  const readout = document.querySelector(".ls-brain__readout");
  const lines = brain.querySelector(".ls-brain__chars").textContent.split("\n");
  const hint = readout.textContent;
  const outlines = [...brain.querySelectorAll(".ls-brain__lines path[data-a]")];

  const B62 = "0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ";
  const areaAt = new Int16Array(data.rows * data.cols).fill(-1);
  data.codes.forEach((row, r) => {
    for (let c = 0; c < data.cols; c++) {
      const code = B62.indexOf(row[2 * c]) * 62 + B62.indexOf(row[2 * c + 1]);
      if (code > 0) areaAt[r * data.cols + c] = (code - 1) >> 1;
    }
  });

  let shown = -1;
  function show(area) {
    if (area === shown) return;
    shown = area;
    outlines.forEach((path) => path.classList.toggle("is-lit", Number(path.dataset.a) === area));
    if (area < 0) {
      lit.textContent = "";
      readout.textContent = hint;
      brain.classList.remove("is-pointing");
      return;
    }
    const rows = [];
    for (let r = 0; r < data.rows; r++) {
      const line = lines[r] || "";
      let s = "";
      for (let c = 0; c < data.cols; c++) {
        s += areaAt[r * data.cols + c] === area ? line[c] || " " : " ";
      }
      rows.push(s.trimEnd());
    }
    lit.textContent = rows.join("\n");
    const name = data.areas[area];
    const acronym = document.createElement("b");
    acronym.textContent = name.a;
    const full = document.createElement("span");
    full.textContent = name.n;
    readout.replaceChildren(acronym, full);
    brain.classList.add("is-pointing");
  }

  function at(event) {
    const box = stage.getBoundingClientRect();
    const c = Math.floor(((event.clientX - box.left) / box.width) * data.cols);
    const r = Math.floor(((event.clientY - box.top) / box.height) * data.rows);
    if (c < 0 || r < 0 || c >= data.cols || r >= data.rows) return -1;
    return areaAt[r * data.cols + c];
  }

  stage.addEventListener("pointermove", (event) => show(at(event)));
  stage.addEventListener("pointerdown", (event) => show(at(event)));
  stage.addEventListener("pointerleave", () => show(-1));
})();
