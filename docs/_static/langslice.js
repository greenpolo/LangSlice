// Front page: point at the typed-out section and the page names the atlas region under the pointer.
// The hero (docs/_includes/ascii_hero.html) carries a region map, two base-62 characters per character cell:
// 0 for no region, else 2 * area + side + 1, where side tells the two hemispheres apart. The hovered region's
// characters are typed again in off-white on a layer above the section; the rest of the section dims.
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

  const B62 = "0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ";
  const codes = new Uint16Array(data.rows * data.cols);
  data.codes.forEach((row, r) => {
    for (let c = 0; c < data.cols; c++) {
      codes[r * data.cols + c] = B62.indexOf(row[2 * c]) * 62 + B62.indexOf(row[2 * c + 1]);
    }
  });

  let shown = 0;
  function show(code) {
    if (code === shown) return;
    shown = code;
    if (!code) {
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
        s += codes[r * data.cols + c] === code ? line[c] || " " : " ";
      }
      rows.push(s.trimEnd());
    }
    lit.textContent = rows.join("\n");
    const area = data.areas[(code - 1) >> 1];
    const acronym = document.createElement("b");
    acronym.textContent = area.a;
    const name = document.createElement("span");
    name.textContent = area.n;
    readout.replaceChildren(acronym, name);
    brain.classList.add("is-pointing");
  }

  function at(event) {
    const box = stage.getBoundingClientRect();
    const c = Math.floor(((event.clientX - box.left) / box.width) * data.cols);
    const r = Math.floor(((event.clientY - box.top) / box.height) * data.rows);
    if (c < 0 || r < 0 || c >= data.cols || r >= data.rows) return 0;
    return codes[r * data.cols + c];
  }

  stage.addEventListener("pointermove", (event) => show(at(event)));
  stage.addEventListener("pointerdown", (event) => show(at(event)));
  stage.addEventListener("pointerleave", () => show(0));
})();
