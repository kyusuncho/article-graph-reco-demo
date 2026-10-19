/* 기사 추천 그래프 — static demo.
 *
 * Data (built by pipeline/build.py):
 *   data/meta.json    stats, categories, themes, facet table [type, name, df], tag aliases
 *   data/graph.json   per-article arrays (title, x, y, category, theme, community, facets) + edges
 *   data/details.json per-article [dek, body, form, tone, intent, season], loaded lazily
 *   data/emb128.i8    BGE-m3 embeddings, PCA to 128 dims, int8, row-major
 *
 * Recommendation (on 추천하기) = weighted sum of three channels, then MMR:
 *   graph     Forward Push (Andersen-Chung-Lang, the local approximation of Personalized
 *             PageRank) on the article-facet bipartite graph, seeded by history articles
 *             (recency-weighted) and interest tags
 *   embedding cosine to a profile vector (history + tag/category centroids) and to the
 *             closest history article
 *   interest  category and tag match
 */
(() => {
  "use strict";

  const $ = (sel) => document.querySelector(sel);
  const esc = (s) => String(s).replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
  const keyOf = (s) => s.normalize("NFKC").toLowerCase().replace(/[^\p{L}\p{N}]/gu, "");
  const truncate = (s, n) => (s.length > n ? s.slice(0, n - 1) + "…" : s);

  const TYPE_LABEL = { topic: "topic", person: "person", brand: "brand", place: "place", org: "org", need: "need", activity: "activity", life_context: "life context" };
  const ALPHA = 0.15;          // forward-push restart (teleport) probability
  const EPS = 2e-6;            // forward-push residual threshold (per unit degree)
  const RECENCY = 0.8;         // history weight decay per step back in time
  const THEME_CAP = 2;         // at most two recommendations from one theme
  const TOP_N = 5;

  const state = {
    interests: new Set(),      // category indices
    tags: new Set(),           // topic facet indices
    history: [],               // article indices, oldest first
    recs: [],
    selected: null,
    hovered: null,
    weights: { graph: 0.45, emb: 0.35, interest: 0.2 },
    diversity: 0.3,
  };

  let M, G, D = null, N, F, DIM, emb;
  let artFacets, artW, artWSum, facetArts, facetW, topicByName, tagIndex, catMembers;
  let groups, catGroup, palette, faded, edgeColor, edgeHover, surface, ink;
  let graph, renderer, neighbours, cachedFocus = new Set(), labelSets;

  // ------------------------------------------------------------------ load
  async function load() {
    const [meta, g, buf] = await Promise.all([
      fetch("data/meta.json").then((r) => r.json()),
      fetch("data/graph.json").then((r) => r.json()),
      fetch("data/emb128.i8").then((r) => r.arrayBuffer()),
    ]);
    M = meta; G = g; N = G.ids.length; F = M.facets.length;
    const raw = new Int8Array(buf);
    DIM = raw.length / N;
    emb = new Float32Array(raw.length);
    for (let i = 0; i < N; i++) {
      let s = 0;
      for (let k = 0; k < DIM; k++) { const v = raw[i * DIM + k]; s += v * v; }
      const inv = 1 / Math.sqrt(s || 1);
      for (let k = 0; k < DIM; k++) emb[i * DIM + k] = raw[i * DIM + k] * inv;
    }
    prepareFacets();
    fetch("data/details.json").then((r) => r.json()).then((d) => {
      D = d;
      renderRecs(lastInfo);
      if (state.selected != null) renderDetail();
    });
  }

  function prepareFacets() {
    const [dfMin, dfMax] = M.stats.df_window;
    facetW = new Float32Array(F);
    facetArts = Array.from({ length: F }, () => []);
    topicByName = new Map();
    M.facets.forEach(([type, name, df], f) => {
      if (df >= dfMin && df <= dfMax) facetW[f] = M.type_weight[type] * Math.log(N / df);
      if (type === "topic") topicByName.set(name, f);
    });
    artFacets = []; artW = []; artWSum = new Float32Array(N);
    for (let i = 0; i < N; i++) {
      const fs = G.facets[i].filter((f) => facetW[f] > 0);
      const ws = fs.map((f) => facetW[f]);
      artFacets.push(fs); artW.push(ws);
      artWSum[i] = ws.reduce((a, b) => a + b, 0);
      for (const f of fs) facetArts[f].push(i);
    }
    tagIndex = new Map();
    for (const t of M.tags) {
      tagIndex.set(keyOf(t.tag), { tag: t.tag, via: null });
      for (const a of t.aliases) if (!tagIndex.has(keyOf(a))) tagIndex.set(keyOf(a), { tag: t.tag, via: a });
    }
    groups = [...new Set(M.categories.map((c) => c.group))];
    catGroup = M.categories.map((c) => groups.indexOf(c.group));
    catMembers = M.categories.map(() => []);
    for (let i = 0; i < N; i++) catMembers[G.category[i]].push(i);
  }

  // ------------------------------------------------------------------ theme colours
  function hex(c) { const m = c.replace("#", ""); return [0, 2, 4].map((i) => parseInt(m.slice(i, i + 2), 16)); }
  function mix(a, b, t) {
    const x = hex(a), y = hex(b);
    return "#" + x.map((v, i) => Math.round(v * t + y[i] * (1 - t)).toString(16).padStart(2, "0")).join("");
  }
  function luminance(c) { const [r, g, b] = hex(c); return (0.299 * r + 0.587 * g + 0.114 * b) / 255; }
  function readTheme() {
    const cs = getComputedStyle(document.documentElement);
    const v = (n) => cs.getPropertyValue(n).trim();
    surface = v("--surface-1");
    ink = v("--text-primary");
    palette = [0, 1, 2, 3, 4].map((k) => v(`--g${k}`));
    faded = palette.map((c) => mix(c, surface, 0.2));
    const dark = luminance(surface) < 0.5;
    edgeColor = dark ? "rgba(220,218,210,0.07)" : "rgba(60,58,52,0.09)";
    edgeHover = dark ? "rgba(240,239,236,0.45)" : "rgba(11,11,11,0.35)";
  }

  // ------------------------------------------------------------------ graph
  function buildGraph() {
    graph = new graphology.Graph({ type: "undirected", multi: true });
    const deg = new Uint16Array(N);
    for (const [a, b] of G.edges) { deg[a]++; deg[b]++; }
    for (let i = 0; i < N; i++) {
      graph.addNode(String(i), {
        x: G.x[i], y: G.y[i],
        size: 1.3 + Math.sqrt(deg[i]) * 0.45,
        color: palette[catGroup[G.category[i]]],
        label: G.titles[i],
      });
    }
    neighbours = Array.from({ length: N }, () => []);
    G.edges.forEach(([a, b, w], k) => {
      graph.addEdgeWithKey(`e${k}`, String(a), String(b), { size: 0.35 + w * 0.6, color: edgeColor });
      neighbours[a].push(b); neighbours[b].push(a);
    });

    const SigmaCtor = Sigma.Sigma || Sigma.default || Sigma;
    renderer = new SigmaCtor(graph, $("#graph"), {
      zIndex: true,
      hideEdgesOnMove: true,
      renderLabels: true,
      labelRenderedSizeThreshold: 7,
      labelDensity: 0.6,
      labelGridCellSize: 140,
      labelFont: getComputedStyle(document.body).fontFamily,
      labelSize: 12,
      labelWeight: "500",
      minCameraRatio: 0.03,
      maxCameraRatio: 1.6,
      defaultDrawNodeLabel: drawLabel,
      defaultDrawNodeHover: drawHover,
      nodeReducer,
      edgeReducer,
    });

    renderer.on("enterNode", ({ node }) => { state.hovered = +node; showTooltip(+node); renderer.refresh({ skipIndexation: true }); });
    renderer.on("leaveNode", () => { state.hovered = null; hideTooltip(); renderer.refresh({ skipIndexation: true }); });
    renderer.on("clickNode", ({ node }) => select(+node, false));
    renderer.on("clickStage", () => { if (state.selected != null) { state.selected = null; renderDetail(); refreshGraph(); } });
    renderer.on("afterRender", placeClusterLabels);
    $("#graph").addEventListener("mousemove", moveTooltip);
  }

  function drawLabel(ctx, data, settings) {
    if (!data.label) return;
    const size = data.forceLabel ? 13 : settings.labelSize;
    ctx.font = `${data.forceLabel ? 700 : 500} ${size}px ${settings.labelFont}`;
    const x = data.x + data.size + 4, y = data.y + size / 3;
    ctx.lineJoin = "round"; ctx.lineWidth = 4; ctx.strokeStyle = surface;
    ctx.strokeText(data.label, x, y);
    ctx.fillStyle = ink;
    ctx.fillText(data.label, x, y);
  }
  function drawHover(ctx, data, settings) {
    ctx.beginPath();
    ctx.arc(data.x, data.y, data.size + 3, 0, Math.PI * 2);
    ctx.lineWidth = 2; ctx.strokeStyle = ink; ctx.stroke();
    ctx.beginPath();
    ctx.arc(data.x, data.y, data.size + 1, 0, Math.PI * 2);
    ctx.lineWidth = 2; ctx.strokeStyle = surface; ctx.stroke();
    if (data.label) drawLabel(ctx, { ...data, forceLabel: true }, settings);
  }

  function isActive() { return state.history.length > 0 || state.recs.length > 0 || state.selected != null; }

  function focusSet() {
    const s = new Set(state.history);
    state.recs.forEach((r) => s.add(r.i));
    if (state.selected != null) { s.add(state.selected); neighbours[state.selected].forEach((j) => s.add(j)); }
    return s;
  }

  function nodeReducer(node, data) {
    const i = +node;
    const res = { ...data };
    const g = catGroup[G.category[i]];
    const hov = state.hovered;
    if (hov != null) {
      if (i !== hov && !neighbours[hov].includes(i) && !cachedFocus.has(i)) { res.color = faded[g]; res.label = ""; res.zIndex = 0; }
      else res.zIndex = 2;
    } else if (isActive()) {
      if (!cachedFocus.has(i)) { res.color = faded[g]; res.label = ""; res.zIndex = 0; }
      else res.zIndex = 1;
    }
    // History and recommendation labels are drawn by the HTML overlay
    // (placeFocusLabels), which can keep them from overlapping.
    if (state.history.includes(i)) {
      Object.assign(res, { highlighted: true, zIndex: 3, size: Math.max(res.size, 5.5), color: palette[g], label: "" });
    }
    const r = state.recs.findIndex((x) => x.i === i);
    if (r >= 0) {
      Object.assign(res, { highlighted: true, zIndex: 4, size: Math.max(res.size, 6.5), color: palette[g], label: "" });
    }
    if (i === state.selected) Object.assign(res, { highlighted: true, forceLabel: true, zIndex: 5, size: Math.max(res.size, 6), color: palette[g] });
    return res;
  }

  function edgeReducer(edge, data) {
    const res = { ...data };
    if (data.path) { res.color = ink; res.size = 1.6; res.zIndex = 3; return res; }
    const [s, t] = graph.extremities(edge);
    const a = +s, b = +t;
    const hov = state.hovered != null ? state.hovered : state.selected;
    if (hov != null) {
      if (a === hov || b === hov) { res.color = edgeHover; res.size = 1; res.zIndex = 2; }
      else res.hidden = true;
    } else if (state.history.length || state.recs.length) {
      res.hidden = !(cachedFocus.has(a) && cachedFocus.has(b));
      if (!res.hidden) { res.color = edgeHover; res.size = 0.8; }
    }
    return res;
  }

  function refreshGraph() {
    cachedFocus = focusSet();
    graph.forEachEdge((e, attr) => { if (attr.path) graph.dropEdge(e); });
    state.recs.forEach((r, k) => {
      if (r.anchor != null) graph.addEdgeWithKey(`p${k}`, String(r.anchor), String(r.i), { path: true, size: 1.6, color: ink });
    });
    renderer.refresh();
  }

  // ------------------------------------------------------------------ cluster labels
  function prepareLabels() {
    const median = (arr) => { const s = [...arr].sort((a, b) => a - b); return s[s.length >> 1]; };
    const byTheme = new Map();
    for (let i = 0; i < N; i++) (byTheme.get(G.theme[i]) || byTheme.set(G.theme[i], []).get(G.theme[i])).push(i);
    const themeLabels = [...byTheme].map(([t, ms]) => ({
      text: M.themes[t].name, cat: M.themes[t].category, weight: ms.length,
      x: median(ms.map((i) => G.x[i])), y: median(ms.map((i) => G.y[i])),
    }));
    // A category may be split across the layout; label it where most of its articles sit.
    const catLabels = M.categories.flatMap((c, ci) => {
      const members = catMembers[ci];
      let best = null;
      for (const l of themeLabels.filter((l) => l.cat === ci)) {
        const near = members.filter((i) => Math.hypot(G.x[i] - l.x, G.y[i] - l.y) < 0.15).length;
        if (!best || near > best.near) best = { near, x: l.x, y: l.y };
      }
      return best ? [{ text: c.name, x: best.x, y: best.y, weight: members.length }] : [];
    });
    labelSets = {
      cat: catLabels.sort((a, b) => b.weight - a.weight),
      theme: themeLabels.sort((a, b) => b.weight - a.weight),
    };
  }

  function placeClusterLabels() {
    const box = $("#clusterLabels");
    if (!labelSets) return;
    const ratio = renderer.getCamera().ratio;
    const set = ratio > 0.42 ? labelSets.cat : ratio > 0.1 ? labelSets.theme : [];
    const { width, height } = renderer.getDimensions();
    const dim = isActive() || state.hovered != null;
    const placed = [];
    let html = "";
    for (const l of set) {
      const p = renderer.graphToViewport({ x: l.x, y: l.y });
      if (p.x < 0 || p.y < 0 || p.x > width || p.y > height) continue;
      const w = l.text.length * 12 + 10, h = 20;
      const r = { x0: p.x - w / 2, x1: p.x + w / 2, y0: p.y - h / 2, y1: p.y + h / 2 };
      if (placed.some((q) => r.x0 < q.x1 && r.x1 > q.x0 && r.y0 < q.y1 && r.y1 > q.y0)) continue;
      placed.push(r);
      html += `<div class="cluster-label" style="left:${p.x}px;top:${p.y}px;opacity:${dim ? 0.3 : 0.95}">${esc(l.text)}</div>`;
    }
    box.innerHTML = html + focusLabels(width, height);
  }

  // Labels for history (H) and recommendations (1-5): placed to the right of
  // the node and pushed down until they clear every label already placed.
  function focusLabels(width, height) {
    const items = [
      ...state.recs.map((r, k) => ({ i: r.i, tag: String(k + 1), text: truncate(G.titles[r.i], 20) })),
      ...state.history.map((i) => ({ i, tag: "H", text: truncate(G.titles[i], 14) })),
    ];
    const placed = [];
    let html = "";
    for (const it of items) {
      const d = renderer.getNodeDisplayData(String(it.i));
      if (!d) continue;
      const p = renderer.framedGraphToViewport({ x: d.x, y: d.y });
      if (p.x < -50 || p.y < -50 || p.x > width + 50 || p.y > height + 50) continue;
      const w = 26 + it.text.length * 12.5, h = 20;
      let x = p.x + 10, y = p.y - h / 2;
      for (let tries = 0; tries < 12; tries++) {
        const hit = placed.find((q) => x < q.x + q.w && x + w > q.x && y < q.y + q.h && y + h > q.y);
        if (!hit) break;
        y = hit.y + hit.h + 2;
      }
      placed.push({ x, y, w, h });
      const leader = Math.abs(y + h / 2 - p.y) > 4
        ? `<svg class="leader" style="left:${p.x}px;top:${Math.min(p.y, y + h / 2)}px;width:12px;height:${Math.abs(y + h / 2 - p.y)}px"><line x1="0" y1="${y + h / 2 > p.y ? 0 : "100%"}" x2="10" y2="${y + h / 2 > p.y ? "100%" : 0}"/></svg>`
        : "";
      html += `${leader}<div class="focus-label${it.tag === "H" ? " hist" : ""}" style="left:${x}px;top:${y}px"><b>${it.tag}</b>${esc(it.text)}</div>`;
    }
    return html;
  }

  // ------------------------------------------------------------------ tooltip
  let mouse = { x: 0, y: 0 };
  function moveTooltip(e) {
    const rect = $("#graph").getBoundingClientRect();
    mouse = { x: e.clientX - rect.left, y: e.clientY - rect.top };
    if ($("#tooltip").style.display === "block") positionTooltip();
  }
  function positionTooltip() {
    const tip = $("#tooltip");
    const pane = $("#graph").getBoundingClientRect();
    let x = mouse.x + 14, y = mouse.y + 14;
    if (x + tip.offsetWidth > pane.width - 8) x = mouse.x - tip.offsetWidth - 14;
    if (y + tip.offsetHeight > pane.height - 8) y = mouse.y - tip.offsetHeight - 14;
    tip.style.left = x + "px"; tip.style.top = y + "px";
  }
  function showTooltip(i) {
    const tip = $("#tooltip");
    const c = G.category[i];
    tip.innerHTML = `<div class="t-title">${esc(G.titles[i])}</div>
      <div class="t-meta"><span class="dot" style="background:${palette[catGroup[c]]}"></span>${esc(M.categories[c].name)} · ${esc(M.themes[G.theme[i]].name)}</div>`;
    tip.style.display = "block";
    positionTooltip();
  }
  function hideTooltip() { $("#tooltip").style.display = "none"; }

  // ------------------------------------------------------------------ recommendation
  function forwardPush(seeds) {
    // Forward push (Andersen-Chung-Lang) on articles [0, N) and facets [N, N+F).
    const p = new Float64Array(N + F), r = new Float64Array(N + F);
    const inQ = new Uint8Array(N + F);
    const queue = [];
    const deg = (u) => Math.max(1, u < N ? artFacets[u].length : facetArts[u - N].length);
    const enqueue = (u) => { if (!inQ[u] && r[u] >= EPS * deg(u)) { inQ[u] = 1; queue.push(u); } };
    for (const [u, m] of seeds) r[u] += m;
    for (const u of seeds.keys()) enqueue(u);
    let pushes = 0;
    while (queue.length) {
      const u = queue.pop(); inQ[u] = 0;
      const ru = r[u];
      if (ru < EPS * deg(u)) continue;
      pushes++;
      p[u] += ALPHA * ru; r[u] = 0;
      const rest = (1 - ALPHA) * ru;
      if (u < N) {
        const fs = artFacets[u], ws = artW[u], tot = artWSum[u];
        if (!fs.length) { p[u] += rest; continue; }
        for (let k = 0; k < fs.length; k++) { const v = N + fs[k]; r[v] += rest * ws[k] / tot; enqueue(v); }
      } else {
        const arts = facetArts[u - N];
        const share = rest / arts.length;
        for (const v of arts) { r[v] += share; enqueue(v); }
      }
    }
    return { p, pushes };
  }

  const centroidCache = new Map();
  function centroid(key, members) {
    if (centroidCache.has(key)) return centroidCache.get(key);
    const v = new Float32Array(DIM);
    for (const i of members) for (let k = 0; k < DIM; k++) v[k] += emb[i * DIM + k];
    normalise(v);
    centroidCache.set(key, v);
    return v;
  }
  function normalise(v) { let s = 0; for (const x of v) s += x * x; s = Math.sqrt(s) || 1; for (let k = 0; k < v.length; k++) v[k] /= s; return v; }
  function dotRow(i, v) { let s = 0; const o = i * DIM; for (let k = 0; k < DIM; k++) s += emb[o + k] * v[k]; return s; }
  function cosRows(i, j) { let s = 0; const a = i * DIM, b = j * DIM; for (let k = 0; k < DIM; k++) s += emb[a + k] * emb[b + k]; return s; }

  function historyWeights() {
    const H = state.history.length;
    const w = state.history.map((_, k) => Math.pow(RECENCY, H - 1 - k));
    const s = w.reduce((a, b) => a + b, 0) || 1;
    return w.map((x) => x / s);
  }

  function recommend() {
    const t0 = performance.now();
    const hist = state.history, hw = historyWeights();
    const tags = [...state.tags], cats = [...state.interests];
    const histSet = new Set(hist);

    // graph channel: history articles and interest tags seed the walk
    const seeds = new Map();
    hist.forEach((i, k) => seeds.set(i, (seeds.get(i) || 0) + hw[k]));
    tags.forEach((f) => seeds.set(N + f, (seeds.get(N + f) || 0) + (hist.length ? 0.6 : 1) / tags.length));
    const graphOn = seeds.size > 0;
    const { p, pushes } = graphOn ? forwardPush(seeds) : { p: null, pushes: 0 };

    // embedding channel: one profile vector from history, tag centroids and category centroids
    const profile = new Float32Array(DIM);
    const add = (v, w) => { for (let d = 0; d < DIM; d++) profile[d] += w * v[d]; };
    hist.forEach((i, k) => add(emb.subarray(i * DIM, (i + 1) * DIM), hw[k]));
    tags.forEach((f) => add(centroid("f" + f, facetArts[f]), (hist.length ? 0.4 : 1) / tags.length));
    cats.forEach((c) => add(centroid("c" + c, catMembers[c]), (hist.length || tags.length ? 0.25 : 1) / cats.length));
    const embOn = hist.length > 0 || tags.length > 0 || cats.length > 0;
    normalise(profile);

    const interestOn = cats.length > 0 || tags.length > 0;
    if (!graphOn && !embOn) { state.recs = []; return { ms: 0, pushes: 0 }; }

    const catSet = new Set(cats), tagSet = new Set(tags);
    const cand = [];
    for (let i = 0; i < N; i++) {
      if (histSet.has(i)) continue;
      const e = dotRow(i, profile);
      let maxH = -1, maxHi = null;
      for (const h of hist) { const c = cosRows(i, h); if (c > maxH) { maxH = c; maxHi = h; } }
      let interest = 0;
      if (interestOn) {
        const parts = [];
        if (cats.length) parts.push(catSet.has(G.category[i]) ? 1 : 0);
        if (tags.length) {
          let hits = 0; for (const f of G.facets[i]) if (tagSet.has(f)) hits++;
          parts.push(Math.min(1, hits / Math.min(2, tags.length)));
        }
        interest = parts.reduce((a, b) => a + b, 0) / parts.length;
      }
      cand.push({ i, g: graphOn ? p[i] : 0, e: hist.length ? 0.6 * e + 0.4 * maxH : e, interest, embAnchor: maxHi, embAnchorSim: maxH });
    }

    // put the channels on one scale over the candidate set
    let gMax = 0; for (const c of cand) if (c.g > gMax) gMax = c.g;
    const es = cand.map((c) => c.e).sort((a, b) => a - b);
    const eMed = es[es.length >> 1], eMax = es[es.length - 1];
    const w = state.weights;
    const wg = graphOn ? w.graph : 0, we = w.emb, wi = interestOn ? w.interest : 0;
    const wsum = wg + we + wi || 1;
    for (const c of cand) {
      const gn = gMax ? c.g / gMax : 0;
      const en = Math.max(0, (c.e - eMed) / ((eMax - eMed) || 1));
      c.parts = { graph: (wg / wsum) * gn, emb: (we / wsum) * en, interest: (wi / wsum) * c.interest };
      c.score = c.parts.graph + c.parts.emb + c.parts.interest;
    }
    cand.sort((a, b) => b.score - a.score);
    const pool = cand.slice(0, 300);

    // MMR re-rank with a theme cap
    const lambda = 1 - state.diversity;
    const top = pool[0].score || 1;
    const picked = [], themeCount = new Map();
    while (picked.length < TOP_N && pool.length) {
      let best = -Infinity, bestK = -1;
      for (let k = 0; k < pool.length; k++) {
        const c = pool[k];
        if ((themeCount.get(G.theme[c.i]) || 0) >= THEME_CAP) continue;
        let red = 0;
        for (const s of picked) red = Math.max(red, cosRows(c.i, s.i));
        const v = lambda * (c.score / top) - (1 - lambda) * Math.max(0, red);
        if (v > best) { best = v; bestK = k; }
      }
      if (bestK < 0) break;
      const c = pool.splice(bestK, 1)[0];
      themeCount.set(G.theme[c.i], (themeCount.get(G.theme[c.i]) || 0) + 1);
      picked.push(c);
    }
    state.recs = picked.map((c) => ({ ...c, ...explain(c, catSet, tagSet) }));
    return { ms: performance.now() - t0, pushes };
  }

  function explain(c, catSet, tagSet) {
    const why = { shared: null, similar: null, interest: [] };
    let anchor = null;
    // the history article that shares the most facet weight with this one
    const mine = new Set(G.facets[c.i]);
    let bestH = null, bestScore = 0, bestFacets = [];
    for (const h of state.history) {
      const shared = G.facets[h].filter((f) => mine.has(f) && facetW[f] > 0);
      const s = shared.reduce((a, f) => a + facetW[f], 0);
      if (s > bestScore) { bestScore = s; bestH = h; bestFacets = shared; }
    }
    if (bestH != null) {
      bestFacets.sort((a, b) => facetW[b] - facetW[a]);
      why.shared = { h: bestH, facets: bestFacets.slice(0, 4) };
      anchor = bestH;
    }
    if (c.embAnchor != null && c.embAnchor !== bestH) {
      why.similar = { h: c.embAnchor, sim: c.embAnchorSim };
      if (anchor == null) anchor = c.embAnchor;
    }
    if (catSet.has(G.category[c.i])) why.interest.push(["분야", M.categories[G.category[c.i]].name]);
    for (const f of G.facets[c.i]) if (tagSet.has(f)) why.interest.push(["태그", M.facets[f][1]]);
    return { why, anchor };
  }

  // ------------------------------------------------------------------ rendering: side panel
  function renderPresets() {
    $("#presets").innerHTML = PRESETS.map((p, k) => `<button class="chip" type="button" data-k="${k}" title="${esc(p.note)}">${esc(p.name)}</button>`).join("");
    $("#presets").querySelectorAll("button").forEach((b) => b.addEventListener("click", () => applyPreset(PRESETS[+b.dataset.k])));
  }

  function renderInterests() {
    $("#catChips").innerHTML = groups.map((g, gi) => {
      const cats = M.categories.map((c, ci) => ({ c, ci })).filter(({ c }) => c.group === g);
      return `<div class="cat-group"><div class="cat-group-name"><span class="dot" style="background:${palette[gi]}"></span>${esc(g)}</div>
        <div class="chips">${cats.map(({ c, ci }) => `<button class="chip" type="button" data-c="${ci}" aria-pressed="${state.interests.has(ci)}">${esc(c.name)}</button>`).join("")}</div></div>`;
    }).join("");
    $("#catChips").querySelectorAll("button").forEach((b) => b.addEventListener("click", () => {
      const c = +b.dataset.c;
      state.interests.has(c) ? state.interests.delete(c) : state.interests.add(c);
      update();
    }));
    $("#tagChips").innerHTML = [...state.tags].map((f) =>
      `<span class="chip" aria-pressed="true">${esc(M.facets[f][1])}<button class="x" type="button" data-f="${f}" aria-label="${esc(M.facets[f][1])} 삭제">×</button></span>`).join("");
    $("#tagChips").querySelectorAll("button").forEach((b) => b.addEventListener("click", () => { state.tags.delete(+b.dataset.f); update(); }));
    $("#interestCount").textContent = state.interests.size + state.tags.size ? `${state.interests.size}개 분야 · ${state.tags.size}개 태그` : "";
  }

  function renderHistory() {
    const list = $("#historyList");
    $("#historyCount").textContent = state.history.length ? `${state.history.length}개` : "";
    if (!state.history.length) { list.innerHTML = `<li class="empty">아직 읽은 기사가 없습니다.</li>`; return; }
    list.innerHTML = [...state.history].reverse().map((i) => `<li>
        <span class="dot" style="background:${palette[catGroup[G.category[i]]]}"></span>
        <span class="h-title" data-i="${i}" title="${esc(G.titles[i])}">${esc(G.titles[i])}</span>
        <span class="h-cat">${esc(M.categories[G.category[i]].name)}</span>
        <button class="x" type="button" data-i="${i}" aria-label="history에서 삭제">×</button></li>`).join("");
    list.querySelectorAll(".h-title").forEach((el) => el.addEventListener("click", () => select(+el.dataset.i, true)));
    list.querySelectorAll(".x").forEach((el) => el.addEventListener("click", () => removeHistory(+el.dataset.i)));
  }

  function facetChip(f, hit) {
    const [type, name] = M.facets[f];
    return `<span class="chip tag${hit ? " hit" : ""}"><b>${TYPE_LABEL[type]}</b>${esc(name)}</span>`;
  }

  let lastInfo = { ms: 0, pushes: 0 };
  function renderRecs(info) {
    const box = $("#recList");
    if (!state.recs.length) {
      box.innerHTML = hasInputs()
        ? `<p class="empty">설정이 준비되었습니다. 위의 <b>추천하기</b>를 누르면 추천 5개가 나타납니다.</p>`
        : `<p class="empty">관심 분야를 고르거나 기사를 history에 추가한 뒤 <b>추천하기</b>를 눌러 주세요. 위의 프로필 예시로 바로 시작할 수도 있습니다.</p>`;
      $("#recTiming").textContent = "";
      return;
    }
    $("#recTiming").textContent = `${info.ms.toFixed(0)}ms · Forward Push ${info.pushes.toLocaleString()}회`;
    const meter = (label, v) => `<span>${label}</span><span class="meter"><i style="width:${Math.min(100, v * 100).toFixed(0)}%"></i></span><output>${v.toFixed(2)}</output>`;
    box.innerHTML = state.recs.map((r, k) => {
      const c = G.category[r.i];
      const lines = [];
      if (r.why.shared) lines.push(`<div>history <q>${esc(truncate(G.titles[r.why.shared.h], 24))}</q>와 공유한 facet</div><div class="chips">${r.why.shared.facets.map((f) => facetChip(f, true)).join("")}</div>`);
      if (r.why.similar) lines.push(`<div>history <q>${esc(truncate(G.titles[r.why.similar.h], 24))}</q>와 embedding 유사도 ${r.why.similar.sim.toFixed(2)}</div>`);
      if (r.why.interest.length) lines.push(`<div>관심 ${r.why.interest.map(([t, v]) => `${t} <b>${esc(v)}</b>`).join(", ")}와 일치</div>`);
      return `<button class="rec" type="button" data-i="${r.i}">
        <span class="rank">${k + 1}</span>
        <span>
          <div class="r-title">${esc(G.titles[r.i])}</div>
          <div class="r-meta"><span class="dot" style="background:${palette[catGroup[c]]}"></span>${esc(M.categories[c].name)} · ${esc(M.themes[G.theme[r.i]].name)}</div>
          ${D ? `<div class="r-dek">${esc(D[r.i][0])}</div>` : ""}
          <div class="why">${lines.join("")}</div>
          <div class="meters">${meter("forward push", r.parts.graph)}${meter("embedding", r.parts.emb)}${meter("관심 분야", r.parts.interest)}</div>
        </span></button>`;
    }).join("");
    box.querySelectorAll(".rec").forEach((el) => el.addEventListener("click", () => select(+el.dataset.i, true)));
  }

  function renderDetail() {
    const box = $("#detail");
    const i = state.selected;
    if (i == null) { box.classList.remove("open"); return; }
    const c = G.category[i];
    const inHist = state.history.includes(i);
    const histFacets = new Set(state.history.flatMap((h) => G.facets[h]));
    const byType = new Map();
    for (const f of G.facets[i]) { const t = M.facets[f][0]; (byType.get(t) || byType.set(t, []).get(t)).push(f); }
    const facetRows = M.facet_types.filter((t) => byType.has(t)).map((t) =>
      `<dt>${TYPE_LABEL[t]}</dt><dd>${byType.get(t).map((f) => `<span class="chip tag${histFacets.has(f) && !inHist ? " hit" : ""}">${esc(M.facets[f][1])}</span>`).join("")}</dd>`).join("");
    const d = D ? D[i] : null;
    let commSize = 0; for (const x of G.community) if (x === G.community[i]) commSize++;
    const rank = state.recs.findIndex((r) => r.i === i);
    box.innerHTML = `
      <div class="d-head"><h3>${esc(G.titles[i])}</h3><button class="btn ghost small" type="button" id="detailClose" aria-label="닫기">✕</button></div>
      <div class="d-meta"><span class="dot" style="background:${palette[catGroup[c]]}"></span><span>${esc(M.categories[c].name)} · ${esc(M.themes[G.theme[i]].name)} · ${G.published[i]}${d ? ` · ${esc(d[2])}` : ""}</span></div>
      ${rank >= 0 ? `<p class="hint">추천 ${rank + 1}위</p>` : ""}
      ${d ? `<p class="d-dek">${esc(d[0])}</p><div class="d-body">${d[1].split("\n").filter(Boolean).map((p) => `<p>${esc(p)}</p>`).join("")}</div>` : `<p class="hint">본문을 불러오는 중…</p>`}
      <div class="d-actions">
        <button class="btn small" type="button" id="detailHist">${inHist ? "history에서 빼기" : "history에 추가"}</button>
        <span class="hint" style="align-self:center">이웃 ${neighbours[i].length}개 · Leiden community ${G.community[i]} (${commSize}개 기사)</span>
      </div>
      <h4 class="d-sub">Normalized facet</h4>
      <dl class="facet-grid">${facetRows}</dl>
      ${d && d[6] ? rawFacetBlock(d[6]) : ""}`;
    box.classList.add("open");
    $("#detailClose").addEventListener("click", () => { state.selected = null; renderDetail(); refreshGraph(); });
    $("#detailHist").addEventListener("click", () => (inHist ? removeHistory(i) : addHistory(i)));
  }

  function rawFacetBlock(rf) {
    const pair = ([raw, norm]) => raw === norm
      ? `<li><span class="raw">${esc(raw)}</span></li>`
      : `<li><span class="raw">${esc(raw)}</span><span class="arrow">→</span><span class="norm">${esc(norm)}</span></li>`;
    const plain = (xs) => xs.map((x) => `<li><span class="raw">${esc(x)}</span></li>`).join("");
    return `<h4 class="d-sub">LLM raw facet → normalized</h4>
      <dl class="facet-grid raw-grid">
        <dt>topic</dt><dd><ul>${rf.topics.map(pair).join("")}</ul></dd>
        <dt>entity</dt><dd><ul>${rf.entities.map(pair).join("")}</ul></dd>
        <dt>activity</dt><dd><ul>${plain(rf.activity_frame)}</ul></dd>
        <dt>need</dt><dd><ul>${plain(rf.underlying_need)}</ul></dd>
        <dt>life context</dt><dd><ul>${plain(rf.life_context)}</ul></dd>
      </dl>`;
  }

  function renderLegend() {
    const rows = groups.map((g, gi) => {
      const cats = M.categories.filter((c) => c.group === g).map((c) => c.name).join(" · ");
      return `<div class="legend-row"><span class="dot" style="background:${palette[gi]}"></span><b>${esc(g)}</b><span>${esc(cats)}</span></div>`;
    }).join("");
    $("#legend").innerHTML = `<div class="legend-title">색 = 카테고리 그룹 · 위치 = facet·embedding 유사도</div>${rows}
      <div class="key-row"><span><span class="ring"></span>history</span><span><span class="ring num">1</span>추천 순위</span><span>굵은 선 = 추천 근거 (history → 추천)</span></div>`;
  }

  // ------------------------------------------------------------------ state changes
  // Editing interests or history only redraws the inputs and clears stale
  // results; recommendations are computed when the reader presses 추천하기.
  let pending = false;
  function update() {
    state.recs = [];
    if (pending) return;
    pending = true;
    requestAnimationFrame(() => {
      pending = false;
      renderAll();
    });
  }

  function renderAll() {
    renderInterests();
    renderHistory();
    renderRecs(lastInfo);
    renderCta();
    if (state.selected != null) renderDetail();
    refreshGraph();
  }

  function hasInputs() { return state.history.length + state.interests.size + state.tags.size > 0; }

  function runRecommend(moveCamera) {
    if (!hasInputs()) return;
    lastInfo = recommend();
    renderAll();
    if (moveCamera) {
      requestAnimationFrame(() => requestAnimationFrame(fitToProfile));
      $("#recList").closest(".card").scrollIntoView({ behavior: "smooth", block: "nearest" });
    }
  }

  function renderCta() {
    const btn = $("#recommendBtn");
    btn.disabled = !hasInputs();
    const parts = [];
    if (state.interests.size) parts.push(`관심 분야 ${state.interests.size}개`);
    if (state.tags.size) parts.push(`태그 ${state.tags.size}개`);
    if (state.history.length) parts.push(`history ${state.history.length}개`);
    $("#recommendHint").textContent = parts.length
      ? `${parts.join(" · ")}로 추천합니다`
      : "관심 분야나 history를 먼저 설정해 주세요";
  }

  function addHistory(i) { state.history = state.history.filter((x) => x !== i).concat(i); update(); }
  function removeHistory(i) { state.history = state.history.filter((x) => x !== i); update(); }

  function select(i, moveCamera) {
    state.selected = i;
    renderDetail();
    refreshGraph();
    if (moveCamera) focusNode(i);
  }

  function focusNode(i) {
    const d = renderer.getNodeDisplayData(String(i));
    const cam = renderer.getCamera();
    if (d) cam.animate({ x: d.x, y: d.y, ratio: Math.min(cam.ratio, 0.22) }, { duration: 500 });
  }

  function fitToProfile() {
    const ids = [...state.history, ...state.recs.map((r) => r.i)];
    if (!ids.length) return;
    const pts = ids.map((i) => renderer.getNodeDisplayData(String(i)));
    const xs = pts.map((p) => p.x), ys = pts.map((p) => p.y);
    const span = Math.max(Math.max(...xs) - Math.min(...xs), Math.max(...ys) - Math.min(...ys));
    renderer.getCamera().animate({
      x: (Math.max(...xs) + Math.min(...xs)) / 2,
      y: (Math.max(...ys) + Math.min(...ys)) / 2,
      ratio: Math.min(1, Math.max(0.12, span * 1.5)),
    }, { duration: 600 });
  }

  // ------------------------------------------------------------------ presets
  const PRESETS = [
    { name: "퇴근 후 러너", note: "스포츠·운동에 관심, 러닝·스트레스 관리 기사를 읽음", cats: ["스포츠·운동"], tags: ["수면"], history: ["러닝", "스트레스 관리", "건강한 식단"] },
    { name: "캠핑 가는 부모", note: "아웃도어·육아에 관심, 캠핑·가족 여행 기사를 읽음", cats: ["낚시·아웃도어", "육아·가족"], tags: ["아이와 함께"], history: ["캠핑", "차박", "가족 여행"] },
    { name: "첫 월급 사회초년생", note: "경제·진로에 관심, ETF·청약 기사를 읽음", cats: ["경제·금융", "진로·직업"], tags: [], history: ["ETF", "청약", "취업 준비"] },
    { name: "전시·공연 애호가", note: "미술·공연에 관심, 클래식·박물관 기사를 읽음", cats: ["미술·전시", "공연·음악"], tags: ["독서 모임"], history: ["클래식 음악", "박물관", "LP 수집"] },
    { name: "전기차 고민 중", note: "모빌리티·환경에 관심, 전기차·스마트홈 기사를 읽음", cats: ["자동차·모빌리티", "환경·에너지"], tags: ["탄소중립"], history: ["전기차", "스마트홈", "절약"] },
    { name: "history만 있는 독자", note: "관심 분야 없이 서로 다른 분야의 기사 4개만 읽음", cats: [], tags: [], history: ["제주", "웹툰", "홈카페", "인공지능"] },
  ];

  function applyPreset(p) {
    state.interests = new Set(p.cats.map((n) => M.categories.findIndex((c) => c.name === n)).filter((x) => x >= 0));
    state.tags = new Set(p.tags.map((t) => topicByName.get(t)).filter((x) => x != null));
    const used = new Set(), themes = new Set(), hist = [];
    for (const t of p.history) {
      const f = topicByName.get(t);
      if (f == null) continue;
      // the most focused article for the tag: fewest other facets, so the tag dominates it
      const arts = facetArts[f].length ? [...facetArts[f]] : G.facets.map((fs, i) => (fs.includes(f) ? i : -1)).filter((i) => i >= 0);
      arts.sort((a, b) => artFacets[a].length - artFacets[b].length || a - b);
      const pick = arts.find((i) => !used.has(i) && !themes.has(G.theme[i]));
      if (pick != null) { used.add(pick); themes.add(G.theme[pick]); hist.push(pick); }
    }
    state.history = hist;
    state.selected = null;
    renderDetail();
    update();
    requestAnimationFrame(() => requestAnimationFrame(fitToProfile));
    $("#recommendBtn").focus({ preventScroll: true });
  }

  // ------------------------------------------------------------------ search boxes
  function searchTitles(q, limit = 8) {
    const k = q.trim().toLowerCase();
    if (!k) return [];
    const out = [];
    for (let i = 0; i < N && out.length < limit; i++) if (G.titles[i].toLowerCase().includes(k)) out.push(i);
    return out;
  }

  function wireSuggest(input, list, getItems, onPick) {
    let items = [], active = 0;
    const render = () => {
      list.innerHTML = items.map((it, k) => `<li role="option" class="${k === active ? "active" : ""}" data-k="${k}">${it.html}</li>`).join("");
      list.querySelectorAll("li").forEach((li) => li.addEventListener("mousedown", (e) => { e.preventDefault(); pick(+li.dataset.k); }));
    };
    const pick = (k) => { if (!items[k]) return; onPick(items[k].value); input.value = ""; items = []; render(); };
    input.addEventListener("input", () => { items = getItems(input.value); active = 0; render(); });
    input.addEventListener("keydown", (e) => {
      if (e.key === "ArrowDown") { active = Math.min(items.length - 1, active + 1); render(); e.preventDefault(); }
      else if (e.key === "ArrowUp") { active = Math.max(0, active - 1); render(); e.preventDefault(); }
      else if (e.key === "Enter") { pick(active); e.preventDefault(); }
      else if (e.key === "Escape") { input.value = ""; items = []; render(); }
    });
    input.addEventListener("blur", () => setTimeout(() => { items = []; render(); }, 120));
  }

  function titleItem(i) {
    return { value: i, html: `<span class="dot" style="background:${palette[catGroup[G.category[i]]]}"></span><span class="s-title">${esc(G.titles[i])}</span><span class="s-meta">${esc(M.categories[G.category[i]].name)}</span>` };
  }

  function tagItems(q) {
    const k = keyOf(q);
    if (!k) return [];
    const seen = new Set(), out = [];
    for (const [key, hit] of tagIndex) {
      if (!key.includes(k) || seen.has(hit.tag)) continue;
      const f = topicByName.get(hit.tag);
      if (f == null || state.tags.has(f)) continue;
      seen.add(hit.tag);
      out.push({ value: f, rank: key === k ? 0 : key.startsWith(k) ? 1 : 2,
        html: `<span class="s-title">${esc(hit.tag)}${hit.via ? ` <span class="via">← ${esc(hit.via)} (alias)</span>` : ""}</span><span class="s-meta">기사 ${M.facets[f][2]}개</span>` });
    }
    return out.sort((a, b) => a.rank - b.rank).slice(0, 8);
  }

  function wireInputs() {
    wireSuggest($("#graphSearch"), $("#graphSuggest"), (q) => searchTitles(q).map(titleItem), (i) => select(i, true));
    wireSuggest($("#historyInput"), $("#historySuggest"), (q) => searchTitles(q).filter((i) => !state.history.includes(i)).map(titleItem), addHistory);
    wireSuggest($("#tagInput"), $("#tagSuggest"), tagItems, (f) => { state.tags.add(f); update(); });

    // The sliders tune results already on screen, so they re-run immediately.
    const retune = () => { if (state.recs.length) runRecommend(false); };
    for (const [id, key] of [["wGraph", "graph"], ["wEmb", "emb"], ["wInterest", "interest"]]) {
      const el = $("#" + id), out = el.parentElement.querySelector("output");
      out.textContent = (+el.value).toFixed(2);
      el.addEventListener("input", () => { state.weights[key] = +el.value; out.textContent = (+el.value).toFixed(2); retune(); });
    }
    const div = $("#wDiv"), divOut = div.parentElement.querySelector("output");
    divOut.textContent = (+div.value).toFixed(2);
    div.addEventListener("input", () => { state.diversity = +div.value; divOut.textContent = (+div.value).toFixed(2); retune(); });
    $("#recommendBtn").addEventListener("click", () => runRecommend(true));

    $("#resetBtn").addEventListener("click", () => {
      state.interests.clear(); state.tags.clear(); state.history = []; state.selected = null;
      renderDetail(); update();
      renderer.getCamera().animatedReset({ duration: 500 });
    });
    $("#zoomIn").addEventListener("click", () => renderer.getCamera().animatedZoom({ duration: 300 }));
    $("#zoomOut").addEventListener("click", () => renderer.getCamera().animatedUnzoom({ duration: 300 }));
    $("#zoomReset").addEventListener("click", () => renderer.getCamera().animatedReset({ duration: 400 }));
    $("#legendToggle").addEventListener("click", () => $("#legend").classList.toggle("open"));
    $("#howBtn").addEventListener("click", () => $("#howDialog").showModal());
    $("#howDialog").addEventListener("click", (e) => { if (e.target.id === "howDialog") e.target.close(); });
    $("#themeBtn").addEventListener("click", () => {
      document.documentElement.dataset.theme = luminance(surface) < 0.5 ? "light" : "dark";
      try { localStorage.setItem("theme", document.documentElement.dataset.theme); } catch (e) { /* storage blocked */ }
      applyTheme();
    });
    window.matchMedia("(prefers-color-scheme: dark)").addEventListener("change", applyTheme);
  }

  function applyTheme() {
    readTheme();
    graph.forEachNode((n) => graph.setNodeAttribute(n, "color", palette[catGroup[G.category[+n]]]));
    graph.forEachEdge((e, a) => { if (!a.path) graph.setEdgeAttribute(e, "color", edgeColor); });
    renderLegend(); renderInterests(); renderHistory(); renderRecs(lastInfo);
    if (state.selected != null) renderDetail();
    refreshGraph();
  }

  // ------------------------------------------------------------------ how it works
  function renderHow() {
    $("#howBody").innerHTML = `
      <div class="dlg-head"><h2 id="howTitle">작동 방식</h2><button class="btn ghost small" type="button" id="howClose" aria-label="닫기">✕</button></div>
      <p class="lead">기사 추천 PoC의 파이프라인을 합성 데이터 위에서 다시 돌린 데모입니다. 기사·인물·브랜드·단체는 모두 LLM이 만든 가상 데이터입니다.</p>

      <h3 class="phase">A. 색인 파이프라인 <span>기사가 들어올 때 한 번, 오프라인</span></h3>
      <ol class="steps">
        <li><span class="n">1</span><div><h4>LLM facet 추출</h4><p>LLM이 기사 본문을 읽고 raw facet을 뽑습니다: topic 표현, entity(person·brand·place·org), activity frame, underlying need, life context, 그리고 intent·형식·톤. raw는 기사에 쓰인 그대로의 표현이라 같은 개념도 기사마다 다르게 적힙니다.</p></div></li>
        <li><span class="n">2</span><div><h4>Normalization</h4><p>raw topic은 표기·띄어쓰기를 접는 key fold와 alias 사전으로 canonical tag에 묶이고 (예: 「야간 달리기」·「조깅」 → 러닝), 사전에서 갈리는 동의어는 LLM 리뷰로 병합합니다. 여러 카테고리에 걸친 tag는 카테고리 사이를 잇는 bridge가 됩니다. entity는 표기 변형(「크루 리더 서도윤」 → 서도윤)을 하나의 이름으로, activity·need·life context는 닫힌 목록의 항목으로 매핑합니다.</p></div></li>
        <li><span class="n">3</span><div><h4>Embedding</h4><p>제목·부제·본문으로 기사 embedding을 만들고, 브라우저에서 바로 계산할 수 있도록 압축해 보냅니다.</p></div></li>
        <li><span class="n">4</span><div><h4>기사–facet 그래프</h4><p>normalized facet을 노드로, 기사와 facet을 edge로 잇습니다. 너무 드문 facet은 아무것도 잇지 못하고 너무 흔한 facet은 모든 것을 이어 버리므로, 그 사이 빈도의 facet만 씁니다. edge 가중치는 facet의 희소성(IDF)과 facet type별 중요도로 정합니다. 이 그래프가 추천 시 Forward Push가 걷는 그래프입니다.</p></div></li>
        <li><span class="n">5</span><div><h4>기사 지도와 community</h4><p>기사 쌍의 유사도를 facet overlap과 embedding 유사도로 함께 재고, 기사마다 가장 가까운 이웃과 자기 community 밖의 가까운 기사를 이어 왼쪽 그래프를 만듭니다. Leiden으로 community를 찾고, 같은 유사도의 t-SNE로 배치합니다.</p></div></li>
      </ol>

      <h3 class="phase">B. 추천 파이프라인 <span>추천하기를 누를 때, 브라우저에서</span></h3>
      <ol class="steps">
        <li><span class="n">1</span><div><h4>입력</h4><p>관심 분야, 관심 tag(alias로 입력해도 canonical tag로 바뀜), user history. history는 최근에 읽은 기사일수록 크게 반영합니다.</p></div></li>
        <li><span class="n">2</span><div><h4>Graph 채널 · Forward Push</h4><p>history 기사와 관심 tag 노드에 확률 질량을 놓고 A-4의 기사–facet 그래프 위에서 Forward Push를 돌립니다. 잔차가 큰 노드만 이웃으로 밀어 내므로 전체 그래프를 반복 계산하지 않고 Personalized PageRank 점수를 얻습니다. 같은 인물·장소·tag를 공유하는 기사가 높게 올라옵니다.</p></div></li>
        <li><span class="n">3</span><div><h4>Embedding 채널</h4><p>history embedding과 관심 tag·관심 분야 기사들의 중심을 합쳐 profile 벡터를 만들고, 후보마다 profile과의 유사도와 가장 가까운 history와의 유사도를 함께 봅니다. facet을 공유하지 않아도 내용이 비슷한 기사를 잡습니다.</p></div></li>
        <li><span class="n">4</span><div><h4>관심 분야 채널</h4><p>후보가 고른 카테고리에 속하는지, 관심 tag를 갖는지를 점수화합니다.</p></div></li>
        <li><span class="n">5</span><div><h4>결합과 재정렬</h4><p>읽은 기사를 빼고, 세 채널을 같은 척도로 맞춘 뒤 가중합합니다 (비중은 오른쪽 아래에서 조정). 상위 후보를 MMR로 재정렬해 서로 비슷한 기사를 걸러 내고, 한 theme에 추천이 몰리지 않게 한 뒤 최종 추천을 고릅니다.</p></div></li>
        <li><span class="n">6</span><div><h4>추천 근거</h4><p>추천마다 facet을 가장 많이 공유한 history 기사와 그 facet, 가장 가까운 history와의 embedding 유사도, 관심 분야·tag 일치를 보여 주고, 그래프에는 history → 추천 선으로 그립니다.</p></div></li>
      </ol>`;
    $("#howClose").addEventListener("click", () => $("#howDialog").close());
  }

  // ------------------------------------------------------------------ boot
  load().then(() => {
    readTheme();
    buildGraph();
    prepareLabels();
    renderLegend();
    renderPresets();
    renderHow();
    wireInputs();
    update();
    $("#loading").remove();
  }).catch((err) => {
    console.error(err);
    $("#loading").textContent = "데이터를 불러오지 못했습니다. 로컬에서는 정적 서버로 열어 주세요 (예: python3 -m http.server).";
  });
})();
