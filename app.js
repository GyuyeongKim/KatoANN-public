"use strict";
// Replay mode plays recorded answers (replay.json) after the times they took;
// live mode asks the demo server (/api/ask).
const $ = (id) => document.getElementById(id);
const state = { replay: null, live: false, busy: null, rate: 0, target: 0, scoreTarget: 0, current: null, note: null,
  status: null };
const reduceMotion = matchMedia("(prefers-reduced-motion: reduce)");
const nb = (s) => String(s).replace(/ /g, " ");

// English is the page's own text: static strings come from index.html (data-i18n
// keys), dynamic ones from I18N.en. Korean replaces both; recorded questions,
// answers, and abstracts stay as recorded.
const I18N = {
  en: {
    title: "KatoANN RAG demo",
    langAria: "Switch the language",
    dark: "Dark", light: "Light",
    working: "Working…",
    noServer: "The demo server did not answer.",
    searchNote: (n) => `${n} searches in one batch, 10 nearest of 10,000,000 for each`,
    live: (n, engine) => `Live: ${n} abstracts, ${engine}`,
    none: "No recording and no server",
    offered: (r) => `offered: ${r}`,
    notReached: "not reached",
    reachesAtMost: (r) => `reaches recall ${r} at most`,
    servesAtMost: (r) => `serves ${r} at most`,
    fallsBehind: "falls behind",
    searchTimes: (p50, p99) => `search ${p50} median, ${p99} p99`,
    times: (x) => `${x}×`,
    leadCap: (name, r) => `the searches per second of ${name}, the fastest other system at recall ${r}`,
    noLead: (r) => `at recall ${r}`,
    trend: (items) => `KatoANN's lead over the fastest other system grows with recall: ${items}.`,
    trendItem: (x, r) => `${x} at ${r}`,
    liveH: (r, d) => `Searches arrive at ${r} at every retriever. One dot is ${d} searches.`,
    keepsUp: "keeps up",
    waiting: (n) => `${n} searches waiting`,
    upTo: (r) => `answers up to ${r}`,
    animNote: "The dots run on the measured capacities: searches arrive at the offered load, and each retriever " +
      "answers them in order at its capacity at this recall. A system was not run at loads above one where it already " +
      "fell behind.",
    seg: { plan: "plan", embed: "embed", search: "search", answer: "answer" },
    footSettings: "Settings at this recall: ",
    footUnreached: (name, r) => `${name} reaches recall ${r} at most`,
    footQuestion: (q) => `The question times use the recorded question "${q}".`,
    footer: 'Vector search by <a href="https://github.com/GyuyeongKim/KatoANN-public">KatoANN</a>. ' +
      "Abstracts from PubMed (MedRAG/pubmed) with BGE-M3 embeddings (Qdrant/PubMed-MV); answers by Qwen3-4B-Instruct-2507 " +
      "on llama.cpp. Answers come from a small language model and can be wrong; check the cited abstracts.",
  },
  ko: {
    title: "KatoANN RAG 데모",
    langAria: "언어 변경",
    themeAria: "색상 테마 변경",
    dark: "다크",
    light: "라이트",
    tag: "PubMed RAG 데모",
    paper: "논문",
    h1: "높은 recall을 유지하며 초당 수천 건을 처리하는 RAG 검색",
    lede: "KatoANN은 근사 최근접 이웃(ANN) 검색 시스템으로, Intel AMX 같은 CPU 행렬 연산 엔진을 사용해 벡터 검색을 " +
      "가속합니다. 이 RAG 데모에서는 PubMed 초록 1,000만 건을 초당 수천 회 검색하며, recall은 0.99를 넘습니다. " +
      "질문을 고르면 저장된 실행 과정을 재생합니다.",
    question: "질문",
    pick: "아래에서 예시 선택",
    ask: "질문하기",
    stepPlan: "언어 모델로 검색어 생성",
    stepEmbed: "BGE-M3로 검색어 임베딩",
    stepSearch: "KatoANN으로 초록 1,000만 건 검색",
    stepAnswer: "언어 모델로 답변 생성",
    placeholder: "답변과 답변에 인용한 초록을 여기에 표시합니다.",
    sources: "답변의 근거가 된 초록",
    busyH: "여러 에이전트가 동시에 검색하면",
    busyLede: "여러 에이전트가 동시에 검색하는 서비스에서는 벡터 검색 처리량이 인덱스 하나로 감당할 수 있는 에이전트 수를 결정합니다. 모든 에이전트가 같은 인덱스에 검색을 요청하며, " +
      "위 질문 하나에도 검색이 다섯 번 필요합니다. 요청량이 처리 용량을 넘지 않으면 제때 처리할 수 있습니다. 한계를 넘으면 대기열이 끝없이 늘어나고, 모든 답변이 검색 " +
      "차례를 기다려야 합니다. 각 시스템에는 선택한 recall을 만족하는 가장 가벼운 설정을 적용합니다.",
    rateLabel: "다른 에이전트가 보내는 초당 검색 요청 수",
    loadAria: "부하",
    barsH: "현재 부하에서 위 질문에 답하기까지 걸리는 시간",
    segPlan: "검색어 생성",
    segEmbed: "임베딩",
    segSearch: "검색",
    segAnswer: "답변 생성",
    segWait: "대기열 무한 증가",
    working: "처리 중…",
    noServer: "데모 서버가 응답하지 않았습니다.",
    searchNote: (n) => `검색 ${n}건을 한 번에 처리합니다. 검색마다 초록 1,000만 건 중 가장 가까운 10건을 찾습니다.`,
    live: (n, engine) => `실시간: 초록 ${n}건, ${engine}`,
    none: "실행 기록 없음 · 서버 없음",
    offered: (r) => `요청량: ${r}`,
    notReached: "미달",
    reachesAtMost: (r) => `최대 recall ${r}`,
    servesAtMost: (r) => `최대 ${r} 처리 가능`,
    fallsBehind: "과부하",
    searchTimes: (p50, p99) => `검색 시간: 중앙값 ${p50}, p99 ${p99}`,
    footSettings: "선택한 recall에서 사용한 설정: ",
    footUnreached: (name, r) => `${name}: 최대 recall ${r}`,
    note: "데모 서버의 16코어에서 측정했습니다. 검색 요청은 Poisson 과정에 따라 도착하도록 했으며, 측정 지점마다 20,000건씩 보냈습니다.",
    footQuestion: (q) => `질문 처리 시간은 "${q}"의 실행 기록을 기준으로 계산합니다.`,
    footer: "벡터 검색은 <a href=\"https://github.com/GyuyeongKim/KatoANN-public\">KatoANN</a>이 담당합니다. 초록은 " +
      "PubMed(MedRAG/pubmed)에서, BGE-M3 임베딩은 Qdrant/PubMed-MV에서 가져왔습니다. 답변은 llama.cpp에서 실행하는 " +
      "Qwen3-4B-Instruct-2507이 작성합니다. 소형 언어 모델이 작성하므로 답변이 틀릴 수 있습니다. 인용한 초록을 확인해야 합니다.",
    seg: { plan: "검색어 생성", embed: "임베딩", search: "검색", answer: "답변 생성" },
    scoreH: "같은 recall에서 초당 검색 수",
    scoreNote: "배수는 KatoANN의 초당 검색 수가 각 시스템의 몇 배인지 나타냅니다. 16코어, 초록 1,000만 건 기준입니다.",
    times: (x) => `${x}배`,
    leadCap: (name, r) => `recall ${r}에서 다른 시스템 중 가장 빠른 ${name}의 초당 검색 수 대비`,
    noLead: (r) => `recall ${r} 기준`,
    trend: (items) => `KatoANN은 recall이 높을수록 다른 시스템 중 가장 빠른 시스템보다 초당 검색 수에서 더 크게 앞섭니다(${items}).`,
    trendItem: (x, r) => `recall ${r}에서 ${x}`,
    liveH: (r, d) => `모든 검색기에 ${r}로 검색 요청이 들어옵니다. 점 하나는 검색 ${d}건을 나타냅니다.`,
    keepsUp: "정상 처리",
    waiting: (n) => `검색 ${n}건 대기 중`,
    upTo: (r) => `최대 ${r} 처리 가능`,
    dotIn: "검색 유입",
    dotWait: "대기열에서 대기 중",
    dotOut: "처리 완료",
    animNote: "점의 움직임은 측정한 처리 속도를 재현합니다. 검색 요청은 설정한 부하만큼 들어오고, 각 검색기는 해당 recall에서 측정한 속도로 요청을 순서대로 처리합니다. 과부하가 발생한 시스템은 더 높은 부하에서 실행하지 않았습니다.",
    labels: { svs: "SVS (Vamana 그래프, LeanVec)", scann: "ScaNN (IVF, 4비트 코드)", cabana: "CABANA (IVF, bf16 스캔)" },
  },
};
let lang = "en";

function t(key, ...args) {
  const v = key in I18N[lang] ? I18N[lang][key] : I18N.en[key];
  return typeof v === "function" ? v(...args) : v;
}
const label = (k, e) => (I18N[lang].labels && I18N[lang].labels[k]) || e.label;

function fmtTime(s) {
  if (s == null) return "";
  if (s < 1) return nb(`${Math.round(s * 1000)} ms`);
  return nb(`${s.toFixed(s < 10 ? 1 : 0)} s`);
}

function setTheme(t) {
  document.documentElement.dataset.theme = t;
  try { localStorage.setItem("theme", t); } catch (e) {}
  $("theme").textContent = I18N[lang][t === "dark" ? "light" : "dark"];
  if (lanes.list.length) { lanes.colors = laneColors(); lanes.list.forEach((L) => drawLane(L, 1)); }
}

function setLang(l, save) {
  lang = l === "ko" ? "ko" : "en";
  if (save) {
    try {  // the URL carries the choice, so a reload keeps it and a plain link opens in English
      const u = new URL(location.href);
      if (lang === "en") u.searchParams.delete("lang"); else u.searchParams.set("lang", lang);
      history.replaceState(null, "", u);
    } catch (e) {}
  }
  document.documentElement.lang = lang;
  document.title = t("title");
  document.querySelectorAll("[data-i18n]").forEach((el) => { el.textContent = t(el.dataset.i18n); });
  document.querySelectorAll("[data-i18n-ph]").forEach((el) => { el.placeholder = t(el.dataset.i18nPh); });
  document.querySelectorAll("[data-i18n-aria]").forEach((el) => el.setAttribute("aria-label", t(el.dataset.i18nAria)));
  $("lang").textContent = lang === "en" ? "한국어" : "English";
  $("lang").setAttribute("aria-label", t("langAria"));
  setTheme(document.documentElement.dataset.theme || "light");
  if (state.note != null) $("search-note").textContent = t("searchNote", state.note);
  if (state.status) $("status").textContent = t(...state.status);
  $("footer").innerHTML = t("footer");
  if (score.rows) drawScore(false);
  drawBusy();
}

const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

function stepEl(name) { return document.querySelector(`.steps li[data-step="${name}"]`); }
function setStep(name, cls, t) {
  const el = stepEl(name);
  el.classList.remove("active", "done");
  if (cls) el.classList.add(cls);
  el.querySelector(".time").textContent = t == null ? "" : fmtTime(t);
}

function renderDocs(docs) {
  const ol = $("docs");
  ol.innerHTML = "";
  docs.forEach((d, i) => {
    const li = document.createElement("li");
    li.id = `doc-${i + 1}`;
    const a = document.createElement("a");
    a.className = "title";
    a.href = `https://pubmed.ncbi.nlm.nih.gov/${d.pmid}/`;
    a.target = "_blank";
    a.rel = "noopener";
    a.textContent = d.title;
    const meta = document.createElement("div");
    meta.className = "meta num";
    meta.textContent = `PMID ${d.pmid}`;
    const p = document.createElement("p");
    p.className = "snippet";
    p.textContent = d.abstract.length > 260 ? d.abstract.slice(0, 260).replace(/\s+\S*$/, "") + " …" : d.abstract;
    li.append(a, meta, p);
    ol.append(li);
  });
  $("sources").hidden = false;
}

function answerHTML(text) {
  const esc = text.replace(/&/g, "&amp;").replace(/</g, "&lt;");
  // [3] and [1, 2, 6]: every number links to its abstract
  return esc.replace(/\[(\d+(?:,\s*\d+)*)\]/g, (m, list) =>
    "[" + list.split(/,\s*/).map((n) => `<span class="cite" data-n="${n}">${n}</span>`).join(", ") + "]");
}

async function playAnswer(text, seconds) {
  const box = $("answer");
  box.innerHTML = '<p class="answer-text" id="answer-text"></p>';
  const out = $("answer-text");
  // words come out on the recorded schedule: word i at i / n of the answer time
  const words = text.split(/(\s+)/);
  const n = Math.max(1, words.filter((w) => w.trim()).length);
  const start = performance.now();
  let acc = "", i = 0;
  for (const w of words) {
    acc += w;
    out.innerHTML = answerHTML(acc);
    if (w.trim()) await sleep(Math.max(0, start + (seconds * 1000 * ++i) / n - performance.now()));
  }
}

async function run(q) {
  $("go").disabled = true;
  $("steps").hidden = false;
  $("queries").innerHTML = "";
  $("search-note").textContent = "";
  state.note = null;
  ["plan", "embed", "search", "answer"].forEach((s) => setStep(s, null, null));
  $("answer").innerHTML = `<p class="placeholder">${t("working")}</p>`;
  $("sources").hidden = true;
  let r;
  try {
    if (state.replay) {
      r = state.replay.questions.find((e) => e.question === q);
      const tm = r.timings;
      setStep("plan", "active");
      await sleep(tm.plan_s * 1000);
      r.queries.forEach((s) => { const li = document.createElement("li"); li.textContent = s; $("queries").append(li); });
      setStep("plan", "done", tm.plan_s);
      setStep("embed", "active");
      await sleep(tm.embed_s * 1000);
      setStep("embed", "done", tm.embed_s);
      setStep("search", "active");
      const search = tm.retrieval_wait_s + tm.retrieval_search_s;
      await sleep(search * 1000);
      setStep("search", "done", search);
      state.note = r.queries.length + 1;
      $("search-note").textContent = t("searchNote", state.note);
      renderDocs(r.documents);
      setStep("answer", "active");
      await playAnswer(r.answer, tm.answer_s);
      setStep("answer", "done", tm.answer_s);
    } else {
      setStep("plan", "active");
      const res = await fetch("api/ask", { method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ question: q, sub_queries: 4 }) });
      r = await res.json();
      const tm = r.timings;
      r.queries.forEach((s) => { const li = document.createElement("li"); li.textContent = s; $("queries").append(li); });
      setStep("plan", "done", tm.plan_s);
      setStep("embed", "done", tm.embed_s);
      setStep("search", "done", tm.retrieval_wait_s + tm.retrieval_search_s);
      state.note = r.queries.length + 1;
      $("search-note").textContent = t("searchNote", state.note);
      renderDocs(r.documents);
      $("answer").innerHTML = `<p class="answer-text">${answerHTML(r.answer)}</p>`;
      setStep("answer", "done", tm.answer_s);
    }
    state.current = r;
    drawBusy();
  } catch (e) {
    $("answer").innerHTML = `<p class="placeholder">${t("noServer")}</p>`;
  } finally {
    $("go").disabled = false;
  }
}

function seg(container, items, value, onPick) {
  container.innerHTML = "";
  items.forEach(([v, label]) => {
    const b = document.createElement("button");
    b.type = "button";
    b.role = "radio";
    b.textContent = label;
    b.setAttribute("aria-checked", String(v === value));
    b.addEventListener("click", () => onPick(v));
    container.append(b);
  });
}

const fmtRate = (q) => nb(`${Math.round(q).toLocaleString("en-US")}/s`);
const fmtX = (x) => nb(t("times", x >= 10 ? String(Math.round(x)) : x.toFixed(1)));
const shortName = (e) => e.label.split(" (")[0];

// f(p) for p from 0 to 1 over ms, eased; the first frame is the clock origin, and a timer ends it at f(1)
// when frames stop coming (background tabs).
function tween(ms, f) {
  if (reduceMotion.matches) { f(1); return () => {}; }
  let t0 = null, done = false, raf = 0;
  const end = () => { if (!done) { done = true; cancelAnimationFrame(raf); f(1); } };
  const timer = setTimeout(end, ms + 300);
  const frame = (now) => {
    if (done) return;
    if (t0 == null) t0 = now;
    const p = Math.min(1, (now - t0) / ms);
    if (p >= 1) { clearTimeout(timer); end(); return; }
    f(1 - Math.pow(1 - p, 3));
    raf = requestAnimationFrame(frame);
  };
  raf = requestAnimationFrame(frame);
  return () => { done = true; clearTimeout(timer); cancelAnimationFrame(raf); };
}

// KatoANN's capacity over the fastest other system that reaches the target's recall.
function leadAt(T) {
  const me = T.engines[T.order[0]];
  const others = T.order.slice(1).map((k) => T.engines[k]).filter((e) => !e.unreachable && e.capacity_qps);
  if (!others.length) return null;
  const best = others.reduce((a, e) => (e.capacity_qps > a.capacity_qps ? e : a));
  return { best, x: me.capacity_qps / best.capacity_qps };
}

// The scoreboard beside the introduction: each system's capacity at one recall, bars on one scale for all
// recalls, and KatoANN's lead counted up from 1x.
const score = { rows: null, shown: {}, stop: () => {} };
function drawScore(animate) {
  const B = state.busy;
  if (!B || !B.targets || !B.targets.length) return;
  $("score").hidden = false;
  const T = B.targets[state.scoreTarget];
  seg($("score-recalls"), B.targets.map((x, i) => [i, x.recall]), state.scoreTarget,
      (v) => { state.scoreTarget = v; drawScore(true); });
  const scale = Math.max(...B.targets.flatMap((x) => x.order.map((k) => x.engines[k].capacity_qps || 0)));
  const box = $("score-rows");
  if (!score.rows) {
    box.innerHTML = "";
    score.rows = {};
    T.order.forEach((k, i) => {
      const row = document.createElement("div");
      row.className = "score-row" + (i === 0 ? " me" : "");
      const who = document.createElement("span");
      who.className = "who";
      const track = document.createElement("div");
      track.className = "score-track";
      const fill = document.createElement("div");
      fill.className = "score-fill";
      track.append(fill);
      const val = document.createElement("span");
      val.className = "score-val";
      const x = document.createElement("span");
      x.className = "score-x";
      row.append(who, track, val, x);
      box.append(row);
      score.rows[k] = { row, who, track, fill, val, x };
    });
    void box.offsetWidth;  // bars start from zero width
  }
  const lead = leadAt(T);
  const me = T.engines[T.order[0]].capacity_qps;
  const from = { ...score.shown };
  const to = {};
  T.order.forEach((k, i) => {
    const e = T.engines[k], r = score.rows[k];
    r.who.textContent = shortName(e);
    r.row.classList.toggle("miss", !!e.unreachable);
    r.track.hidden = r.x.hidden = !!e.unreachable;
    r.fill.style.width = `${(100 * (e.capacity_qps || 0)) / scale}%`;
    if (e.unreachable) {
      r.val.textContent = `${t("notReached")} (${t("reachesAtMost", e.max_recall.toFixed(4))})`;
    }
    to[k] = e.capacity_qps || 0;
    to[`x${k}`] = i === 0 || !e.capacity_qps ? 0 : me / e.capacity_qps;
  });
  to.lead = lead ? lead.x : 0;
  $("score-cap").textContent = lead ? t("leadCap", shortName(lead.best), T.recall) : t("noLead", T.recall);
  const paint = (p) => {
    T.order.forEach((k, i) => {
      const e = T.engines[k], r = score.rows[k];
      const a = from[k] == null ? 0 : from[k];
      const v = a + (to[k] - a) * p;
      score.shown[k] = v;
      if (!e.unreachable) r.val.textContent = fmtRate(v);
      const xa = from[`x${k}`] || 1;
      const xv = xa + (to[`x${k}`] - xa) * p;
      score.shown[`x${k}`] = xv;
      r.x.textContent = i === 0 || e.unreachable ? "" : fmtX(xv);
    });
    const la = from.lead || 1;
    const lv = la + (to.lead - la) * p;
    score.shown.lead = lv;
    $("score-big").textContent = to.lead ? fmtX(lv) : "";
  };
  score.stop();
  score.stop = animate ? tween(900, paint) : (paint(1), () => {});
  const steps = B.targets.map((x) => [x.recall, leadAt(x)]);
  const rising = steps.every(([, l], i) => l && (i === 0 || l.x > steps[i - 1][1].x));
  $("score-trend").textContent = rising && steps.length > 1
    ? t("trend", steps.map(([r, l]) => t("trendItem", fmtX(l.x), r)).join(", ")) : "";
}

// Live lanes: searches arrive at the offered load (Poisson) and reach the index TRAVEL seconds later; each
// retriever answers them first come, first served, at its measured capacity. One dot stands for DOT searches.
const DOT = 50, GATE = 0.62, LANE_H = 34, TRAVEL = 2;
const lanes = { key: "", list: [], raf: 0, last: 0, textAt: 0, visible: false, colors: null };

function laneColors() {
  const cs = getComputedStyle(document.documentElement);
  const v = (n) => cs.getPropertyValue(n).trim();
  return { track: v("--surface"), gate: v("--rule-strong"), arrive: v("--ink-3"), wait: v("--bad"),
           me: v("--accent"), out: v("--bar-base") };
}
const expGap = (lam) => (lam > 0 ? (-Math.log(1 - Math.random()) * DOT) / lam : Infinity);
// rank 0 is in service at the gate; the waiting dots stack leftwards in three rows
const slotX = (L, r) => (r === 0 ? L.gx : L.gx - 6 - Math.floor((r - 1) / 3) * 5);
const slotY = (r) => (r === 0 ? LANE_H / 2 : LANE_H / 2 + (((r - 1) % 3) - 1) * 7);

function advanceLane(L, t) {
  L.t = t;
  if (!L.lam) return;
  while (L.nextSpawn <= t) {
    const arrive = L.nextSpawn + TRAVEL;
    const depart = Math.max(arrive, L.lastDepart) + DOT / L.mu;
    L.lastDepart = depart;
    L.d.push({ spawn: L.nextSpawn, arrive, depart, y: LANE_H / 2 + (Math.random() * 2 - 1) * 9, qx: null, qy: 0 });
    L.nextSpawn += expGap(L.lam);
  }
  while (L.h < L.d.length && L.d[L.h].depart <= t) L.out.push(L.d[L.h++]);
  L.a = Math.max(L.a, L.h);
  while (L.a < L.d.length && L.d[L.a].arrive <= t) L.a++;
  if (L.h > 512) { L.d = L.d.slice(L.h); L.a -= L.h; L.h = 0; }
}

const waitingDots = (L) => Math.max(0, L.a - L.h - 1);

function drawLane(L, dt) {
  const c = L.ctx, C = lanes.colors;
  if (!c || !L.w) return;
  c.clearRect(0, 0, L.w, LANE_H);
  c.fillStyle = C.track;
  c.beginPath();
  if (c.roundRect) c.roundRect(0, 0, L.w, LANE_H, 6); else c.rect(0, 0, L.w, LANE_H);
  c.fill();
  if (!L.lam) return;
  c.fillStyle = L.me ? C.me : C.gate;
  c.fillRect(L.gx - 1, 4, 3, LANE_H - 8);
  const t = L.t, v = L.gx / TRAVEL, k = Math.min(1, dt * 12);
  const arriving = [], waiting = [], out = [];
  // dots at the index take their place in line; dots still on the way stop at the end of the line
  const last = Math.min(L.a, L.h + 3 * Math.floor((L.gx - 6) / 5) + 4);
  for (let i = L.h; i < last; i++) {
    const d = L.d[i], r = i - L.h, sx = slotX(L, r);
    if (sx < 0) break;
    if (d.qx == null) { d.qx = Math.min(v * (t - d.spawn), sx); d.qy = d.y; }
    d.qx += (sx - d.qx) * k;
    d.qy += (slotY(r) - d.qy) * k;
    (r > 0 ? waiting : arriving).push(d.qx, d.qy);
  }
  const tail = slotX(L, L.a - L.h);
  for (let i = L.a; i < L.d.length; i++) {
    const d = L.d[i], x = Math.min(v * (t - d.spawn), tail);
    if (x > -3) arriving.push(x, d.y);
  }
  L.out = L.out.filter((d) => {
    const x = L.gx + 3 + v * (t - d.depart);
    if (x < L.w + 4) out.push(x, d.y);
    return x < L.w + 4;
  });
  const dots = (xy, color) => {
    c.fillStyle = color;
    c.beginPath();
    for (let i = 0; i < xy.length; i += 2) { c.moveTo(xy[i] + 2.3, xy[i + 1]); c.arc(xy[i], xy[i + 1], 2.3, 0, 2 * Math.PI); }
    c.fill();
  };
  dots(arriving, C.arrive);
  dots(out, L.me ? C.me : C.out);
  dots(waiting, C.wait);
}

function laneText(L) {
  const e = L.e, pt = L.pt;
  let cls = "ok", main = t("keepsUp"), sub = pt && !pt.overloaded ? t("searchTimes", fmtTime(pt.p50_ms / 1000),
    fmtTime(pt.p99_ms / 1000)) : "";
  if (e.unreachable) { cls = ""; main = t("notReached"); sub = t("reachesAtMost", e.max_recall.toFixed(4)); }
  else if (pt && pt.overloaded) {
    cls = "bad"; main = t("fallsBehind");
    sub = t("waiting", nb((waitingDots(L) * DOT).toLocaleString("en-US")));
  }
  L.st.className = `lane-st ${cls}`;
  L.main.textContent = main;
  L.sub.textContent = sub;
  L.whoSub.textContent = e.unreachable ? "" : t("upTo", fmtRate(e.capacity_qps));
}

function sizeLanes() {
  const dpr = window.devicePixelRatio || 1;
  for (const L of lanes.list) {
    const w = L.cv.clientWidth;
    if (!w) continue;
    L.w = w; L.gx = w * GATE;
    L.cv.width = Math.round(w * dpr);
    L.cv.height = Math.round(LANE_H * dpr);
    L.ctx = L.cv.getContext("2d");
    L.ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    L.d.forEach((d) => { d.qx = null; });
    drawLane(L, 1);
  }
}

function laneFrame(now) {
  lanes.raf = 0;
  if (!lanes.visible || document.hidden || reduceMotion.matches) return;
  const dt = lanes.last ? Math.min(0.25, (now - lanes.last) / 1000) : 0;
  lanes.last = now;
  for (const L of lanes.list) { advanceLane(L, L.t + dt); drawLane(L, dt); }
  if (now - lanes.textAt > 250) { lanes.textAt = now; lanes.list.forEach(laneText); }
  lanes.raf = requestAnimationFrame(laneFrame);
}
function laneStart() {
  if (lanes.raf || !lanes.visible || document.hidden || reduceMotion.matches) return;
  lanes.last = 0;
  lanes.raf = requestAnimationFrame(laneFrame);
}

function drawLanes(T) {
  $("lanes-h").textContent = t("liveH", fmtRate(state.rate), DOT);
  const key = `${state.target}:${state.rate}`;
  if (lanes.key !== key) {
    lanes.key = key;
    const box = $("lanes");
    box.innerHTML = "";
    // the stream starts already reaching the index; with reduced motion, the lanes show the state after 10 s
    const t0 = reduceMotion.matches ? 10 + TRAVEL : TRAVEL;
    lanes.list = T.order.map((k, i) => {
      const e = T.engines[k];
      const row = document.createElement("div");
      row.className = "lane-row";
      const who = document.createElement("div");
      who.className = "who" + (i === 0 ? " me" : "");
      who.textContent = label(k, e);
      const whoSub = document.createElement("span");
      whoSub.className = "who-sub num";
      who.append(whoSub);
      const cv = document.createElement("canvas");
      cv.className = "lane-cv";
      cv.setAttribute("aria-hidden", "true");
      const st = document.createElement("div");
      const main = document.createElement("span");
      main.className = "main";
      const sub = document.createElement("span");
      sub.className = "sub";
      st.append(main, sub);
      row.append(who, cv, st);
      box.append(row);
      const lam = e.unreachable || !e.capacity_qps ? 0 : state.rate;
      const L = { k, e, pt: e.points[String(state.rate)], me: i === 0, cv, st, main, sub, whoSub, who, lam,
                  mu: e.capacity_qps || 0, d: [], out: [], h: 0, a: 0, t: 0, nextSpawn: expGap(lam), lastDepart: 0,
                  w: 0, gx: 0, ctx: null };
      advanceLane(L, t0);
      return L;
    });
    lanes.colors = laneColors();
    sizeLanes();
  } else {
    lanes.list.forEach((L) => { L.who.firstChild.textContent = label(L.k, L.e); });
  }
  lanes.list.forEach(laneText);
  laneStart();
}

const bars = { key: "", visible: false };

// The busy-server panel. At the chosen recall, the live lanes run each system at the
// offered load from other agents; below capacity a question's time is set by the
// language model, above it the search queue grows without bound.
function drawBusy() {
  const B = state.busy;
  if (!B || !B.targets || !B.targets.length) { $("busy").hidden = true; return; }
  $("busy").hidden = false;
  const T = B.targets[state.target];
  if (!T.rates.includes(state.rate)) state.rate = T.default_rate;
  const ref = state.current || (state.replay && state.replay.questions[0]);
  const tm = ref ? ref.timings : { plan_s: 2, embed_s: 0.07, answer_s: 12 };
  seg($("recalls"), B.targets.map((x, i) => [i, x.recall]), state.target,
      (v) => { state.target = v; state.rate = B.targets[v].default_rate; drawBusy(); });
  seg($("rates"), T.rates.map((r) => [r, r.toLocaleString("en-US")]), state.rate,
      (v) => { state.rate = v; drawBusy(); });

  drawLanes(T);

  const rows = T.order.map((k) => {
    const e = T.engines[k];
    const pt = e.points[String(state.rate)];
    const search = pt && !pt.overloaded ? pt.p50_ms / 1000 : null;
    return { k, e, label: label(k, e), pt, search,
             total: search == null ? null : tm.plan_s + tm.embed_s + search + tm.answer_s };
  });
  const max = Math.max(...rows.map((r) => r.total || 0), tm.plan_s + tm.embed_s + tm.answer_s) * 1.02;
  const barsKey = `${state.target}:${state.rate}:${ref ? ref.question : ""}`;
  const grow = bars.key !== barsKey;
  bars.key = barsKey;
  const box = $("bars");
  box.innerHTML = "";
  rows.forEach((r, i) => {
    const row = document.createElement("div");
    row.className = "bar-row";
    const who = document.createElement("div");
    who.className = "who" + (i === 0 ? " me" : "");
    who.textContent = r.label;
    const sub = document.createElement("span");
    sub.className = "who-sub num";
    const track = document.createElement("div");
    track.className = "track" + (grow ? (bars.visible ? " grow" : " grow-wait") : "");
    const total = document.createElement("div");
    total.className = "total";
    if (r.total == null) {
      sub.textContent = r.e.unreachable ? t("reachesAtMost", r.e.max_recall.toFixed(4))
        : t("servesAtMost", fmtRate(r.e.capacity_qps));
      total.textContent = r.e.unreachable ? t("notReached") : t("fallsBehind");
      if (!r.e.unreachable) {  // the hatch means a growing queue; a missed recall stays empty
        const w = document.createElement("div");
        w.className = "seg-bar wait";
        w.style.width = "100%";
        track.append(w);
      }
    } else {
      sub.textContent = t("searchTimes", fmtTime(r.pt.p50_ms / 1000), fmtTime(r.pt.p99_ms / 1000));
      [["plan", tm.plan_s], ["embed", tm.embed_s], ["search", r.search], ["answer", tm.answer_s]].forEach(([c, v]) => {
        const s = document.createElement("div");
        s.className = `seg-bar s-${c}`;  // s- prefix: .answer is the answer box
        s.style.width = `${(100 * v) / max}%`;
        s.title = `${t("seg")[c]}: ${fmtTime(v)}`;
        track.append(s);
      });
      total.textContent = fmtTime(r.total);
    }
    who.append(sub);
    row.append(who, track, total);
    box.append(row);
  });
  $("busy-foot").textContent = t("footSettings") + T.order.map((k) => {
    const e = T.engines[k];
    const name = e.label.split(" (")[0];
    return e.unreachable ? t("footUnreached", name, e.max_recall.toFixed(4))
      : `${name} ${e.setting} (recall ${e.recall.toFixed(3)})`;
  }).join(", ") + `. ${lang === "en" ? B.note || "" : t("note")} ` + t("animNote") + " " +
    t("footQuestion", ref ? ref.question : "");
}

async function init() {
  document.querySelectorAll("[data-i18n]").forEach((el) => { I18N.en[el.dataset.i18n] = el.textContent; });
  document.querySelectorAll("[data-i18n-ph]").forEach((el) => { I18N.en[el.dataset.i18nPh] = el.placeholder; });
  document.querySelectorAll("[data-i18n-aria]").forEach((el) => { I18N.en[el.dataset.i18nAria] = el.getAttribute("aria-label"); });
  lang = new URLSearchParams(location.search).get("lang") === "ko" ? "ko" : "en";
  setTheme(document.documentElement.dataset.theme ||
           (matchMedia("(prefers-color-scheme: dark)").matches ? "dark" : "light"));
  $("lang").addEventListener("click", () => setLang(lang === "en" ? "ko" : "en", true));
  $("theme").addEventListener("click", () =>
    setTheme(document.documentElement.dataset.theme === "dark" ? "light" : "dark"));
  let examples = [];
  try {
    const r = await fetch("replay.json?v=8cc3aeba70");
    if (r.ok) state.replay = await r.json();
  } catch (e) {}
  if (state.replay) {
    examples = state.replay.questions.map((e) => e.question);
    $("input").readOnly = true;
  } else {
    try {
      const info = await (await fetch("api/info")).json();
      state.live = true;
      state.status = ["live", info.documents.toLocaleString("en-US"), info.engine];
    } catch (e) {
      state.status = ["none"];
    }
  }
  try {
    const r = await fetch("load.json?v=9a6907c1bb");
    if (r.ok) state.busy = await r.json();
  } catch (e) {}
  if (state.busy && state.busy.targets && state.busy.targets.length) {
    // open at recall 0.995 where it was measured, else the highest target
    const i = state.busy.targets.findIndex((x) => x.recall === "0.995");
    state.target = i >= 0 ? i : state.busy.targets.length - 1;
    state.rate = state.busy.targets[state.target].default_rate;
    const leads = state.busy.targets.map((x) => (leadAt(x) || { x: 0 }).x);
    state.scoreTarget = leads.indexOf(Math.max(...leads));
  }
  const ex = $("examples");
  examples.forEach((q) => {
    const b = document.createElement("button");
    b.type = "button";
    b.textContent = q;
    b.addEventListener("click", () => { $("input").value = q; run(q); });
    ex.append(b);
  });
  $("form").addEventListener("submit", (e) => {
    e.preventDefault();
    const q = $("input").value.trim();
    if (q && (state.live || (state.replay && examples.includes(q)))) run(q);
  });
  $("answer").addEventListener("click", (e) => {
    const n = e.target.dataset && e.target.dataset.n;
    if (!n) return;
    const li = $(`doc-${n}`);
    if (li) { li.scrollIntoView({ behavior: "smooth", block: "nearest" }); li.classList.add("flash");
      setTimeout(() => li.classList.remove("flash"), 900); }
  });
  setLang(lang, false);  // also draws the busy panel and the footer
  if (state.busy) drawScore(true);
  new IntersectionObserver((es) => { lanes.visible = es[es.length - 1].isIntersecting; laneStart(); }).observe($("lanes"));
  new ResizeObserver(() => sizeLanes()).observe($("lanes"));
  new IntersectionObserver((es) => {
    bars.visible = es[es.length - 1].isIntersecting;
    if (bars.visible) document.querySelectorAll(".track.grow-wait").forEach((el) => el.classList.replace("grow-wait", "grow"));
  }).observe($("bars"));
  document.addEventListener("visibilitychange", laneStart);
  // a link ending in #busy opens at the panel (it is hidden until load.json arrives)
  if (location.hash === "#busy" && !$("busy").hidden) $("busy").scrollIntoView();
}
init();
