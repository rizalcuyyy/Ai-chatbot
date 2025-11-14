// api/ask.js
const fs = require('fs');
const path = require('path');

// Tunable params
const TOP_K_VOCAB = 4000; // limit vocab to save memory
const FALLBACK_THRESHOLD = 0.12; // similarity threshold

let INDEX = null;

function tokenize(s) {
  return (s || "").toLowerCase()
    .replace(/[^a-z0-9\s\-\+\.#]/g, ' ')
    .split(/\s+/)
    .filter(t => t && t.length > 1);
}

function termFreq(tokens) {
  const tf = {};
  tokens.forEach(t => tf[t] = (tf[t] || 0) + 1);
  const n = tokens.length || 1;
  for (let k in tf) tf[k] = tf[k] / n;
  return tf;
}

function dot(a, b) {
  let s = 0;
  for (let i = 0; i < a.length; i++) s += a[i] * (b[i] || 0);
  return s;
}

function norm(a) {
  let s = 0; for (let v of a) s += v * v; return Math.sqrt(s);
}

function buildIndex(docs) {
  const docTokens = docs.map(d => tokenize(d));
  const df = {};
  docTokens.forEach(tokens => {
    const seen = new Set(tokens);
    for (let t of seen) df[t] = (df[t] || 0) + 1;
  });

  const vocab = Object.keys(df).sort((a,b)=> (df[b]-df[a]) || a.localeCompare(b));
  const vocabLimited = vocab.slice(0, TOP_K_VOCAB);
  const idf = {};
  const N = Math.max(1, docs.length);
  for (let t of vocabLimited) idf[t] = Math.log((N) / (1 + df[t]));

  const docVecs = docTokens.map(tokens => {
    const tf = termFreq(tokens);
    const vec = new Array(vocabLimited.length).fill(0);
    for (let i = 0; i < vocabLimited.length; i++) {
      const term = vocabLimited[i];
      vec[i] = (tf[term] || 0) * (idf[term] || 0);
    }
    return vec;
  });

  const norms = docVecs.map(v => norm(v));

  return { docs, vocab: vocabLimited, idf, docVecs, norms };
}

async function ensureIndex() {
  if (INDEX) return INDEX;
  const dataPath = path.join(process.cwd(), 'public', 'data.json');
  let raw = '[]';
  try {
    raw = fs.readFileSync(dataPath, 'utf8');
  } catch (e) {
    console.error('data.json not found in /public.');
  }
  let arr = [];
  try { arr = JSON.parse(raw); }
  catch { arr = raw.split(/\r?\n/).map(l=>l.trim()).filter(Boolean); }

  const docs = arr.map(x => {
    if (typeof x === 'string') return x;
    if (x && typeof x === 'object') {
      const q = x.q || x.question || '';
      const a = x.a || x.answer || '';
      if (q && a) return q + "\n---\n" + a;
      return a || q || '';
    }
    return String(x);
  });

  INDEX = buildIndex(docs);
  return INDEX;
}

module.exports = async function handler(req, res) {
  const idx = await ensureIndex();
  const body = req.body || {};
  const query = (body.query || "").toString();
  if (!query) return res.json({ answer: null });

  const qTokens = tokenize(query);
  const qtf = termFreq(qTokens);
  const qvec = new Array(idx.vocab.length).fill(0);
  for (let i=0;i<idx.vocab.length;i++) {
    const t = idx.vocab[i];
    qvec[i] = (qtf[t] || 0) * (idx.idf[t] || 0);
  }
  const qnorm = norm(qvec);

  let bestScore = -1; let bestIdx = -1;
  for (let i=0;i<idx.docVecs.length;i++) {
    const denom = qnorm * (idx.norms[i] || 1e-9);
    let sim = 0;
    if (denom > 0) sim = dot(idx.docVecs[i], qvec) / denom;
    if (sim > bestScore) { bestScore = sim; bestIdx = i; }
  }

  const fallback = [
    "Maaf, gue belum nangkep maksudnya. Coba jelasin lagi.",
    "Kayaknya kurang jelas, coba detailin.",
    "Aku AI offline, tolong kasih konteks.",
    "Belum nemu jawabannya. Jelasin ulang?",
    "Sepertinya konteks kurang lengkap.",
  ];

  const answer = (bestScore >= FALLBACK_THRESHOLD && bestIdx >= 0)
    ? idx.docs[bestIdx]
    : fallback[Math.floor(Math.random()*fallback.length)];

  return res.json({ answer, score: bestScore, index: bestIdx });
};