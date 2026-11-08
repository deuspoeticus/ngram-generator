// ngram.js: a small n-gram language model that can generate text and explain
// where its output came from. Works in the browser (loaded by index.html) and
// in Node:
//
//   node ngram.js [n] [corpus.txt] [word|char]

const BOS = '<s>';   // beginning-of-sentence padding
const EOS = '</s>';  // end-of-sentence marker

// ---------------------------------------------------------------------------
// Step 1: Tokenize. Split text into sentences, and sentences into tokens.
//
// Each token keeps its character offsets in the original text, so we can
// later point back to where a copied stretch came from. In both modes a
// sentence ends at . ! or ? followed by whitespace (or the end of the text).
// ---------------------------------------------------------------------------

// Word mode: words and punctuation marks are tokens. \p{L} is any Unicode
// letter, so accented words ("café") stay whole, and "don't" is one token.
const WORD_RE = /\p{L}+(?:['’]\p{L}+)*|\p{N}+|[.,!?;:]/gu;

function wordTokenize(text) {
  const sentences = [];
  let current = [];
  for (const m of text.matchAll(WORD_RE)) {
    const start = m.index, end = start + m[0].length;
    current.push({ token: m[0].toLowerCase(), start, end });
    if (/^[.!?]$/.test(m[0]) && (end === text.length || /\s/.test(text[end]))) {
      sentences.push(current);
      current = [];
    }
  }
  if (current.length) sentences.push(current);
  return sentences;
}

// Character mode: every character is a token, including spaces (a run of
// whitespace becomes a single ' '). Other symbols such as quotes and
// brackets are dropped.
const CHAR_RE = /\s+|[\p{L}\p{N}.,!?;:'’-]/gu;

function charTokenize(text, { keepTrailingSpace = false } = {}) {
  const sentences = [];
  let current = [];
  const last = () => current[current.length - 1];
  const finish = () => {
    if (!keepTrailingSpace) while (current.length && last().token === ' ') current.pop();
    if (current.length) sentences.push(current);
    current = [];
  };
  for (const m of text.matchAll(CHAR_RE)) {
    const start = m.index;
    if (/^\s/.test(m[0])) {
      if (current.length === 0) continue;                       // no leading space
      if (/^[.!?]$/.test(last().token)) { finish(); continue; } // sentence ends
      if (last().token !== ' ') current.push({ token: ' ', start, end: start + 1 });
      continue;
    }
    current.push({ token: m[0].toLowerCase(), start, end: start + m[0].length });
  }
  finish();
  return sentences;
}

// Tokens back into readable text (word mode): no space before punctuation.
function joinWords(tokens) {
  return tokens.join(' ').replace(/ ([.,!?;:])/g, '$1');
}

// Pad a sentence so the first token has a full context and the model learns
// where sentences end. For n = 3: [<s>, <s>, the, cat, sat, </s>]
function pad(sentence, n) {
  return [...Array(n - 1).fill(BOS), ...sentence, EOS];
}

// Lists of tokens are stored as Map keys by joining them with a space. This
// is unambiguous: word tokens never contain spaces, and in character mode
// the space token never appears twice in a row.
const key = tokens => tokens.join(' ');

// ---------------------------------------------------------------------------
// The model.
//
// Training slides a window of n tokens over every padded sentence and counts
//   context (the previous n-1 tokens) -> { total, next: Map(token -> count) }
// so "what can follow this context?" is a single Map lookup.
//
// For explaining the output it also keeps the whole source as one token list,
// with an index from each token to every position where it occurs.
// ---------------------------------------------------------------------------
class NGramModel {
  // mode: 'word' (tokens are words and punctuation) or 'char' (characters).
  constructor(n, { mode = 'word' } = {}) {
    if (!Number.isInteger(n) || n < 1) throw new Error('n must be an integer >= 1');
    this.n = n;
    this.mode = mode;
    this.table = new Map();     // context key -> { total, next: Map(token -> count) }
    this.sentences = new Set(); // every source sentence, to detect verbatim copies
    this.source = [];           // all source tokens in order, </s> between sentences
    this.positions = new Map(); // token -> every index where it occurs in this.source
    this.offsets = [];          // [start, end] character offsets of each source token
  }

  // Split text into sentences of { token, start, end }.
  tokenize(text) {
    return this.mode === 'char' ? charTokenize(text) : wordTokenize(text);
  }

  // Tokens for a prompt such as the starting words. In character mode a
  // trailing space is kept, so "the " continues with a new word.
  tokenizePrompt(text) {
    const sentences = this.mode === 'char'
      ? charTokenize(text, { keepTrailingSpace: true })
      : wordTokenize(text);
    return sentences.flat().map(t => t.token);
  }

  // Tokens back into text. Characters are simply concatenated.
  join(tokens) {
    return this.mode === 'char' ? tokens.join('') : joinWords(tokens);
  }

  // -------------------------------------------------------------------------
  // Step 2: Count.
  // -------------------------------------------------------------------------
  train(text) {
    const n = this.n;
    for (const sentenceTokens of this.tokenize(text)) {
      const sentence = sentenceTokens.map(t => t.token);
      this.sentences.add(key(sentence));

      for (const { token, start, end } of sentenceTokens) {
        let list = this.positions.get(token);
        if (!list) this.positions.set(token, list = []);
        list.push(this.source.length);
        this.source.push(token);
        this.offsets.push([start, end]);
      }
      this.source.push(EOS); // stops copied stretches from crossing sentences
      this.offsets.push(null);

      const padded = pad(sentence, n);
      for (let i = n - 1; i < padded.length; i++) {
        const context = key(padded.slice(i - n + 1, i));
        const next = padded[i];

        let entry = this.table.get(context);
        if (!entry) this.table.set(context, entry = { total: 0, next: new Map() });
        entry.total++;
        entry.next.set(next, (entry.next.get(next) ?? 0) + 1);
      }
    }
    return this;
  }

  // -------------------------------------------------------------------------
  // Step 3: Probabilities.
  // -------------------------------------------------------------------------

  // P(token | context) = count(context, token) / count(context)
  probability(context, token) {
    const entry = this.table.get(key(context));
    if (!entry) return 0;
    return (entry.next.get(token) ?? 0) / entry.total;
  }

  // Every token that can follow a context, with its probability, most likely
  // first. Temperature reshapes the probabilities:
  //   T = 1  the true probabilities, count / total
  //   T < 1  sharper: likely tokens become even likelier (T near 0 always picks the top one)
  //   T > 1  flatter: rare tokens get more of a chance
  // Each count is raised to the power 1/T, then everything is renormalized.
  // Dividing by the largest count first keeps the powers from overflowing.
  distribution(context, temperature = 1) {
    const entry = this.table.get(key(context));
    if (!entry) return [];
    const t = Math.max(temperature, 0.01);

    let max = 0;
    for (const count of entry.next.values()) max = Math.max(max, count);

    const rows = [];
    let sum = 0;
    for (const [token, count] of entry.next) {
      const weight = Math.pow(count / max, 1 / t);
      rows.push({ token, count, weight });
      sum += weight;
    }
    for (const row of rows) row.p = row.weight / sum;
    return rows.sort((a, b) => b.p - a.p);
  }

  // -------------------------------------------------------------------------
  // Step 4: Generate.
  // -------------------------------------------------------------------------

  // Pick the next token at random, in proportion to its probability: take a
  // random number in [0, 1) and walk the probabilities until it is used up.
  sampleNext(context, temperature = 1) {
    const entry = this.table.get(key(context));
    if (!entry) return EOS; // unseen context: end the sentence

    if (temperature === 1) { // fast path: walk the raw counts, no sorting needed
      let r = Math.random() * entry.total;
      for (const [token, count] of entry.next) {
        r -= count;
        if (r < 0) return token;
      }
      return EOS; // only reached through floating-point rounding
    }

    let r = Math.random();
    for (const { token, p } of this.distribution(context, temperature)) {
      r -= p;
      if (r < 0) return token;
    }
    return EOS;
  }

  // The n-1 tokens before position i of a sentence, padded with <s> at the
  // start. This is the context the model used to choose tokens[i].
  contextFor(tokens, i) {
    const padded = [...Array(this.n - 1).fill(BOS), ...tokens];
    return padded.slice(i, i + this.n - 1);
  }

  // Generate one sentence as a list of tokens. `start` is a list of tokens the
  // sentence must begin with; the model continues from there. Character
  // sentences need many more tokens than word sentences.
  generateTokens(maxTokens = this.mode === 'char' ? 200 : 30, { start = [], temperature = 1 } = {}) {
    const out = [...start];
    let context = this.contextFor(out, out.length);

    while (out.length < start.length + maxTokens) {
      const token = this.sampleNext(context, temperature);
      if (token === EOS) break;
      out.push(token);
      if (this.n > 1) context = [...context.slice(1), token]; // slide the window
    }
    return out;
  }

  // Generate one sentence as readable text.
  generate(maxTokens, options = {}) {
    const text = this.join(this.generateTokens(maxTokens, options));
    return text.charAt(0).toUpperCase() + text.slice(1);
  }

  // -------------------------------------------------------------------------
  // Step 5: Explain the output by comparing it to the source.
  // -------------------------------------------------------------------------

  // True if this exact token list is a whole sentence in the training text.
  inSource(tokens) {
    return this.sentences.has(key(tokens));
  }

  // Split a token list into the fewest chunks that each appear token for
  // token somewhere in the source. Each boundary between chunks is a "jump":
  // the model left one source passage and continued from another.
  //
  // Greedy longest match: start with every place the first token occurs, then
  // keep extending one token at a time, dropping places where the next token
  // differs. When none are left, the chunk ends and a new one starts.
  segment(tokens) {
    const chunks = [];
    let i = 0;
    while (i < tokens.length) {
      let starts = this.positions.get(tokens[i]);
      if (!starts) { // token never seen at all
        chunks.push({ tokens: [tokens[i]], seen: false });
        i++;
        continue;
      }
      let len = 1;
      while (i + len < tokens.length) {
        const next = tokens[i + len];
        const kept = starts.filter(p => this.source[p + len] === next);
        if (kept.length === 0) break;
        starts = kept;
        len++;
      }
      // `from` = where the first matching passage starts in this.source.
      chunks.push({ tokens: tokens.slice(i, i + len), seen: true, from: starts[0] });
      i += len;
    }
    return chunks;
  }

  // Character range [start, end) in the training text of source tokens
  // from..to (inclusive), e.g. to highlight a copied stretch.
  sourceRange(from, to) {
    return [this.offsets[from][0], this.offsets[to][1]];
  }

  // Size of the model, and how sparse it is: of all n-grams that could exist
  // with this vocabulary, how many were actually seen?
  stats() {
    let ngrams = 0;
    for (const entry of this.table.values()) ngrams += entry.next.size;
    const types = this.positions.size;
    return {
      tokens: this.offsets.filter(Boolean).length, // skips the </s> markers
      types,
      contexts: this.table.size,
      ngrams,
      possible: Math.pow(types + 1, this.n), // +1 for </s>, which can end an n-gram
    };
  }
}

// In Node, export for require(). In a browser, the class and constants above
// are already globals once this file is loaded with <script>.
const isNode = typeof module !== 'undefined' && typeof require !== 'undefined';
if (isNode) module.exports = { NGramModel, BOS, EOS };

// ---------------------------------------------------------------------------
// Command-line demo: node ngram.js [n] [corpus.txt] [word|char]
// ---------------------------------------------------------------------------
if (isNode && require.main === module) {
  const fs = require('node:fs');
  const mode = process.argv[4] === 'char' ? 'char' : 'word';
  const n = Number(process.argv[2] ?? (mode === 'char' ? 4 : 2));
  const file = process.argv[3];

  const corpus = file
    ? fs.readFileSync(file, 'utf8')
    : `The cat sat on the mat. The dog sat on the log. The cat saw the dog.
       The dog saw the cat on the mat. A cat and a dog sat together.`;

  const model = new NGramModel(n, { mode }).train(corpus);
  const st = model.stats();
  console.log(`${mode} mode, n = ${n}: ${st.tokens} tokens, ${st.types} types, ` +
              `${st.ngrams} distinct n-grams seen of ${st.possible} possible\n`);

  for (let i = 0; i < 5; i++) {
    const tokens = model.generateTokens();
    const copied = model.inSource(tokens) ? '  (copied verbatim)' : '';
    const text = model.join(tokens);
    console.log('  ' + text.charAt(0).toUpperCase() + text.slice(1) + copied);
  }
}
