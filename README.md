# N-gram generator

A small n-gram language model that generates text and shows you where every
part of its output came from. Built step by step for learning, with no
libraries and no build step.

## Running it

**In the browser:** open `index.html`. That's it. Everything runs locally and
works offline.

**In a terminal** (needs Node.js):

```
node ngram.js [n] [corpus.txt] [word|char]
node ngram.js 3 book.txt
node ngram.js 5 book.txt char
```

This prints a few model stats and five generated sentences.

## Using the page

1. The page starts with Genesis 1 from the King James Bible (public domain).
   Its heavy repetition makes copied stretches and jumps easy to see. Replace
   it by pasting your own text into the left panel, or click **Load .txt**.
2. Pick **tokens** (words or characters), **n**, the number of sentences and
   the **temperature**, then click **Generate**.
3. Read the annotated output on the right. Click any word to explore it.

| Control | What it does |
|---|---|
| start with | Every sentence begins with these words, and the model continues from them. |
| tokens | Words: words and punctuation marks are tokens. Characters: every letter, digit, punctuation mark and space is a token. |
| n | The model looks at the previous n-1 tokens to pick the next one. Switching modes sets a sensible default (2 for words, 4 for characters). |
| temperature | 1.0 uses the true probabilities. Lower favors likely tokens; higher gives rare tokens more of a chance. |
| Copy | Copies the sentences on screen as plain text. |

The generated text always fits on one screen. If there is too much, the font
shrinks a little, and then sentences are dropped from the end. The summary
line says how many didn't fit.

### What the colors mean

| Color | Meaning |
|---|---|
| Green | The whole sentence appears word for word in the source. |
| Blue | A stretch longer than n tokens copied from the source. The model reproduced more than its window guarantees, so this is memorization. |
| Orange | A jump: this token never follows the stretch before it in the source. The model left one passage and continued from another. |
| No color | Stretches of n tokens or fewer. Any n tokens in a row always match the source, because that is how the model works, so they say nothing. |

### Clicking a word

- A popup shows the context the model saw and every token it could have
  picked there, with probabilities at the current temperature. The token it
  actually picked is marked with an arrow.
- Click any token in the popup to rewrite the sentence from that point. Your
  pick stays, and the model writes the rest again. Pick
  "end of sentence" to cut the sentence there.
- If the word is in a blue or green stretch, the matching passage is selected
  in the source text on the left.

### The stats line

Under the controls: how many tokens and distinct token types the source has,
and how many distinct n-grams were seen out of all that are possible. The
last number is usually tiny. That is **sparsity**: most combinations of
words never occur, even in a large text.

## How it works

All the model code is in `ngram.js`, in five steps:

1. **Tokenize.** Split the text into sentences, and each sentence into
   tokens (words or characters). Each token remembers its position in the
   original text, so the page can point back to it later.
2. **Count.** Pad every sentence with start markers `<s>` and an end marker
   `</s>`, slide a window of n tokens over it, and count which token follows
   each context of n-1 tokens. The counts live in a `Map` from context to
   `{ total, next: Map(token -> count) }`.
3. **Probabilities.** P(token | context) = count(context, token) /
   count(context). Temperature raises each count to the power 1/T and
   renormalizes.
4. **Generate.** Start from `<s>` (or the starting words), pick a next token
   at random in proportion to its probability, slide the window, and repeat
   until `</s>`.
5. **Explain.** `segment()` splits a generated sentence into the fewest
   chunks that each appear in the source, using a greedy longest match. The
   boundaries between chunks are the jumps.

`index.html` contains only the page: layout, annotation rendering, the popup
and fitting the text to the screen.

## Things to try

- **Overfitting:** load a few pages of a book and raise n from 2 to 5. Blue
  and green take over, and jumps per sentence fall toward 0.
- **Temperature:** at 0.1 nearly every sentence comes out the same. At 3 the
  text falls apart.
- **Character mode:** step n from 2 to 8. You go from gibberish, to invented
  words that look plausible, to real phrases.
- **Jumps:** click an orange word. It often had a fairly high probability.
  It only counts as new because the stretch before it never continued that
  way in the source.
- **Sparsity:** compare the stats line at n = 1, 2 and 3.

## Limitations

- Abbreviations such as "Mr." end a sentence.
- Tokenizing assumes spaces between words, so it doesn't suit languages like
  Chinese or Japanese in word mode.
- When a stretch appears several times in the source, only the first
  occurrence is highlighted.
- There is no smoothing, so anything not seen in the source has probability
  0. That's fine for generating, but it means the model can't score text it
  hasn't seen (no perplexity).
- Jumping to a source passage clears the undo history of the source text box.

## Files

```
index.html   the page
ngram.js     the model (also runs in Node)
fonts/       both fonts, bundled so the page looks the same offline and on any system
```

Fonts:

- **Nimbus Sans Narrow** for the interface. Copyright URW, licensed under
  AGPL-3 with a font exception (`fonts/NimbusSansNarrow-LICENSE.txt`).
- **Instrument Serif** for the title and the generated text. Licensed under
  the SIL Open Font License (`fonts/OFL.txt`).

If you share the project, keep the license files next to the fonts.

## License

The code (`index.html`, `ngram.js`) is released under the MIT License, see
`LICENSE`. The bundled fonts keep their own licenses, listed above.
