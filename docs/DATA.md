# Training corpus

MiniGPT learns from the twelve chapters of Lewis Carroll’s *Alice’s Adventures in Wonderland*. The source is the Standard Ebooks English edition at Git commit `8da5cac1a4ec2c811322eff79daa5e11e851a813`. Alex Cabal produced this edition using a Project Gutenberg transcription by Arthur DiBianca and David Widger, together with scans from the Internet Archive. Only the chapter XHTML files are downloaded. The repository records their URLs and checksums. [Pinned edition](https://github.com/standardebooks/lewis-carroll_alices-adventures-in-wonderland_john-tenniel/tree/8da5cac1a4ec2c811322eff79daa5e11e851a813) · [Edition colophon](https://github.com/standardebooks/lewis-carroll_alices-adventures-in-wonderland_john-tenniel/blob/8da5cac1a4ec2c811322eff79daa5e11e851a813/src/epub/text/colophon.xhtml) · [Per-chapter manifest](../minigpt/alice_manifest.json)

## Rights

The novel appeared in 1865 and Carroll died in 1898. The original published text is in the public domain in the UK under the usual life-plus-70-year term. Standard Ebooks identifies the source as public domain in the US and dedicates its contributors’ editorial work to the public domain worldwide under CC0 1.0. The edition’s rights page distinguishes these two grants. Readers in other jurisdictions can consult that page and their local term for the original text. [Oxford author and publication history](https://www.chch.ox.ac.uk/college/library-archives/highlights-jon-lindseth-lewis-carroll-collection) · [UK copyright term](https://www.legislation.gov.uk/ukpga/1988/48/section/12) · [Publisher’s notice](https://github.com/standardebooks/lewis-carroll_alices-adventures-in-wonderland_john-tenniel/blob/8da5cac1a4ec2c811322eff79daa5e11e851a813/src/epub/text/uncopyright.xhtml) · [CC0 dedication](https://creativecommons.org/publicdomain/zero/1.0/)

## Reproduction

Run from the project root with the project environment active.

```sh
python -m minigpt.data
```

The command retrieves each pinned XHTML file over HTTPS and checks its SHA-256 before extraction. It parses body paragraphs in document order, including chapter titles and verse, joins inline markup, and treats verse line breaks as spaces. It drops images, alt text, Roman chapter numbers, and front/back matter. It collapses whitespace inside paragraphs, removes the invisible word joiner, maps curly quotation marks to straight quotes, converts em dashes to two hyphens, en dashes to one hyphen, and ellipses to three dots. It keeps capitalization and separates paragraphs with two newlines, with one final newline per chapter.

The command checks the UTF-8 output against an independently recorded SHA-256 before writing it. It produces `data/prepared/01.txt` through `12.txt`. A second run verifies existing chapters and works offline once all twelve exist. A missing chapter is fetched on the next preparation. A changed chapter raises an error identifying the path. Move that file aside and rerun to restore the pinned version. The training loader also verifies the files. The measured manifest fingerprint is `3463e6e35af58f0258898c071227b7e74680156b99987b71c66cd3c642cb3620`.

## Size and split design

The twelve XHTML files total **186,803 bytes**. The prepared corpus contains **142,840 byte tokens**, all ASCII in this particular edition after normalization. UTF-8 tokenization also accepts Unicode prompts. The model has 256 byte IDs, of which 64 occur in the training chapters.

| Split | Chapters | Prepared bytes | Possible adjacent byte pairs |
| --- | --- | ---: | ---: |
| Training | 1–10 | 120,930 | 120,920 |
| Validation | 11 | 10,351 | 10,350 |
| Test | 12 | 11,559 | 11,558 |

Splitting by chapter first keeps validation and test windows out of the training sampler. No window crosses a chapter boundary. A training batch samples starting offsets uniformly with replacement over all valid windows in the selected split. For context length `T`, a chapter of `N` bytes contributes `max(0, N − T)` starts. Each sampled window supplies `T` inputs and the same number of one-byte-shifted targets.

Held-out chapters share the author, characters and style of the training text. This evaluation measures how well the model continues that particular book. A single short novel provides little evidence about general English, factual knowledge or conversation.
