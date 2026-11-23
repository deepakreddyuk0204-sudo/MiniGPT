# MiniGPT model card

MiniGPT is an educational English next-byte predictor built with PyTorch. It uses 1,298,880 trainable parameters, all initialized locally from scratch. The intended use is to inspect the decoder, reproduce a bounded experiment and study how a held-out loss relates to generated text. Codex generated the initial code, tests and documentation. This card records measurements from that implementation.

## Training material

The corpus is Lewis Carroll’s *Alice’s Adventures in Wonderland*, from a pinned Standard Ebooks edition. The original text is public domain in the UK. The publisher dedicates its editorial contributions under CC0 1.0. The preparation script verifies both the source XHTML and prepared bytes. Chapters 1–10 supply 120,930 training bytes. Chapter 11 supplies 10,351 validation bytes and chapter 12 supplies 11,559 test bytes. No training window spans chapter boundaries. [Source, attribution, rights, preprocessing and checksums](DATA.md)

## Recorded experiment

| Setting | Value |
| --- | --- |
| Hardware and software | Apple M4 Pro, 20 GPU cores, 24 GiB unified memory, macOS 27.2.1, Python 3.12.14, PyTorch 2.15.0, MPS, float32 |
| Decoder | 4 pre-normalized blocks, width 160, 4 heads of width 40, feed-forward width 640, GELU, learned absolute positions, tied input/output token weights |
| Tokenizer and context | 256 UTF-8 byte values, 128 bytes of context |
| Optimizer | AdamW, constant learning rate 0.001, betas (0.9, 0.999), epsilon 1e-8, weight decay 0.01, gradient norm clipped to 1.0 |
| Sampling for training | Seed 2026, batch size 8, random windows with replacement, dropout 0 |
| Budget and outcome | 600 updates or 600 seconds. Reached step 600 after 614,400 target presentations, approximately 5.08 times the size of the training text |
| Timing | 8.51 s training command wall time including preflight, 7.43 s training loop through final validation, 1.48 s separate evaluation and sample generation |
| Checkpoint selection | Final scheduled checkpoint. Validation used at step 0 and every 100 updates. Test chapter scored after training. |

The preflight performed four full-size updates, including AdamW state allocation, before a fresh training initialization. Peak sampled Metal driver allocation was 128.75 MiB, process peak RSS was 442.77 MiB, and the GPU’s recommended working set was 17.76 GiB. The measurements describe one local run. [Preflight samples](../results/alice/preflight.json) · [Run metadata](../results/alice/run.json)

## Evaluation

Loss is mean natural-log cross-entropy over **every within-chapter next-byte target**. Fixed consecutive chunks reset the context every 128 inputs. Evaluation includes shorter tails, ignores right padding, and weights chunks by their number of real targets. Byte perplexity is the exponential of mean loss. The baseline always predicts the add-one-smoothed training byte frequencies.

| Split | Scored targets | Model loss | Model perplexity | Baseline perplexity |
| --- | ---: | ---: | ---: | ---: |
| Training | 120,920 | 1.8868 | 6.598 | 22.380 |
| Validation | 10,350 | 1.8923 | 6.635 | 22.843 |
| Test | 11,558 | 1.9376 | 6.942 | 22.607 |

Validation loss declined from 5.6658 at random initialization to 2.4195 at step 100 and 1.8923 at step 600. Repeated random windows mean individual passages receive unequal exposure. All splits share one novel, so the test result measures prediction of another chapter in the same narrative. Different tokenizers create different prediction units and perplexity scales. [Validation history](../results/alice/history.json) · [Exact evaluation and checkpoint hash](../results/alice/evaluation.json)

## Generated text

Both examples use the final checkpoint on CPU with seed 42, temperature 0.8, top-k 20, and 120 new bytes. Each block shows the prompt followed by its sampled continuation. The JSON record preserves the exact bytes as decoded text, including trailing whitespace.

```text
Alice sould coull thid drepthe on hared at whise feloud the hise whes all curunes it, the at canoke of sing the make shis, and
```

```text
The Rabbit as the was: and scome many, "I'lll deed gean min the cange on onse what thing yo wore do in the the he Dissss oung, not
```

The model places spaces and short common words plausibly in places. It also invents strings, repeats letters and loses grammatical and narrative coherence. One book, a byte vocabulary and a 128-byte context severely limit what it can learn. The corpus reflects its period and author. Generation may echo training passages. The model received no instruction tuning or factuality evaluation. The chat-style interface simply continues the supplied passage.

## Checks and artifacts

A smoke test took an optimizer step, checked the weights changed, reloaded matching logits, and generated four bytes. A separate one-block, width-48 diagnostic model trained repeatedly on the same 64 genuine training targets. Its loss fell from 5.5547 to 0.0366 in 40 updates against a prespecified threshold of 0.10. This verifies memorization on that batch. [Diagnostic record](../results/alice/checks.json)

The local checkpoint at `runs/alice/last.pt` includes the optimizer and random states for resumption. Its SHA-256 is `5056293ccea67615e0409124527c1bc4ff5ca38a5bf1e869bdb113d155eb4338`. Git contains the code, lockfile, source manifest and measured reports. [Setup and reproduction](../README.md) · [Architecture and equations](LEARNING.md)
