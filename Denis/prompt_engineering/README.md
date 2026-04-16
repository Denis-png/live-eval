# Prompt engineering

## Sampling

Samples from the C4_200M-*.tsv files. Two options: demo(100 sentences) and main run(1000 sentences)


## Generation

Generating new errored sentences via defined LLM (default 'nvidia/nemotron-3-super-120b-a12b:free' over openrouter). 
Implemented different prompting strategies: zero-shot, one-shot, few-shot-3(3 most common ERRANT error types), few-shot-12(all ERRANT error types), Chain-Of-Thought(CoT).


## Evaluation
Similarity
CoLA scores (A,B,C) for original(A), generated(B) and corrected(C) sentences, asserting on B < A and C > B
ERRANT F0.5 scores (main metric)


## Ablation
Experimenting with changing one of the variables at a time: temperature, instruction wording, number of examples and CoT usage. Baseline is few-shot-12 with temperature of 0.7.

