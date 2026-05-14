# playgroup_202605_make_gpt - Let's Build Some of GPT

Choose your own adventure to build (bits of) GPT in a playgroup day. Pick your level of challenge and self-organise with peers, collaborate and share as you go. None of this is fixed or 'right', but they're probably good starting points, feel free to go on your own adventure too. 

Share results as you go in the #transformers channel, particularly good visuals with notes that'd give useful intuitions to everyone else. If you find good resources like other nice tutorials or videos, feel free to raise a PR (or bug report as an easy log) against this repo so I can add them for the future.

## Level 1 - you've seen some videos and maybe run the code before

Goal - run MicroGPT (see bullet below), try to figure out all the components of what's behind it. It 'should just run', you can debug it or talk about it and play with tutorials. 

* https://gist.github.com/karpathy/8627fe009c40f57531cb18360106ce95 MicroGPT code 'it should just run'
* https://bbycroft.net/llm - great interactive tutorial (I've followed it repeatedly), 1hr+, based on the bigger nanoGPT but the ideas are the same
* https://karpathy.github.io/2026/02/12/microgpt/ - friendly high level walk through for 'microgpt' (pure python minimal GPT), 1hr

### Resources for MicroGPT

* https://growingswe.com/blog/microgpt - great interactive tutorial
* https://speakerdeck.com/ianozsvald/build-your-own-llm-live-with-microgpt - Ian's PyDataLondon 2026 talk on MicroGPT
* https://www.reddit.com/r/learnmachinelearning/comments/1r3qaky/andrej_karpathys_microgpt_architecture_stepbystep/ - nice visual overview
* https://www.youtube.com/watch?v=KJtZARuO3JY&t=2671s - lovely 3blue1brown 45min intro to Attention (not specific to microGPT)
* https://www.youtube.com/watch?v=kCc8FmEb1nY - excellent 2 hour video building up nanoGPT (karpathy's pytorch based GPT) which explains all the concepts, min 2hr
* https://www.youtube.com/watch?v=KJtZARuO3JY&t=2671s - attention talk by 3blue1brown, builds great intuitions, 1 hr

### Good questions you might want to discuss

* What's the purpose of `wpe` and `wte`?
* What are K V Q for?
* What is the start state for all the parameters?
* What is encoded as you add Layers?
* Why are multiple Heads on a Layer useful?
* How come each Head can learn a different thing?
* What's different between the K V Q embeddings and the `LMHead` output?
* How do you go from making a prediction and calculating an error to modifying the parameters?

## Level 2

Goal - run the included nanoGPT, fill in the blanks (SEE BELOW)

First - go into `nanogpt_cleaned`, this is a cleaned-up variant of Karpathy's excellent `nanogpt`. This one is hardcoded to work on cpu (not gpu) shakespeare character text generation, with the multi-GPU and GPT2 specific code removed. There's a `README.md` in that folder, get setup and try running the code.

Karpathy's code uses the MIT license, I've modified his code and continue the MIT license here. Full respect to Karpathy.

### Good questions to answer before you start

* In `train.py` how many layers and heads do we have? What is the context length (Time)? What's the embedding size? Start with CPU mode (the default)
  * Contrast with the bbycroft visualisation link (above)
  * You might want to make a note of the relevant variable names
* How big are the Q K V matrices?
* Where is self attention calculated in `model.py`? Can you grok it?

### Modified `model.py` files for you to fill in

A great challenge is to _write code_ to fill in the blanks. The following files are direct copies of `model.py` with particular items removed. Can you recover the missing code, without copying it and without asking an agent to write it for you?

Nothing runs these files, you need to move your `model.py` aside (e.g. as `model_orig.py`) and then copy e.g. `cp model_missing_layernorm.py model.py` and then fill in what's missing, then confirm it runs to train and you can still sample. See `nanogpt_cleaned/EXPECTED_OUTPUT.md` for examples of what to expect.

In each file below look for "MISSING" and you'll see what needs completing.

* `model_missing_wpewte.py` you need to get `wte` and `wpe` and combine them in `GPT.forward`, otherwise nothing goes into the Transformer
* `model_missing_layernorm.py` missing 1 line of layer norm - can you put it back?
* `model_missing_mlp.py` MLP's calculations are missing in `forward`, can you fix it?
* `model_missing_block.py` the Block can't calculate the Transformer step, can you fill in the gaps?
* `model_missing_attn.py` missing a few attention calculation steps - can you complete the calculation of attention?

### Visualisations you might create

* Draw the initial and learned `wpe` and `wte` matrices for nanoGPT (which has a similarly small character vocab), how are they different to Ian's MicroGPT visualisations (https://www.linkedin.com/posts/ianozsvald_tonight-im-speaking-at-pydata-london-on-ugcPost-7457412625845575682-63wf?utm_source=share&utm_medium=member_desktop&rcm=ACoAAAABKaUBPYZfMn8VYoI5sVKX8lZuVm5-tZA) and in #transformers?
  * Ian created a dot-product wte and then a cosine similarity wpe and wte (all for MicroGPT), an audience member suggested t-sne might be a good way to group the characters that have similarity, maybe that's a good idea to try?
  * https://speakerdeck.com/ianozsvald/build-your-own-llm-live-with-microgpt?slide=8 cosine similarity for vowel similarity
  * does NanoGPT on Shakespeare show vowel self-similarity for the `wte`?
* Draw probability distributions for tokens likelihood over time, moving from equally likely to peaks
* Draw the before and after RMSNorm matrix to explain what it does
* Look into a 'logit probe' to convert earlier Transformer layers back to LMHead
* The `Adam` optimizer has a pair of params per model param - how do they evolve over time?

### Resources

* https://github.com/stanford-cs336/assignment1-basics/blob/main/cs336_assignment1_basics.pdf - Stanford course on writing an LLM
* The books that Ian will bring
* https://github.com/raiyanyahya/how-to-train-your-gpt

## Level 3

You're comfy with `pytorch` and you know enough of the Transformer architecture. Possible tasks:

* Draw the whole architecture from scratch and then implement what you can (i.e. "just do what Karpathy does"!)
  * OR take the existing code and delete each `forward` function `model.py` and fill them back in by hand
* Take the Stanford "build your own LLM code" course (above) and follow it
* Replace ReLU, RMSNorm with e.g. GeLU and another Norm
* Add a KV Cache, measure a speed increase at inference time
* If in `model.py` we set `self.flash = True`, how much faster is it than the current non-optimised version?
* Swap the Muon optimiser in place of Adam, does it run faster? Does loss come down faster?
