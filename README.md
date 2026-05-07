# playgroup_202605_make_gpt

Choose your own adventure to build (bits of) GPT in a playgroup day. Pick your level of challenge and self-organise with peers, collaborate and share as you go. None of this is fixed or 'right', but they're probably good starting points, feel free to go on your own adventure too. 

Share results as you go in the #transformers channel, particularly good visuals with notes that'd give useful intuitions to everyone else. If you find good resources like other nice tutorials or videos, feel free to raise a PR (or bug report as an easy log) against this repo so I can add them for the future.


## Level 1 - you've seen some videos and maybe run the code before

Goal - run MicroGPT (see bullet below), try to figure out all the components of what's behind it. 

* https://gist.github.com/karpathy/8627fe009c40f57531cb18360106ce95 MicroGPT code 'it should just run'
* https://bbycroft.net/llm - great interactive tutorial (I've followed it repeatedly), 1hr+, based on the bigger nanoGPT but the ideas are the same
* https://karpathy.github.io/2026/02/12/microgpt/ - friendly high level walk through for 'microgpt' (pure python minimal GPT), 1hr
* https://growingswe.com/blog/microgpt - great interactive tutorial
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

TODO FILL IN THE BLANK VARIANTS

### Visualisations you might create

* Draw the initial and learned wpe and wte matrices, how are they different to Ian's MicroGPT visualisations (https://www.linkedin.com/posts/ianozsvald_tonight-im-speaking-at-pydata-london-on-ugcPost-7457412625845575682-63wf?utm_source=share&utm_medium=member_desktop&rcm=ACoAAAABKaUBPYZfMn8VYoI5sVKX8lZuVm5-tZA) and in #transformers?
  * Ian created a dot-product wte and then a cosine similarity wpe and wte (all for MicroGPT), an audience member suggested t-sne might be a good way to group the characters that have similarity, maybe that's a good idea to try?
* Draw probability distributions for tokens likelihood over time
* Draw the before and after RMSNorm matrix to explain what it does

## Level 3

You're comfy with `pytorch` and you know enough of the Transformer architecture. Possible tasks:

* Draw the whole architecture from scratch from memory
* Implement it
* Replace ReLU, RMSNorm with e.g. GeLU and another Norm
* Add a KV Cache
