**AI for Psychological Measurement**

This repository contains deep learning experiments associated with my evolving book on AI for psychological measurement, available for free at https://psychometrics.ai. I have used AI assistance to better understand deep learning models that I explain how to use in psychological measurement contexts. 

So far, the repo includes:
- New methods including **Semantic Item Alignment**, **Pseudo-Factor-Analysis** and **LLM Convex Hulls**,
- A **decoupled reconstruction of a MiniLM encoder**,
- A **decoupled reconstruction of the GPT-2 small decoder**,
- The code for **Seedling**.

**Seedling** is a GPT-2–style, ~50 million parameter LLM, coded from scratch in PyTorch with AI assistance. You can read more about its architecture and development process on the book's website. The name Seedling reflects both the early inchoate state of this model (loss = 4.63) and the hope that with a bit of TLC, this model will become a strong educational resource for quantitative psychologists. While the loss is still high, the smaller model size and single-file design (with separate testing script) make it accessible. Next steps may include modularizing the code like Andrej Karpathy’s excellent examples, and expanding the architecture and data exposure.

![Seedling Loss](images/Seedling-loss-MeasureCo.ai.png)

Everything here is shared under a Creative Commons Attribution-NonCommercial 4.0 International License. You're very welcome to use the ideas, methods, and tools in your own work, whether that’s product development, internal tools, client services, or research. All I ask is that you don’t sell or repackage this content as-is, and you credit psychometrics.ai when using it: Guenole, N. (2025). Psychometrics.ai: AI for Psychological Measurement. https://psychometrics.ai. 

[![License: CC BY-NC 4.0](https://img.shields.io/badge/License-CC%20BY--NC%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by-nc/4.0/)





