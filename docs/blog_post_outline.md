# Blog Post Outline

**Title:** How I Built a 1B Mathematical Reasoning LLM from Scratch on a TPU v4-32 Cluster in 3 Weeks

## 1. The Hook (Introduction)
*Draft:* "We live in an era where massive tech conglomerates train frontier models on tens of thousands of GPUs. But what happens when a single engineer with a limited cloud budget decides to build a language model entirely from scratch? Not just fine-tuning, but the full stack—from raw unwashed web data to a fully functioning 1.12 Billion parameter mathematical reasoning engine. This is the story of TinyMathReason-1B, an intensive 3-week sprint utilizing a preemptible TPU v4-32 cluster, PyTorch, JAX, and the fascinating world of preference optimization."

## 2. The 'Why': Motivation and Architecture
- **Key Points:** Explain the motivation (learning the entire LLM lifecycle). Why mathematical reasoning? It provides a strict, verifiable testbed for logic.
- **Details:** Introduce the architecture. Justify using GQA and SwiGLU to pack maximum reasoning power into 1.1B parameters.
- **Visual:** Architecture diagram or parameter count breakdown table.

## 3. Data is All You Need (The Pipeline)
- **Key Points:** Processing 300 Billion tokens (~1TB of text) on a single Vultr instance. The 40/35/15/10 dataset mixture.
- **Details:** The intricacies of MinHash deduplication and packing sequences to exactly 4096 tokens. The custom 32k BPE tokenizer trained specifically for math.
- **Visual:** Pie chart of the data mix.

## 4. Taming the TPU v4-32
- **Key Points:** Setting up MaxText on Google Cloud. The fear and reality of spot instance preemptions.
- **Details:** Hitting 100k+ tokens/sec. The engineering behind writing a preemption-handling daemon and successfully converting Orbax checkpoints to HuggingFace safetensors.
- **Visual:** Screenshot of the MaxText W&B loss curve plummeting.

## 5. Teaching it to Think (SFT & DPO)
- **Key Points:** The transition from TPU/JAX to GPU/PyTorch.
- **Details:** Supervised fine-tuning using TRL. The magic of Direct Preference Optimization (DPO)—how generating candidate solutions and using correctness as a reward dramatically shifted the model's logic. Brief mention of DeepSeek-R1's GRPO method.
- **Visual:** Side-by-side snippet showing the model failing before DPO, and successfully generating a pristine Chain-of-Thought after DPO.

## 6. The Results: Did it Work?
- **Key Points:** Running `lm-evaluation-harness`.
- **Details:** Compare GSM8K and MATH scores across the Base, SFT, and DPO stages. Compare against TinyLlama-1.1B. 
- **Visual:** Bar chart showing performance improvements across the training stages.

## 7. Lessons Learned & What's Next
- **Key Points:** Honest reflections on infrastructure bugs, dataset biases, and the value of end-to-end ML engineering.
- **Call to Action:** "Check out the weights on HuggingFace, explore the full codebase on GitHub, and read the comprehensive technical report linked below!"
