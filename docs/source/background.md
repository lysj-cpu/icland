# 🔍 Background

## 🌟 Reinforcement Learning
Reinforcement Learning (RL) has been a transformative force in artificial intelligence, allowing systems to surpass human performance in tasks such as playing [Go](https://deepmind.google/research/breakthroughs/alphago/) or [Atari video games](https://deepmind.google/discover/blog/agent57-outperforming-the-human-atari-benchmark/).

Unlike traditional rule-based AI, RL agents can adapt to dynamic environments and discover new strategies that human designers might never have considered. This makes RL particularly powerful for solving complex, real-world problems.

##  Generalisation in RL
Current RL implementations suffer from a major limitation: they struggle to generalise across different tasks. When faced with a new problem, RL agents typically need to undergo the entire training process from scratch. This results in:

- ⏳ **Inefficiency** – Slow learning curve
- 📉 **Inflexibility** – Poor adaptability to new tasks
- 🔒 **Limited Real-World Use** – Hindered deployment

## XLand - Google DeepMind
[DeepMind’s XLand](https://deepmind.google/discover/blog/generally-capable-agents-emerge-from-open-ended-play/) addresses this challenge by training RL agents in a massively diverse, procedurally generated environment.

Rather than overfitting to a single task, agents are exposed to a wide range of scenarios, allowing them to develop generalised strategies that can be applied to new and unseen challenges. This approach moves RL closer to the goal of [artificial general intelligence (AGI)](https://en.wikipedia.org/wiki/Artificial_general_intelligence).

## ICLand - The Open Source Alternative
RL research remains largely dominated by well-funded organisations such as Google, with XLand being proprietary software.

To bridge this gap, we present an open-source alternative, leveraging:

- 🚀 [JAX](https://docs.jax.dev/en/latest/) → Scalable computation
- ⚡️ [Mujoco-XLA](https://mujoco.readthedocs.io/en/stable/mjx.html) → High-performance physics simulation

By enabling massive parallelisation, we allow independent researchers to explore and contribute to RL advancements without requiring vast computational resources.
