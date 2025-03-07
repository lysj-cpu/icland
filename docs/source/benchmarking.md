# 🔥 Benchmarking

## 🎨 Rendering

Our GPU-based renderer replaces traditional mesh-based rendering with [Signed Distance Functions (SDFs)](https://en.wikipedia.org/wiki/Signed_distance_function), enabling smoother surfaces, infinite detail, and more efficient rendering. By using [ray marching](https://en.wikipedia.org/wiki/Ray_marching) instead of rasterisation or ray tracing, the system eliminates the need for complex acceleration structures while achieving real-time global illumination and soft shadows.

Additionally, it supports dynamic procedural world generation, allowing seamless terrain morphing, adaptive object placement, and real-time physics interactions. This makes it ideal for open-world games, simulations, and AI-driven environments.

## 🛠 GPU-Accelerated Model Editing

This method introduces GPU-accelerated model editing, moving beyond traditional CPU-based approaches. By harnessing GPU parallelism, model edits are processed in real-time, enabling efficient and scalable terrain modifications.

Unlike systems that only alter obstacles, our approach dynamically modifies both terrain and objects, providing a more flexible and immersive experience in interactive environments.
