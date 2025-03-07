import jax

def check_jax_backend():
    device = jax.devices()[0]  # Get the first available device
    if "gpu" in device.device_kind.lower():
        print(f"JAX is using an NVIDIA GPU: {device}")
    else:
        print("JAX is running on CPU, no NVIDIA GPU detected.")

if __name__ == "__main__":
    check_jax_backend()