import jax.numpy as jnp
from jax import grad
import torch
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# SDF for a sphere
def sdf_sphere(x, y, z, radius=1.0):
    return jnp.sqrt(x**2 + y**2 + z**2) - radius

# Warp function
def warp_function(u, sigma=0.1):
    return 1 / (1 + jnp.exp(-u / sigma))

# Sphere tracing (PyTorch)
def sphere_tracing(ray_origin, ray_dir, max_steps=100, eps=1e-3):
    t = torch.tensor(0.0)
    pos = ray_origin
    for _ in range(max_steps):
        sdf_val = sdf_sphere(pos[0], pos[1], pos[2])
        if sdf_val < eps:
            return pos
        t += sdf_val
        pos = ray_origin + t * ray_dir
    return None  # No intersection

# Simulate rendering with Area Sampling
def simulate_rendering(dim=(64, 64), sigma=0.1):
    x = jnp.linspace(-1.5, 1.5, dim[1])
    y = jnp.linspace(-1.5, 1.5, dim[0])
    xx, yy = jnp.meshgrid(x, y)
    zz = jnp.zeros_like(xx)
    sdf_values = sdf_sphere(xx, yy, zz)
    warped_values = warp_function(sdf_values, sigma)
    return warped_values

# Visualize 3D reconstruction
def visualize_reconstruction(warped_values, dim=(64, 64)):
    x = np.linspace(-1.5, 1.5, dim[1])
    y = np.linspace(-1.5, 1.5, dim[0])
    X, Y = np.meshgrid(x, y)
    Z = warped_values  # 2D projection

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(X, Y, Z, cmap='RdBu', edgecolor='none')
    ax.set_title('3D Reconstruction from 2D Images')
    ax.set_xlabel('x'); ax.set_ylabel('y'); ax.set_zlabel('Warped SDF')
    plt.savefig('reconstruction_3d.png', dpi=300, bbox_inches='tight')
    plt.close()

# Example usage
if __name__ == "__main__":
    # Simulate rendering
    warped_values = simulate_rendering()
    
    # Visualize
    visualize_reconstruction(warped_values)
    
    # Compute gradients (JAX)
    grad_warp = grad(warp_function)
    u = jnp.array(0.0)
    grad_val = grad_warp(u)
    print(f"Gradient at u=0: {grad_val}")