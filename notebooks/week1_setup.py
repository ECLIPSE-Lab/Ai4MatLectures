# %% [markdown]
# # Connect VS Code to JupyterHub (HPC Portal)
#
# **Prerequisite:** You need an HPC portal account.
#
# ## A) Start JupyterHub session in the browser
# 1. Open the HPC Portal: https://portal.hpc.fau.de/
# 2. Go to the **User** page.
# 3. Under **Your accounts**, select the account you want to use.
# 4. Click **Go to JupyterHub**.
# 5. Accept the Terms of Service if prompted.
# 6. On a page like `https://hub.hpc.fau.de/tier3-jupyter/hub/spawn/...`,
#    choose **1x GPU, 4 hours** and click **Start**.
# 7. In the JupyterLab launcher (URL similar to
#    `https://hub.hpc.fau.de/tier3-jupyter/user/kimt100v/lab`),
#    select **Notebook -> PyTorch 2.6.0**.
# 8. Clone this repository in JupyterLab (did not work for me, we will fix this later):
#    `Git -> Clone a repository -> https://github.com/ECLIPSE-Lab/Ai4MatLectures.git`
#
# ## B) Get your JupyterHub token
# 9. Open: https://hub.hpc.fau.de/tier3-jupyter/hub/token
# 10. Click **Request API token** and keep that tab open.
#
# ## C) Connect from local VS Code
# 11. Open VS Code on your local machine.
# 11.1 Clone the Ai4MatLectures repository: https://github.com/ECLIPSE-Lab/Ai4MatLectures.git 
# 12. Open this project and then open the Week 1 notebook/script.
# 13. In the kernel selector (top-right), choose:
#     **Existing JupyterHub Server**.
# 14. Enter:
#     - server URL: `https://hub.hpc.fau.de/tier3-jupyter/user/kimt100v`
#     - user: `<your_hpc_username>`
#     - token: `<token from step 10>`
# 15. Select the **PyTorch 2.6** kernel.
#
# You are now connected to an HPC node (e.g., GTX1080Ti, ~125 GB RAM).

# %%
import torch
import matplotlib.pyplot as plt

# %%
import os
os.system('nvidia-smi')

# %%
os.system('pwd')

# %% [markdown]
# # Week 1 Learning Outcomes
# By the end of this notebook, you will be able to:
# 1. **Connect & Setup:** Connect your local VS Code to HPC JupyterHub resources.
# 2. **Tensor Operations:** Manipulate ML datasets using PyTorch tensors (indexing, slicing, broadcasting).
# 3. **Data Processing:** Apply advanced array slicing to filter experimental data (e.g., HRTEM microscopy artifacts).
# 4. **Linear Algebra:** Scale and normalize features to stabilize Machine Learning models.
# 5. **Autograd & Optimization:** Understand PyTorch's automatic differentiation and construct a manual training loop.
# 6. **Model Generalization:** Visually identify underfitting vs. overfitting in regression models.
# 
# ---
# 
# # Part 1: PyTorch Tensors and Data Manipulation
# Welcome to your first exercise! In Materials Informatics, we represent materials data 
# (like crystal structures, composition, or property targets) using multi-dimensional arrays, 
# which we call **tensors** in PyTorch. 
# 
# Let's start with basic tensor operations: creating tensors, indexing, and slicing.

# %%
# Create a 1D tensor (e.g., a vector of phase compositions)
composition = torch.tensor([0.2, 0.3, 0.5])
print(f"Composition vector: {composition}")

# Create a 2D tensor (e.g., experimental features for multiple materials)
# 3 materials, 4 features each
features = torch.rand(3, 4)
print(f"Random Features:\n{features}")

# Let's examine the tensor properties
print(f"Shape: {features.shape}")
print(f"Number of elements: {features.numel()}")
print(f"Data type: {features.dtype}")

# %% [markdown]
# ## Indexing and Slicing
# Advanced slicing allows you to extract specific parts of your dataset rapidly. 
# This is crucial when processing large experimental datasets.

# %%
# Get the first material (first row)
print(f"Material 0 features: {features[0, :]}")

# Get the second feature across all materials (second column)
print(f"Feature 1 for all materials: {features[:, 1]}")

# Slicing: get the first two materials and their last two features
print(f"Subset (Materials 0-1, Features 2-3):\n{features[:2, -2:]}")

# %% [markdown]
# ## Real-world Data: Working with Microscopy Images
# In Materials Informatics, experimental data like high-resolution Transmission Electron Microscopy (HRTEM) 
# images are extremely common. Under the hood, an image is simply a 2D array (or tensor) of pixel intensities.
# Let's load a real material micrograph and see how basic slicing and indexing concepts apply to real data.

# %%
filename = '../data/hrtem/image_000.tiff'
import imageio
im = imageio.imread(filename)
print(im.shape)
 
plt.imshow(im, cmap='gray')
plt.title('im')
plt.colorbar()
plt.show()
#%%
import numpy as np

# Compute the 2nd and 98th percentiles for robust display
vmin, vmax = np.percentile(im, [2, 98])

plt.imshow(im, cmap='gray', vmin=vmin, vmax=vmax)
plt.title('im (2nd to 98th percentile)')
plt.colorbar()
plt.show()

# %% [markdown]
# ### Application: Advanced Slicing and Filtering
# Experimental data such as high-resolution microscopy images often contain artifacts—like intense bright spots from sensor noise or stray radiation.
# Identifying and cleaning up these specific pixels is a perfect application of **advanced array slicing and boolean indexing**.
# 
# Instead of using slow `for` loops across millions of pixels, we can use a boolean mask condition (e.g., `image > cutoff`) 
# to instantly "slice out" the properties of interest. We can then efficiently locate, extract, and replace only these problematic pixels.

# %%
def remove_intensity_outliers(image: np.ndarray, threshold: float = 3000) -> np.ndarray:
    """
    Step-by-step outlier cleanup (x-ray artifact removal):
    1) Validate the input.
    2) Convert to float and copy so the original image stays unchanged.
    3) Clip negative values to 0.
    4) Detect bright outliers using: median(image) + threshold.
    5) Process strongest outliers first.
    6) Replace each outlier:
       - border pixels -> global image mean
       - inner pixels  -> mean of 8 neighbors
    7) Run a second detection pass and report remaining outliers.
    """
    # Step 1: Make sure the function receives a NumPy array.
    if not isinstance(image, np.ndarray):
        raise TypeError('Input must be numpy ndarray.')

    # Step 2: Convert to float for safe averaging and create a working copy.
    cleaned = image.astype(np.float32).copy()

    # Step 3: Clamp any negative values (can appear in some acquisition pipelines).
    cleaned[cleaned < 0] = 0

    # Step 4: Define a robust outlier cutoff relative to the median intensity.
    cutoff = np.median(cleaned) + threshold

    # Step 5: Locate candidate outlier pixels and collect their intensities.
    bad_loc = np.argwhere(cleaned > cutoff)
    bad_vals = cleaned[cleaned > cutoff]
    print(f'Number of detected outliers (pass 1): {len(bad_loc)}')

    # Step 6: Sort outliers from brightest to dimmest.
    # Why? Replacing the strongest spikes first stabilizes local neighborhoods.
    sorted_ind = np.argsort(bad_vals)
    bad_loc_sorted = np.flip(bad_loc[sorted_ind, :], axis=0)

    # Step 7: Replace each outlier pixel with a local estimate.
    for row, col in bad_loc_sorted:
        # Border pixels do not have 8 valid neighbors -> use global mean fallback.
        if row == 0 or col == 0 or row == cleaned.shape[0] - 1 or col == cleaned.shape[1] - 1:
            new_pixel_int = cleaned.mean()
        else:
            # Extract the 3x3 neighborhood centered at (row, col).
            neighborhood = cleaned[row - 1:row + 2, col - 1:col + 2]
            # Average the 8 surrounding pixels (exclude center pixel itself).
            new_pixel_int = (neighborhood.sum() - cleaned[row, col]) / 8.0
        cleaned[row, col] = new_pixel_int

    # Step 8: Re-check after one correction pass.
    cutoff = np.median(cleaned) + threshold
    bad_loc = np.argwhere(cleaned > cutoff)
    print(f'Number of detected outliers (pass 2): {len(bad_loc)}')

    # Step 9: Return the cleaned image.
    return cleaned

# %% [markdown]
# ### Slicing Images into Patches for Machine Learning
# Modern machine learning research in materials science heavily relies on Deep Neural Networks (like CNNs or Vision Transformers). 
# However, these models cannot typically process massive high-resolution micrographs all at once due to GPU memory constraints.
# 
# A standard technique is to use **spatial slicing** to divide a large, cleaned image into smaller, uniform regions 
# (e.g., 512x512 pixels). This process solves the memory limitation while taking advantage of array indexing concepts (such as `cropped[start:end, start:end]`).
# It also acts as "data augmentation", artificially increasing the size of our dataset by turning one giant image into hundreds of distinct training examples.
#
# Let's write a function to extract non-overlapping patches from an image plane using array slicing:

# %%
def slice_into_patches(image: np.ndarray, patch_size: int = 512) -> np.ndarray:
    """
    Step-by-step non-overlapping patch extraction:
    1) Read image height/width.
    2) Compute how many full patches fit along each axis.
    3) Crop away right/bottom remainder if dimensions are not divisible.
    4) Loop over grid positions and slice each patch.
    5) Stack patches into one NumPy array with shape (N, patch_size, patch_size).
    """
    # Step 1: Image size.
    h, w = image.shape

    # Step 2: Number of complete patches that fit.
    n_h = h // patch_size
    n_w = w // patch_size

    # Step 3: Keep only the region that fits an exact patch grid.
    cropped = image[:n_h * patch_size, :n_w * patch_size]

    # Step 4: Slice patches row-by-row.
    patches = []
    for j in range(n_h):
        for k in range(n_w):
            patch = cropped[j * patch_size:(j + 1) * patch_size, k * patch_size:(k + 1) * patch_size]
            patches.append(patch)

    # Step 5: Convert list -> array for easier downstream ML processing.
    return np.asarray(patches)
#%%
# Plot histogram of intensities to inform outlier threshold selection
plt.figure(figsize=(8, 4))
plt.hist(im.flatten(), bins=256, color='mediumblue', alpha=0.75)
plt.xlabel('Pixel Intensity')
plt.ylabel('Frequency')
plt.title('Image Intensity Histogram')
plt.grid(True, linestyle='--', alpha=0.3)
plt.tight_layout()
plt.show()
#%%
# Remove bright intensity outliers and create 512x512 slices
im_clean = remove_intensity_outliers(im, threshold=200)
plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.imshow(im, cmap='gray')
plt.title('Original im[0]')
plt.colorbar(fraction=0.046, pad=0.04)

plt.subplot(1, 2, 2)
plt.imshow(im_clean, cmap='gray')
plt.title('Cleaned im_clean[0]')
plt.colorbar(fraction=0.046, pad=0.04)

plt.tight_layout()
plt.show()

patches_512 = slice_into_patches(im_clean, patch_size=1024)

print(f'Cleaned image shape: {im_clean.shape}')
print(f'Patch tensor shape (N, H, W): {patches_512.shape}')

# Visualize first patch as an indexing/slicing example
if len(patches_512) > 0:
    first_patch = patches_512[0]  # same as patches_512[0, :, :]
    
    plt.figure(figsize=(5, 5))
    plt.imshow(first_patch, cmap='gray')
    plt.title('First 512x512 patch after outlier removal')
    plt.colorbar()
    plt.show()


# %% [markdown]
# # Part 2: Linear Algebra Operations
# Machine learning is largely based on applied linear algebra. Operations like dot products,
# matrix-vector multiplications, and matrix-matrix multiplications are used extensively 
# for data transformations and model predictions.

# %%
# Let's evaluate a linear model. We create a weight vector for our 4 features
weights = torch.tensor([0.5, -0.2, 0.1, 0.8])

# Matrix-vector multiplication (dot products for each material)
# You can use torch.mv, torch.matmul, or the @ operator
predictions = features @ weights
print(f"Predictions:\n{predictions}")

# Element-wise operations (e.g., scaling features)
scaled_features = features * 2.0
print(f"Scaled Features:\n{scaled_features}")

# %% [markdown]
# ## Application: Data Normalization
# Neural networks train more stably and efficiently when input features (like image pixel intensities) 
# are on a similar scale, typically `[-1, 1]` or `[0, 1]`. 
# This simple affine transformation is a fundamental element-wise linear algebra operation.

# %%
def scale_to_minus1_plus1(image_array: np.ndarray) -> np.ndarray:
    """
    Min-max scaling to [-1, 1]:
      x_scaled = 2 * (x - x_min) / (x_max - x_min) - 1
    """
    x = image_array.astype(np.float32)
    x_min = np.min(x)
    x_max = np.max(x)

    # Avoid division by zero for constant images.
    if np.isclose(x_max, x_min):
        return np.zeros_like(x, dtype=np.float32)

    x_scaled = 2.0 * (x - x_min) / (x_max - x_min) - 1.0
    return x_scaled


# Example 1: scale the cleaned full image
im_scaled = scale_to_minus1_plus1(im_clean)
print(f'im_scaled range: [{im_scaled.min():.3f}, {im_scaled.max():.3f}]')

# Example 2: scale all image patches at once (shape stays: N, H, W)
patches_512_scaled = np.asarray([scale_to_minus1_plus1(p) for p in patches_512], dtype=np.float32)
print(f'patches_512_scaled shape: {patches_512_scaled.shape}')
print(f'first patch range: [{patches_512_scaled[0].min():.3f}, {patches_512_scaled[0].max():.3f}]')

# Convert to PyTorch tensor and add channel dimension for CNN input: (N, 1, H, W)
patches_tensor = torch.from_numpy(patches_512_scaled).unsqueeze(1)
print(f'patches_tensor shape (N, C, H, W): {patches_tensor.shape}')

# Quick visual check: first scaled patch
plt.figure(figsize=(5, 5))
plt.imshow(patches_512_scaled[3], cmap='gray', vmin=-1, vmax=1)
plt.title('First patch scaled to [-1, 1]')
plt.colorbar()
plt.show()

# %% [markdown]
# # Part 3: Automatic Differentiation (Autograd)
# Calculating gradients (derivatives) is the core mechanism used to optimize machine learning models.
# PyTorch allows you to track computations to automatically calculate gradients using `autograd`.

# %%
# Example 1: scalar autograd
# Let's track a scalar x
x = torch.tensor(3.0, requires_grad=True)

# Define a function y = x^2 + 2x + 1
y = x**2 + 2*x + 1

# Calculate the gradient dy/dx
y.backward()

# Since y = x^2 + 2x + 1, dy/dx = 2x + 2. Evaluated at x=3, the gradient should be 8.
print(f"Gradient dy/dx at x=3: {x.grad}")

# %%
# Example 2: vector gradient and verification
# We want gradients of y = 2 * x^T x with respect to vector x.
x_vec = torch.arange(4.0, requires_grad=True)
y_vec = 2 * torch.dot(x_vec, x_vec)  # scalar output
y_vec.backward()

print(f"x_vec: {x_vec}")
print(f"Autograd gradient: {x_vec.grad}")
print(f"Expected gradient (4*x): {4 * x_vec}")
print(f"Match check: {torch.allclose(x_vec.grad, 4 * x_vec)}")

# %%
# Example 3: gradient accumulation and reset
# By default, PyTorch accumulates gradients in .grad.
x_acc = torch.tensor(2.0, requires_grad=True)

y1 = x_acc**2
y1.backward()
print(f"After first backward, grad = {x_acc.grad}")  # 2*x = 4

y2 = 3 * x_acc
y2.backward()
print(f"After second backward (accumulated), grad = {x_acc.grad}")  # 4 + 3 = 7

# Reset gradients before a new optimization step.
x_acc.grad.zero_()
y3 = x_acc**3
y3.backward()
print(f"After zero_ and new backward, grad = {x_acc.grad}")  # 3*x^2 = 12

# %%
# Example 4: non-scalar outputs
# If output is a vector, reduce to scalar (sum) or pass a gradient vector.
x_non_scalar = torch.arange(4.0, requires_grad=True)
y_non_scalar = x_non_scalar * x_non_scalar  # vector output

y_non_scalar.sum().backward()  # equivalent to backward(ones)
print(f"Gradient of x^2 summed over elements: {x_non_scalar.grad}")  # 2*x

# %%
# Example 5: detach part of the graph
# detach() stops gradients from flowing through a branch.
x_detach = torch.arange(4.0, requires_grad=True)
y_detach = x_detach * x_detach
u = y_detach.detach()     # no gradient history
z = u * x_detach          # z depends on x_detach directly and on detached u

z.sum().backward()
print(f"Gradient with detached branch: {x_detach.grad}")
print(f"Should match detached u: {u}")

# %%
# Example 6: autograd with Python control flow
def piecewise_scale(a):
    b = a * 2
    while b.abs() < 1000:
        b = b * 2
    if b > 0:
        c = b
    else:
        c = 100 * b
    return c


a = torch.randn(size=(), requires_grad=True)
d = piecewise_scale(a)
d.backward()
print(f"a = {a.item():.6f}")
print(f"d = {d.item():.6f}")
print(f"Gradient dd/da = {a.grad.item():.6f}")

# %% [markdown]
# # Part 4: Fitting a Polynomial and Overfitting vs. Underfitting
# Let's apply tensor operations and autograd to a classic problem: curve fitting.
# Imagine we have experimental data relating temperature (x) to the yield strength (y) of a steel alloy.
# We want to find a polynomial that fits this data using Gradient Descent.
# 
# **Underfitting**: A model is too simple (e.g., a straight line) and cannot capture the underlying trend.
# **Overfitting**: A model is too complex (e.g., a high-degree polynomial) and perfectly fits the noise in the data, failing to generalize to new temperatures.

# %%
import matplotlib.pyplot as plt

# 1. Generate synthetic "experimental" data
torch.manual_seed(42)

# Temperatures (normalized between -1 and 1 for numerical stability)
x_data = torch.linspace(-1, 1, 20)

# True relationship (3rd degree):
# y = 2 + 1.5*x - 0.5*x^2 + 1.2*x^3 + measurement noise
true_coeffs = torch.tensor([2.0, 1.5, -0.5, 16.2])
y_data = (
    true_coeffs[0]
    + true_coeffs[1] * x_data
    + true_coeffs[2] * (x_data**2)
    + true_coeffs[3] * (x_data**3)
    + torch.randn(20) * 3.2
)

# %% [markdown]
# ### The Standard ML Training Loop
# Notice the explicit 5-step process in the `fit_polynomial` function below. This pattern is the foundation of almost all Deep Learning model training:
# 1. **Forward pass**: Make predictions based on current weights.
# 2. **Calculate Loss**: Measure how wrong the predictions are (e.g., Mean Squared Error).
# 3. **Backward pass**: Use PyTorch `autograd` (`loss.backward()`) to calculate the gradients for each weight.
# 4. **Update weights**: Adjust the weights slightly in the opposite direction of the gradient.
# 5. **Zero gradients**: Clear the accumulated gradients for the next iteration (`grad.zero_()`).

# %%
# 2. Let's try to fit a polynomial of degree M using Gradient Descent
def fit_polynomial(x, y, degree, learning_rate=0.1, epochs=1000):
    # Initialize weights randomly, requiring gradients
    weights = torch.randn(degree + 1, requires_grad=True)
    
    # Optimizer loop (we will manually do this to understand the mechanism)
    for epoch in range(epochs):
        # Forward pass: calculate predictions
        # y_pred = w0 + w1*x + w2*x^2 + ... + wM*x^M
        y_pred = torch.zeros_like(x)
        for i in range(degree + 1):
            y_pred += weights[i] * (x ** i)
        
        # Calculate loss (Mean Squared Error)
        loss = torch.mean((y_pred - y) ** 2)
        
        # Backward pass: calculate gradients automatically!
        loss.backward()
        
        # Update weights using gradient descent (disable gradient tracking for the update)
        with torch.no_grad():
            weights -= learning_rate * weights.grad
            # Clear gradients for the next iteration
            weights.grad.zero_()
            
    return weights

# We test three degrees:
# - Degree 1: too simple (underfit)
# - Degree 3: matches the true generating process (best fit)
# - Degree 8: overly flexible (overfit)
weights_under = fit_polynomial(x_data, y_data, degree=1, learning_rate=0.1)
weights_best = fit_polynomial(x_data, y_data, degree=3, learning_rate=0.1)
weights_over = fit_polynomial(x_data, y_data, degree=12, learning_rate=0.5, epochs=10000) # Higher LR/epochs to force overfitting

# %% [markdown]
# Now let's visualize the results to understand underfitting and overfitting.

# %%
# Evaluate the polynomials on a dense grid for smooth plotting
x_dense = torch.linspace(-1.1, 1.1, 100) # Go slightly beyond data to show extrapolation

def predict(x_vals, weights):
    y_pred = torch.zeros_like(x_vals)
    for i in range(len(weights)):
        y_pred += weights[i] * (x_vals ** i)
    return y_pred

# Compute training-set MSE for each fitted polynomial
pred_under = predict(x_data, weights_under).detach()
pred_best = predict(x_data, weights_best).detach()
pred_over = predict(x_data, weights_over).detach()

mse_under = torch.mean((pred_under - y_data) ** 2).item()
mse_best = torch.mean((pred_best - y_data) ** 2).item()
mse_over = torch.mean((pred_over - y_data) ** 2).item()

plt.figure(figsize=(10, 6))
plt.scatter(x_data, y_data, color='black', label='Experimental Data', zorder=5)

plt.plot(
    x_dense,
    predict(x_dense, weights_under).detach(),
    label=f'Degree 1 (Weak baseline: underfit, MSE={mse_under:.4f})',
    linestyle='--',
    color='tab:gray',
    alpha=0.6
)
plt.plot(
    x_dense,
    predict(x_dense, weights_best).detach(),
    label=f'Degree 3 (Prominent: true model, MSE={mse_best:.4f})',
    linewidth=3.5,
    color='tab:green'
)
plt.plot(
    x_dense,
    predict(x_dense, weights_over).detach(),
    label=f'Degree 12 (Weak baseline: overfit, MSE={mse_over:.4f})',
    linestyle=':',
    color='tab:red',
    alpha=0.6
)

plt.xlabel('Normalized Temperature')
plt.ylabel('Yield Strength')
plt.title('Polynomial Fitting: Underfitting vs Overfitting')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

# %% [markdown]
# **Discussion:**
# - **Degree 1 (Underfitting)**: The model is too simple to capture the cubic trend, so it misses important curvature.
# - **Degree 3 (Best / true degree)**: This model matches the data-generating process and should generalize best.
# - **Degree 8 (Overfitting)**: The curve can wiggle to fit noise, which may reduce training error but harms extrapolation/generalization.



# %% [markdown]
# # Assignments / Student Exercises
# The following exercise tests your understanding of the core concepts taught so far.
# 
# ## Exercise 1: Advanced Indexing and Slicing (Refers to Part 1)
# Create a smiley face of ones (`1.0`) in a 256x256 image of zeros (`0.0`). 
# You must use **advanced array indexing and boolean masks**—do not use slow `for` loops to modify individual pixels.
# 
# **Steps:**
# 1. Create a 256x256 image of zeros.
# 2. Create a circular arc of ones with inner radius 60 and outer radius 70,
#    going from -10 deg to -170 deg.
# 3. Create eyes as two circles with radius 10, centered at (x,y)=(100, 100) and (x,y)=(160, 100).
# 4. Plot the image.

# ![Smiley Face Output](smiley.png)

#%%
# Step 1: Create a 256x256 image of zeros
image = np.zeros((256, 256), dtype=np.float32)

# Build row/column coordinate grids for vectorized masking
y, x = np.ogrid[:256, :256]

# Step 2: Draw mouth arc (annulus sector)
cx_mouth, cy_mouth = 128, 128
dx_mouth = x - cx_mouth
dy_mouth = y - cy_mouth

radius_mouth = 0 # write the expression, such that radius_mouth is in [60, 70]
angle_mouth = 0 # write the expression, such that angle_mouth is in [-180, 180]

plt.figure(figsize=(6, 5))
plt.imshow(angle_mouth, cmap='twilight', origin='lower')
plt.colorbar(label='Angle [degrees]')
plt.title('angle_mouth values')
plt.axis('off')
plt.show()


# Left half-plane corresponds to angles 90..180 and -180..-90 (90..270 in [0,360) terms)
mouth_mask = 0 # write the expression, such that mouth_mask is True for the mouth arc
image[mouth_mask] = 1.0

# Step 3: Draw two filled circular eyes
r_eye = 10
left_eye_mask = 0 # write the expression, such that left_eye_mask is True for the left eye
right_eye_mask = 0 # write the expression, such that right_eye_mask is True for the right eye
image[left_eye_mask] = 1.0
image[right_eye_mask] = 1.0

# Step 4: Plot the smiley image
plt.figure(figsize=(6, 6))
plt.imshow(image, cmap='gray', vmin=0, vmax=1)
plt.title('Smiley from Advanced Indexing/Slicing')
plt.axis('off')
plt.savefig("smiley.png", bbox_inches='tight')
plt.show()
# Save the image as a binary .npy file for comparison
 

# Example assert for students: their_image should be equal to the ground truth binary file
# Uncomment and use this line after students generate their own image array named `their_image`
gt = np.load("smiley_gt.npy")
assert np.allclose(image, gt), "Your output does not match the expected smiley image."
