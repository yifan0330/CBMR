import numpy as np
import matplotlib.pyplot as plt

intensity_func = np.load("intensity_voxel_space.npy")
middle_slice = intensity_func[:,:,48]

# Create the plot using imshow
plt.imshow(middle_slice, cmap='viridis')  # 'viridis' is just one colormap option, you can choose others like 'gray', 'jet', etc.

# Adding a color bar to show the color scale
plt.colorbar()

# Adding titles and labels (optional)
plt.title("middle slice")
plt.xlabel("X-axis")
plt.ylabel("Y-axis")

# Display the plot
plt.savefig("test.png")
