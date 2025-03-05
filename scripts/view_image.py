import matplotlib.pyplot as plt
import matplotlib.image as mpimg

# Load and display the image
img = mpimg.imread('figures/synthetic_methods_comparison.png')
plt.figure(figsize=(10, 8))
plt.imshow(img)
plt.axis('off')
plt.savefig('/tmp/synthetic_methods_comparison_view.png')
print("Image saved to /tmp/synthetic_methods_comparison_view.png")
