# MATLAB Image-Analytics Toolkit 📊🖼️

A personal MATLAB project that explored core computer-vision and machine-learning techniques:  
edge detection and filtering, K-Nearest Neighbour (KNN) classification (including a manual KNN implementation), and super-pixel clustering for image segmentation.

---

## 🧠 Key Modules

| Module | What It Did | Highlights |
|--------|-------------|------------|
| **Edge & Texture Analysis** | Applied custom convolution kernels and a Laplacian-of-Gaussian (LoG) filter to detect edges in grayscale images. | - 3×3 high-pass kernel for edge maps<br>- LoG kernel generation & 3-image comparison |
| **K-Nearest Neighbour Classification** | Built and evaluated KNN models on the Iris dataset, then re-implemented KNN from scratch for deeper insight. | - Used MATLAB’s `fitcknn` + resub / hold-out testing<br>- Crafted L2-distance loop and confusion matrices manually |
| **Super-Pixel Clustering & Segmentation** | Leveraged `superpixels` to over-segment an image, averaged RGB values per region, and separated foreground / background segments. | - Displayed boundary masks & region contours<br>- Saved split objects (`DogSplit1.png`, `DogSplit2.png`) |

---

## 📂 Repository Structure

```text
├── src/
│   ├── image_convolution.m        % Edge detection & LoG demo
│   ├── knn_classification.m       % Built-in and manual KNN
│   └── superpixel_clustering.m    % Super-pixel segmentation
├── assets/                        % Input images & results
└── README.md
