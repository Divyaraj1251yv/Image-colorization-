This project focuses on automated image colorization using Generative Adversarial Networks (GANs) and Convolutional Neural Networks (CNNs). 
The model takes grayscale images as input and generates realistic colorized versions by learning complex spatial features and color distributions.

📌 Features
✅ Converts black-and-white images into colored images
✅ Uses CNN for feature extraction and GAN for enhanced realism
✅ Trained on a dataset of diverse images for better generalization
✅ Implements PyTorch/Keras with TensorFlow backend
✅ Evaluation using PSNR and SSIM metrics

🛠️ Tech Stack
Deep Learning Framework: TensorFlow / PyTorch
Model Architecture: GAN (Generator & Discriminator), CNN-based Autoencoder
Libraries Used: OpenCV, NumPy, Matplotlib
Dataset: (Mention if you used a specific dataset, e.g., ImageNet, Places365)

🔍 How It Works
Preprocessing: Convert images to LAB color space (L-channel as input, AB channels as output).
CNN-based Generator: Extracts image features and predicts color channels.
Discriminator (GAN): Distinguishes real vs. fake colorized images to improve realism.
Loss Functions: Uses Mean Squared Error (MSE) + Adversarial Loss for better result

📌 Future Improvements
🔹 Train on a larger dataset for better generalization
🔹 Optimize inference time for real-time colorization
🔹 Deploy as a Flask web app for interactive use.
