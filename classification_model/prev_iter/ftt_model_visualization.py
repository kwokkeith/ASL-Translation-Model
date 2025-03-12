import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import argparse

parser = argparse.ArgumentParser(description="Visualize FFT features and labels")
parser.add_argument("--fft_features", type=str, required=True, help="Path to the FFT features .npy file")
parser.add_argument("--labels", type=str, required=True, help="Path to the labels .npy file")
parser.add_argument("--output_dir", type=str, default="fft_plots", help="Directory to save plots")
args = parser.parse_args()

# Load FFT features and labels
X = np.load(args.fft_features)  # Shape: (samples, 2) -> [mean_freq, dominant_freq]
y = np.load(args.labels)  # 0 = Dynamic, 1 = Static

# Split features
mean_freqs = X[:, 0]
dominant_freqs = X[:, 1]

# Define categories
labels = np.array(["Dynamic" if label == 0 else "Static" for label in y])

# Create output directory for saving plots
import os
output_dir = "fft_plots"
os.makedirs(output_dir, exist_ok=True)

#Save FFT Spectrum for a Single Sample
def save_fft_spectrum(sample_index):
    """Saves the FFT spectrum of a single gesture sequence."""
    sample_freqs = np.linspace(0, 30, len(X[sample_index]))  # Generate frequency bins
    plt.figure(figsize=(8, 4))
    plt.plot(sample_freqs, X[sample_index], label=f"Sample {sample_index} FFT Spectrum")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Amplitude")
    plt.title("FFT Spectrum of a Gesture")
    plt.legend()
    
    # Save the figure
    file_path = os.path.join(output_dir, f"fft_spectrum_sample_{sample_index}.png")
    plt.savefig(file_path)
    plt.close()  # Close the figure to free memory
    print(f"Saved: {file_path}")

#Save FFT Spectrum Comparison of Dynamic vs. Static Gestures
def save_fft_comparison():
    """Saves the frequency spectrum comparison of static vs. dynamic gestures."""
    dynamic_samples = X[y == 0]
    static_samples = X[y == 1]

    plt.figure(figsize=(10, 5))
    sns.kdeplot(dynamic_samples[:, 0], label="Dynamic Gestures", fill=True)
    sns.kdeplot(static_samples[:, 0], label="Static Gestures", fill=True)
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Density")
    plt.title("Comparison of FFT Spectrum - Dynamic vs. Static Gestures")
    plt.legend()
    
    # Save the figure
    file_path = os.path.join(output_dir, "fft_comparison.png")
    plt.savefig(file_path)
    plt.close()
    print(f"📁 Saved: {file_path}")

# Save scatter plot: mean vs. dominant Frequency
def save_scatter_plot():
    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(mean_freqs, dominant_freqs, c=y, cmap="coolwarm", alpha=0.7)
    plt.colorbar(scatter, label="Class (0=Dynamic, 1=Static)")
    plt.xlabel("Mean Frequency (Hz)")
    plt.ylabel("Dominant Frequency (Hz)")
    plt.title("Mean vs. Dominant Frequency")
    plt.grid()

    # Save the figure
    file_path = os.path.join(output_dir, "scatter_plot.png")
    plt.savefig(file_path)
    plt.close()
    print(f"Saved: {file_path}")

# Run and Save the Visualizations
save_fft_comparison()  # Save FFT spectrum comparison
save_scatter_plot()  # Save scatter plot of Mean vs. Dominant frequency
# save_fft_spectrum(0)  # Save FFT spectrum of sample index 0
