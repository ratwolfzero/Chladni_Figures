import numpy as np
import matplotlib.pyplot as plt

# Set up parameters
f0 = 10.0  # Resonant frequency
f = np.linspace(0, 20, 1000)  # Frequency range

# Define damping factors (gamma)
gammas = [0.5, 1.0, 2.0]  # Small, medium, large gamma
colors = ['blue', 'green', 'red'] # Colors for the lines
labels = ['Small γ (0.5)', 'Medium γ (1.0)', 'Large γ (2.0)']

# Create the plot
plt.figure(figsize=(10, 6))

for gamma, color, label in zip(gammas, colors, labels):
    # Calculate amplitude (Lorentzian function)
    A = 1 / ((f - f0)**2 + gamma**2)
    plt.plot(f, A, color=color, linewidth=2, label=label)

# Add labels and title
plt.title('Resonance Curve: Effect of Damping Factor γ', fontsize=14)
plt.xlabel('Frequency (f)')
plt.ylabel('Amplitude (A)')
plt.axvline(x=f0, color='k', linestyle='--', alpha=0.5, label=f'Resonance Frequency (f₀ = {f0})')
plt.grid(True, alpha=0.3)
plt.legend()

# Highlight the resonance width for one curve
gamma_example = 1.0
A_max = 1 / (gamma_example**2)
fwhm = 2 * gamma_example  # Full Width at Half Maximum
plt.annotate('', xy=(f0 - gamma_example, A_max/2), xytext=(f0 + gamma_example, A_max/2),
             arrowprops=dict(arrowstyle='<->', color='black', lw=1.5))
plt.text(f0, A_max/2 + 0.02, 'Resonance Width', ha='center', fontweight='bold')

plt.ylim(0, 1.1)
plt.tight_layout()
plt.savefig('Resonance_and_Damping_Factors_Graph.png', dpi=120, bbox_inches='tight')
plt.show()
