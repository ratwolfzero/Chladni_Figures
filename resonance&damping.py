import numpy as np
import matplotlib.pyplot as plt

# Set up parameters
f0 = 10.0  # Resonant frequency
f = np.linspace(0, 20, 1000)  # Frequency range

# Define damping factors (gamma)
gammas = [0.5, 1.0, 2.0]  # Small, medium, large gamma
colors = ['blue', 'green', 'red'] # Colors for the lines
labels = ['Small γ', 'Medium γ', 'Large γ']

plt.figure(figsize=(10, 6))

for gamma, color, label in zip(gammas, colors, labels):
    A = 1 / ((f - f0)**2 + gamma**2)
    plt.plot(f, A, color=color, linewidth=2, label=label)

# Add a dummy plot for FWHM in the legend
plt.plot([], [], color='green', linewidth=2.5, label='FWHM (Full Width Half Maximum)')

# Add labels and title
# Add labels and title
plt.title('Resonance Curve: Effect of Damping Factor γ', fontsize=14)
plt.xlabel('Frequency (f)')
plt.ylabel('Amplitude (A)')

plt.axvline(x=f0, color='k', linestyle='--', alpha=0.5, label=f'Resonance Frequency (f₀ = {f0})')
plt.grid(True, alpha=0.3)
plt.legend()

# Highlight the resonance width for the MEDIUM curve (γ = 1.0)
gamma_example = 1.0
A_max = 1 / (gamma_example**2)
fwhm = 2 * gamma_example  # Full Width at Half Maximum

plt.annotate('', xy=(f0 - gamma_example, A_max/2), xytext=(f0 + gamma_example, A_max/2),
             arrowprops=dict(arrowstyle='<->', color='green', lw=2.5))
plt.text(f0, A_max/2 + 0.01, 'FWHM', ha='center', fontweight='bold', color='green')

plt.ylim(0, 1.1)
plt.tight_layout()
plt.subplots_adjust(bottom=0.15)  # add extra space for footnote

plt.figtext(0.5, 0.02, 
            "Note: γ values exaggerated compared to physical guidelines for clarity", 
            wrap=True, ha='center', fontsize=9, color='black')

plt.savefig('Resonance_and_Damping_Factors_Graph.png', dpi=120, bbox_inches='tight')
plt.show()

