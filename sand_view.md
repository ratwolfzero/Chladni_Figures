# The Sand View (Experimental / Development Feature)

[Chladni](/images&Animation/Chladni_7.png)

* The **Sand View** provides a particle-based visualization that approximates how sand accumulates along nodal lines in physical Chladni experiments.

* Instead of displaying the displacement field directly, this mode generates a large number of simulated “grains” whose spatial distribution is derived from the steady-state amplitude field $Z(x,y;f)$.

* Regions of **low vibration amplitude** (near nodal lines) receive a **higher probability density**, while strongly vibrating regions are statistically suppressed. The result is a granular pattern that visually resembles experimental Chladni figures formed by migrating particles.

## Statistical Model

Let $Z(x,y;f)$ be the steady-state displacement field. The sand distribution is generated using a probability density of the form:

$$
\Large
p(x,y) \propto \exp!\left(-\frac{|Z(x,y;f)|}{Z_{\max} \cdot s}\right)
$$

where:

* $Z_{\max}$ is the maximum absolute displacement for normalization,
* $s$ is a scaling parameter controlling sharpness (`SAND_EXP_SCALE`).

This exponential weighting suppresses high-amplitude regions and favors near-zero displacement regions. After normalization, grain positions are sampled using a Monte Carlo method.

The simulation therefore does **not** compute particle trajectories or forces. Instead, it performs direct probabilistic sampling of a displacement-dependent distribution.

### Visual Interpretation

* Dark clusters represent areas where grains accumulate — i.e., near nodal lines.
* Regions with large vibration amplitude appear nearly empty.
* A small random spatial jitter is added to each sampled grain position to prevent visible grid artifacts and produce a more natural granular appearance.

Because the distribution depends on the full modal superposition, the sand pattern automatically reflects:

* Mode degeneracy
* Damping-induced blending
* Symmetry breaking at higher γ
* Interference between near-resonant modes

Thus, Sand View acts as a visually intuitive “nodal detector” derived from the same resonance-weighted field used in the magnitude and phase views.

#### Relation to Physical Experiments

In real Chladni experiments:

* Grains bounce dynamically due to plate acceleration.
* Migration occurs through complex frictional and collisional processes.
* Accumulation is influenced by time evolution and nonlinear interactions.

The Sand View abstracts these dynamics into a stationary probability model. It captures the **final accumulation pattern** without modeling transient motion.

This approach aligns conceptually with modern studies suggesting that grain redistribution can often be approximated as an effective diffusion process biased by vibration amplitude.

> The Sand View should therefore be understood as a **statistical visualization tool**, not a mechanical particle simulation.
