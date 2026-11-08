# Mathematical Foundations of SONAPSE-EVO

## 1. Baseline Recurrent Rate Dynamics

Let the neural system contain $N$ neurons with continuous activation state $\mathbf{r}_t \in [0, 1]^N$.
The baseline continuous-time abstracted update over each microstep is:

$$\mathbf{r}_{t+1} = \operatorname{clip}_{[0, 1]} \left[ (1 - \eta)\mathbf{r}_t + \eta \phi\left( (W_b + W_p)\mathbf{r}_t + B\mathbf{o}_t + \mathbf{b} \right) \right]$$

where:
- $\eta \in (0, 1]$ is the leak coefficient (default $\eta = 0.35$).
- $\phi(x) = \max(0, \tanh(x))$ is a 1-Lipschitz activation function.
- $W_b$ is the base connectome connectivity.
- $W_p$ is plastic connectivity.
- $B$ is the sensory projection matrix.
- $\mathbf{b}$ is intrinsic excitability bias.

## 2. Lesion Representation

Controlled topological lesions are represented via binary mask $M \in \{0, 1\}^E$:

$$W_b^{(\ell)} = M \odot W_b$$

Lesion families include uniform random edge deletion, inter-region bridge ablation, sensory pathway disconnection, recurrent feedback cycle disruption, and degree-based hub ablation.

## 3. Graft Auxiliary Current Injection

With $K$ installed neural grafts, the substrate dynamics become:

$$\mathbf{r}_{t+1} = \operatorname{clip}_{[0, 1]} \left[ (1 - \eta)\mathbf{r}_t + \eta \phi\left( (W_b^{(\ell)} + W_p)\mathbf{r}_t + B\mathbf{o}_t + \sum_{g=1}^K \mathbf{I}_g(t) + \mathbf{b} \right) \right]$$

subject to the global current boundary:

$$\left\| \sum_{g=1}^K \mathbf{I}_g(t) \right\|_\infty \le B_{\max} \quad (B_{\max} = 0.10)$$

## 4. Function-Preserving Zero-Gain Ramp

A newly installed graft must not perturb functional computation upon commitment.
Setting:

$$\alpha_g(0) = 0 \implies \mathbf{I}_g(0) = \mathbf{0}$$

guarantees exact numerical preservation of baseline dynamics.
Activation follows a smooth linear ramp over $T_{\mathrm{ramp}}$ transitions:

$$\alpha_g(t) = \alpha_{\max} \min\left(1, \frac{t - t_0}{T_{\mathrm{ramp}}}\right)$$

## 5. One-Step and Multi-Step Stability Bounds

Since $\phi$ is 1-Lipschitz, for identical initial states and inputs, the immediate one-microstep rate deviation is bounded by:

$$\|\Delta \mathbf{r}_{t+1}\|_\infty \le \eta B_{\max}$$

For $\eta = 0.35$ and $B_{\max} = 0.10$:

$$\|\Delta \mathbf{r}_{t+1}\|_\infty \le 0.035$$

For multi-step stability, the system continuously tracks the local Jacobian operator norm:

$$J_t = \frac{\partial \mathbf{r}_{t+1}}{\partial \mathbf{r}_t} = (1 - \eta)I + \eta \operatorname{diag}(\phi'(u_t)) (W_b^{(\ell)} + W_p)$$

with contraction condition $\|J_t\|_2 < 1$.

## 6. Recovery Quantification

Given intact return $R_I$, damaged return $R_D$, and repaired return $R_R$, the normalised recovery fraction is:

$$\mathcal{R} = \frac{R_R - R_D}{R_I - R_D}$$

Catastrophic forgetting on historical task distributions $\mathcal{D}_k$ is measured by:

$$F = \frac{1}{K}\sum_{k=1}^K [R_k^{\mathrm{before}} - R_k^{\mathrm{after}}]_+$$
