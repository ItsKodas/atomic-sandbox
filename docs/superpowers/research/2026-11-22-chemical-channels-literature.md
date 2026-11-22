# Chemical channels for the linear Bayesian ACE surrogate: literature notes

Date: 2026-11-22. Question: we want to replace the surrogate's per-element-combination column blocks (Q' = 6 656 at 4 elements, 14 670 at 5; refit about Q'^3, 31 min at 5 elements) with K shared "chemical channels". Each element gets a learnt mix of the channels, learnt from our own DFT samples only. The final fit stays linear with an exact Gaussian posterior on top of a frozen embedding.

Labels used below: **[S]** means a source states it (the citation follows). **[I]** means it is my inference or derivation and no source states it.

---

## 1. Tensor-reduced ACE: how the element embedding is defined and learnt

**Reference.** J. P. Darby, D. P. Kovács, I. Batatia, M. A. Caro, G. L. W. Hart, C. Ortner, G. Csányi, "Tensor-reduced atomic density representations", *Phys. Rev. Lett.* 131, 028001 (2023), open access. arXiv:2210.01705 (v1 Oct 2022, v2 Dec 2022, v3 Oct 2026). Links: <https://arxiv.org/abs/2210.01705>, <https://doi.org/10.1103/PhysRevLett.131.028001>, HTML at <https://ar5iv.labs.arxiv.org/html/2210.01705>.

**Definitions.** The paper defines two reductions.

- **Embedding form (Eq. 6, Table I, "Embedding").** Ā_{i,klm} = Σ_{z,n} W^k_{zn} A_{i,znlm}, for k = 1…K. Full tensor products are then taken over the K channels (Eq. 7). The basis size is O(K^ν) [S, Table I]. The standard ACE basis is O((NS)^ν), where N is the number of elements and S the number of radial functions [S, Table I].
- **Tensor-decomposition form (Eqs. 8–12).** The coefficients are given a symmetric low-rank (CP-type) structure, c_{znlη} = Σ_k λ_{kη} ∏_t W^k_{z_t n_t l_t} (Eq. 9). The resulting features are Ã_{i,k l m} = Σ_{zn} W^{kl}_{zn} A_{i,znlm} (Eq. 10). Products are taken *element-wise in k*, so every factor uses the same channel k, which gives B̃_{i,klη} (Eq. 12). The basis size is **O(K)**, linear in K [S, Table I].
- **Tensor sketch (Eqs. 13–15).** This variant uses independent random matrices per factor. It is equivalent to the decomposition form when all factors share the same weights [S].
- The paper notes that K = 2 with element-only reduction reproduces the older element-weighting schemes [S, after Eq. 12].

**How W is obtained.**

- **Linear ACE (QMugs organics, Fig. 2):** W is **random and fixed**. The paper says it uses "random weights as a simpler alternative". The error bars in Fig. 2 come from 5 fits with independently drawn random weights [S]. The paper gives no normalisation or optimisation of W for linear ACE [S, by omission]. The fit is "linear least squares regression with Laplacian preconditioning regularization", a 4.5 Å cutoff and up to 4-body terms [S, SI "Linear ACE models"].
- **MACE (HME21, Table II):** W is factorised as W^{kl}_{zn} = U_{zn} V^{kl}_n. U (element) and V (radial) are either optimised by back-propagation or frozen at random values [S].
- The paper also says that "embedding weights are optimised either before or during fitting", and that optimising during fitting makes the model non-linear [S]. It does not test learning W first and freezing it before a linear fit [S, by omission].

**Element-only vs element-radial mixing in *linear* ACE.** SI, "Linear ACE models", near SI Fig. 4. At about 20 000 basis functions [S]:

| Reduction | Force error |
|---|---|
| Element-only | 0.2 eV/Å |
| Element-radial | 0.5 eV/Å |
| Fully coupled (full) basis | 0.3 eV/Å |

The authors hypothesise that "the radial decoupling only works in the case of non-linear models" [S]. So, at equal basis size, mixing elements only *beats* the full basis in a linear model, while mixing radial functions as well hurts [S]. The SI does not say clearly which dataset SI Fig. 4 uses (most likely QMugs) [I].

**Learnt-then-frozen vs joint learning: what the evidence says.**

- **HME21 with MACE (Table II, MAE):** 37 elements, energy in meV, force in eV/Å. The column mapping is inferred from a garbled HTML rendering, so check it against the PDF.

  | Model | U (element) | V (radial) | Energy MAE | Force MAE |
  |---|---|---|---|---|
  | MACE(1) | optimised | optimised | 53.5 | 0.189 |
  | MACE(1) | random | optimised | 52.1 | 0.175 |
  | MACE(1) | random | random | 254.8 | 0.456 |
  | MACE(2) | — | — | 15.7 | 0.138 |

  Random *element* weights cost almost nothing ("almost no degradation") **when K is large (256 channels) and the radial weights stay learnt** [S].
- **Linear model with 25 transition metals:** Lopanitsyna et al. (below) found the opposite at small K. With n_alch = 4, a fixed random or "physically initialised" coupling raised the saturation error by about **an order of magnitude** compared with an optimised coupling [S, Sec. IV.A, Fig. 2].
- **Reconciliation [I].** Random embeddings are fine when K is large enough to span the element space, and when K ≥ N they lose nothing. A *small* K (much less than N) has to be learnt. With 5–10 elements, compression only pays off at K < N, so learning W matters for us.

**Exactness observation [I, derivation].** In the embedding form, if K = N and W is invertible, then the ν-fold symmetric tensor power of W is invertible. The channel basis then spans *exactly* the same space as the full element basis. It is a reparametrisation (although the Gaussian prior is not invariant under it). For K < N, the embedding form is the full model with a **Tucker-structured coefficient tensor** that shares one factor W across all species modes. The tensor-decomposition form is the **CP-structured** analogue. This gives a linear-algebra route to W (section 4).

---

## 2. Their numbers

**Darby et al. (PRL 2023):**

- **Fig. 1:** Rank test on 5-element fcc lattices (up to 6 atoms per cell, Gaussian displacements of 0.025 Å and 0.25 Å). The rank of the tensor-reduced basis "grows almost linearly", while the standard ACE basis is redundant [S]. The text gives no numerical ranks.
- **Fig. 2, organics:** QMugs, **10 elements (H, C, N, O, F, P, S, Cl, Br, I)**. 400 training molecules (19–168 atoms, GFN2-xTB labels, 800 K NVT for 1 ps) and 1 000 test molecules. Linear ACE up to 4-body. Increasing K "converge[s] the accuracy to the previous level, whilst reducing the size of the model by a factor of 10" [S]. The text gives no absolute RMSE values; they are only on the plot [S, by omission]. Note that the labels are xTB, not DFT.
- **Fig. 3, high-entropy alloy:** GAP/SOAP on a quinary HEA (S = 5). Element-radial reduced SOAP reaches similar force accuracy with about 10× fewer features. The tensor sketch matches a full random projection for descriptor length above 300. Element-only mixing "did not help" in this SOAP/GAP case [S].
- **Table II, HME21:** 37 elements, MACE (nonlinear). Numbers in section 1.
- **SI Fig. 4:** The linear force errors at about 20 000 basis functions given in section 1.
- **Headline claim:** "10-fold reduction" in features for organics and HEAs [S, abstract and conclusion].

**Molecular / organic chemistry specifically:** QMugs (C, H, N, O, F, P, S, Cl, Br, I) is the only organic test. The revised-MD17 carbon environments in the SI are used only for information-imbalance analysis [S].

---

## 3. Uncertainty with embeddings or compression

**Direct literature: none found.** I found no paper that studies Bayesian or GP posterior variance, committees or calibration *together with* element embeddings or chemical compression, and none on the behaviour for poorly sampled elements. Darby et al. give only fit-to-fit spread over random W, not a predictive uncertainty [S]. Lopanitsyna et al. report no uncertainty quantification [S]. This is a genuine gap [I].

**Adjacent methods that transfer:**

- **Linearised extrapolation over all parameters, including species-dependent ones.** MTP, multicomponent version. Novikov, Gubaev, Podryabinkin, Shapeev, "The MLIP package: moment tensor potentials with MPI and active learning", *Mach. Learn.: Sci. Technol.* 2, 025002 (2021), arXiv:2007.08555, Sec. 2.3: "we hence linearize the energies with respect to the parameters". The active-set matrix is the Jacobian over θ = {ξ, c}, which includes the species-pair radial coefficients c [S]. This is a production precedent for counting embedding-like parameters in the uncertainty through tangent features. MLIP itself is not adopted; only the idea is used.
- **Last-layer Laplace / prediction rigidity (LLPR).** Bigi, Chong, Ceriotti, Grasselli, "A prediction rigidity formalism for low-cost uncertainties in trained neural networks", arXiv:2403.02251 (2024; published in *Mach. Learn.: Sci. Technol.*, verify the volume). Eq. 25 gives σ² = α² f*ᵀ(FᵀF + ς²I)⁻¹f*, which is exactly the Bayesian linear-regression posterior on frozen features. α² *must be calibrated* [S, Sec. 3.5]. The method explicitly assumes "all layers before the final linear layer do not contribute to the uncertainty" [S, Sec. 3.5]. The paper does not analyse the underestimation this causes [S, by omission]. Our frozen-embedding plan is LLPR, with our channel features as the "last layer" [I].
- **Linearised (GGN) Laplace for all parameters.** Immer, Korzepa, Bauer, "Improving predictions of Bayesian neural nets via local linearization", AISTATS 2021, arXiv:2008.08400. Daxberger et al., "Laplace Redux", NeurIPS 2021, arXiv:2106.14806. Linearising in *all* parameters turns the model into a Bayesian linear model in the tangent features ∂E/∂θ [S]. For us this means adding the N_el·K columns ∂E/∂W to the covariance computation (section 5) [I].
- **Misspecification.** Swinburne & Perez, "Parameter uncertainties for imperfect surrogate models in the low-noise regime", *Mach. Learn.: Sci. Technol.* 6 (2025), arXiv:2402.01810. Bayesian-regression parameter uncertainties "are thus significantly underestimated and vanish in the large data limit" when the model is misspecified, which is the normal case for near-deterministic DFT labels [S]. They propose POPS (pointwise optimal parameter sets) for linear models. Applications: Perez, Subramanyam, Maliyov, Swinburne, "Uncertainty quantification for misspecified machine learned interatomic potentials", arXiv:2502.07104 (2025). They apply POPS to qSNAP and to a linear corrector on MACE-MPA-0 descriptors, with about 1–3 % envelope violation [S]. Code: <https://github.com/tomswinburne/POPSRegression> (licence not checked; read before any reuse). A low-K frozen embedding *increases* misspecification, so this effect gets worse [I].
- **Committees from the Bayesian-regression posterior for linear ACE.** van der Oord, Sachs, Kovács, Ortner, Csányi, "Hyperactive learning for data-driven interatomic potentials", *npj Comput. Mater.* 9, 168 (2023), arXiv:2210.04225 [S]. This is the same exact-posterior setting as ours, without compression.

**Poorly sampled elements [I].** With a frozen W, an element seen in few samples has a poorly determined W row. Its features are then linear combinations of channels that the *other* elements have filled. The exact posterior over c can therefore look confident for a configuration whose error is dominated by a wrong embedding. Worse, if W comes from shrunk coefficients, a rare element's row tends towards 0, its features vanish, and the predicted variance vanishes with them. Two things counter this:

1. Tangent columns ∂E/∂W in the covariance.
2. A hard per-element sample-count guard that sends the configuration to DFT.

No source tests this; it should be checked with a leave-element-mostly-out experiment.

---

## 4. Alternatives doing the same job

| Approach | Embedding learnt how | Final model linear? | Notes for us |
|---|---|---|---|
| **Willatt, Musil, Ceriotti**, "Feature optimization for atomistic machine learning yields a data-driven construction of the periodic table of the elements", *PCCP* 20, 29661 (2018), arXiv:1807.00236, <https://ar5iv.labs.arxiv.org/html/1807.00236> | κ_{αα'} = Σ_J u_{αJ}u_{α'J} with d_J pseudo-species (Methods, Eq. 15). u is optimised by "an iterative scheme working in the primal formulation of ridge regression", described as non-linear optimisation of d_J × N weights (Results; SI) [S]. Initialised from Pauling electronegativity and vdW radius (Eq. 20) [S]. | Kernel ridge regression on the compressed SOAP | Elpasolites, **39 elements**: d_J = 1 is poor; d_J = 2 beats SOAP below 2k structures, then saturates; d_J = 4 (99 % fewer features) beats standard SOAP but still saturates; little gain above 4 (Fig. 1) [S]. **The tabulated-property initialisation is forbidden by our rule 1**, so we would have to initialise from data or at random. |
| **Lopanitsyna, Fraux, Springer, De, Ceriotti**, "Modeling high-entropy transition-metal alloys with alchemical compression", *Phys. Rev. Mater.* 7, 045802 (2023), arXiv:2212.13254 | Coupling u (Eq. 3) optimised *jointly* with the model weights by L-BFGS through PyTorch autodiff (Sec. III.C, IV.A) [S] | The learning curves use linear models on the compressed features; the production model adds a NN term (HEA25-4-NN) [S] | **25 elements**: n_alch = 3–4 suffice for 3-body; 5 is no better than 4; random or physical u gives about 10× higher saturation error (Fig. 2) [S]. Production model hold-out errors: 14 meV/atom energy RMSE, 280 meV/Å force RMSE [S]. |
| **Darby, Kermode, Csányi**, "Compressing local atomic neighbourhood descriptors", *npj Comput. Mater.* 8, 166 (2022), arXiv:2112.13055 | No learning: a deterministic compression of the power spectrum from O(N²S²) to O(NS), described as lossless in the element and radial channels, plus an element-agnostic total-density variant [S] | Yes (descriptor only) | Still grows with N (linearly) and is power-spectrum (3-body) oriented. A cheap, rule-safe baseline, but it does not stop growth with N [I]. |
| **MTP (multicomponent)**: Gubaev, Podryabinkin, Hart, Shapeev, *Comput. Mater. Sci.* 156, 148 (2019); MLIP-2 paper above | Species enter only through the radial coefficients c^{(β)}_{μ,z_i,z_j} (Eq. 3), fitted non-linearly together with the linear ξ [S] | No (non-linear in c) | Basis size is set by lev_max and N_Q, not by the species count [S, Sec. 2.1]. |
| **MTP low-rank**: Vorotnikov, Romashov, Rybin, Rakhuba, Novikov, "Low-rank matrix and tensor approximations for compression of MLIPs", *J. Chem. Phys.* 163, 244112 (2025), arXiv:2509.04440 | Low-rank (matrix and tensor-train) factorisation of the species-pair radial tensor, trained under a rank constraint (BFGS, Riemannian BFGS, rank-adaptive) [S] | No | Up to 50 % parameter compression with no accuracy loss. Glycine (Table 3): base 5.2 meV/atom and 109 meV/Å; TF-optimal 4.9 meV/atom and 101 meV/Å [S]. About 1.5× fewer radial parameters for ACE (Sec. II.6) [S]. Shows that species-pair coefficient tensors are low-rank [I]. |
| **MACE / NequIP / Allegro** (context): Batatia et al., NeurIPS 2022, arXiv:2206.07697; Batzner et al., *Nat. Commun.* 13, 2453 (2022); Musaelian et al., *Nat. Commun.* 14, 579 (2023) | One-hot species embedded by a learnt linear map, trained end-to-end [S, general architecture] | No | Not usable with an exact posterior except through a last-layer or linearised Laplace approximation (above). |
| **GRACE**: Bochkarev, Lysogorskiy, Drautz, *Phys. Rev. X* 14, 021036 (2024), arXiv:2311.16326 | Tensor decomposition of the ACE coefficients over species: c = Σ_k c^{(k)} w_{kκ1} w_{kκ2}… (Eq. 51); chemistry folded into the radial functions (Eq. 52); radial functions optimised during training [S, Sec. VI.2] | No | Same CP idea as Darby's decomposition form; no combinatorial explosion [S]. |
| **Embedded cluster expansion**: Müller & Natarajan, arXiv:2409.06071 (2024/25) | Low-dimensional embedding of the site basis learnt jointly by gradient descent [S] | No | Lattice cluster expansion, 6 elements; the embedding enables extrapolation to unseen compositions [S]. |
| **ACEpotentials.jl (ETACE backend)** | Species-pair-indexed linear radial weights W[(Z_i, Z_j)]; design matrix for linear least squares [S, ETACE tutorial] | Yes | I found no documented channel or embedding reduction (not checked exhaustively). |

**Best fit for a linear, exact-posterior model [I].** The candidates are:

- Darby's *element-only* reduction, which is supported for linear fits in the SI.
- A **data-learnt** W instead of a random one (Lopanitsyna's factor-of-10 result at small K).
- Obtaining W by **linear algebra** instead of joint non-linear fitting. One route is a truncated higher-order SVD (HOSVD) or reduced-rank regression on a fitted coefficient tensor, using the Tucker view from section 1: De Lathauwer, De Moor, Vandewalle, "A multilinear singular value decomposition", *SIAM J. Matrix Anal. Appl.* 21, 1253 (2000). Another is a few variable-projection / Gauss-Newton steps: Golub & Pereyra, *SIAM J. Numer. Anal.* 10, 413 (1973).

PCA of per-species density statistics is rule-safe and cheap, but it is unsupervised: it preserves descriptor variance, not energy relevance. Darby's information-imbalance analysis in the SI is the nearest example. Use it only as an initialiser [I].

---

## 5. Recommendation [I unless cited]

**Model structure:**

1. **Keep the 2-body block fully pair-resolved.** It costs N(N+1)/2 × n_radial columns, which is small and grows only as N². Bond lengths and the per-pair repulsion live here, and it gives every element its own honest posterior columns.
2. **Use channels for the many-body blocks only.** Mix elements only, Ā_{i,k nlm} = Σ_j W_{z_j k} R_nl(r_ij) Y_lm, and leave the radial basis unmixed: element-radial mixing hurt linear ACE (Darby SI). Two choices:
   - **Embedding form** (O(K^ν)): this is our existing combination machinery with "elements" replaced by K channels. Its Q' at K equals today's Q' at N = K, about 6.7k at K = 4 if the block sizes carry over and the centre species is embedded too. Keeping the centre species one-hot multiplies this by about N/K.
   - **Decomposition form** (O(K)): cheaper and coarser.

   Start with the embedding form at K = 3–4, plus the full pair block.
3. **Do not use random W at small K.** It is rule-safe, but Lopanitsyna et al. found about 10× worse saturation. Random W only works at K ≥ N or at large K.

**Obtaining W from our own DFT, linearly and deterministically:**

1. Fit, with the existing exact Bayesian regression, a *full* element-resolved model at reduced degree or body order. A pair-resolved 2-body plus a low-degree full 3-body block stays affordable up to about 6–8 elements.
2. Take the posterior-mean coefficient tensor of the many-body block. Unfold it along each neighbour-species mode, stack the unfoldings and take the SVD. The top-K right singular vectors give W (N × K, orthonormal columns). Weighting the coefficients by posterior precision (or using the evidence-optimal prior) reduces the influence of poorly determined elements. Fallback: reduced-rank regression on the pair block only.
3. Optionally run 1–3 variable-projection (Golub–Pereyra) Gauss-Newton steps on W, with the exact regression re-solved for c, at a "re-embed" event. Then **freeze W**. All ordinary refits keep W fixed and stay exactly linear.
4. Choose K by **Bayesian evidence** (log marginal likelihood), which the exact posterior already provides, over K ∈ {2, 3, 4, 6}.

**Suggested K for 5–10 light elements:** K = 3–4, possibly 5–6 at 10 elements.

- The literature saturates at 3–4 learnt channels for 25 transition metals and 39 elpasolite elements.
- Light main-group elements differ more in valence than neighbouring transition metals do, so expect to need the upper end, with the full pair block carrying the rest.
- K ≥ N gives no saving, so compression helps only once N exceeds about 4.

**Uncertainty (the main open point):**

1. **Tangent columns.** Add the N·K columns J_W = ∂E/∂W (and the matching force derivatives) to the covariance computation only. This is the linearised Laplace / MLIP-2 Sec. 2.3 idea. The cost is negligible next to Q'³. Without it the posterior ignores embedding uncertainty (the LLPR assumption) and will be overconfident for rare elements.
2. **Keep a hard per-element sample-count trust guard** that falls back to DFT for configurations containing an element below a minimum sample count.
3. **Calibrate.** Use a global α² scale on held-out samples, as LLPR does, or POPS-style inflation, because frozen low-K channels add misspecification (Swinburne & Perez: Bayesian-regression variances vanish with data under misspecification).
4. **Test with a leave-element-mostly-out experiment.** Train with an element at about 5 % of its samples and check that predicted σ tracks the actual error for that element.

**Main risks:**

1. **Accuracy ceiling at small K.** Every compression paper shows saturation; Willatt's d_J = 4 still saturates.
2. **Overconfident variance for poorly sampled elements.** A shrunk W row means vanishing features and vanishing variance. Mitigations: the pair block, the tangent columns and the guard.
3. **Re-embedding cost.** Many-body features are polynomial in W, so changing W means recomputing every sample's channel features from geometry, not just reweighting cached normal equations. Re-embed rarely, for example when an element joins the collection.
4. **Prior non-invariance.** W is identifiable only up to an invertible K × K transform, and the Gaussian prior on c is not invariant under it. Fix W to orthonormal columns and re-tune the prior by evidence.
5. **Rule 1.** W must come only from our own DFT fits, so no electronegativity or radius initialisation (Willatt's Eq. 20 is out). Random or SVD initialisation is fine.
6. **Evidence for organics comes from xTB labels and a small test.** Darby's QMugs labels are xTB, and the paper gives no K-vs-error numbers in the text. Our own K scan is the real gate.

**Licences.** These are ideas only; no code is adopted. POPSRegression's licence is unchecked. MLIP and GRACE (tensorpotential) were not checked and are not needed.
