# Surrogate ("imitator") renderers and black-box optimisation for photo → CK3 DNA

Method note: WebSearch worked, but every full-text fetch was blocked by the egress proxy (arxiv.org, ar5iv, openaccess.thecvf.com, papers.nips.cc, ojs.aaai.org, semanticscholar, liner.com, university sites). So every finding below comes from search-result abstracts and snippets, linked to the primary page (arXiv/CVF/PMLR/NeurIPS/OpenReview). I could not check numbers against the full papers. Anything that rests only on background knowledge is labelled as such and kept in Inferences or Gaps. "NEW" marks 2024–2026 work.

## 1. Which systems learn neural imitators/proxies of non-differentiable renderers, engines or procedural generators to invert parameters? What architectures and data sizes worked?

### Takeaway
The game-character line of work all comes from NetEase Fuxi and related labs: F2P (ICCV 2019), Fast & Robust F2P (AAAI 2020), PokerFace-GAN, MeInGame (AAAI 2021) and T2P (CVPR 2023). Together with SwiftAvatar (AAAI 2023), AgileAvatar (SIGGRAPH Asia 2022) and a Dec-2024 cross-domain regressor, it shows a pattern. Teams start with "imitator plus per-image optimisation" and quickly move to an amortised translator/regressor trained on many engine renders (about 170k pairs in Fast F2P), using identity embeddings, segmentation or domain-bridging. Outside faces, the same idea shows up as differentiable proxies for non-differentiable procedural nodes (SIGGRAPH 2022) and as neural renderers that replace stroke simulators (Learning to Paint). The newest procedural-inversion work (DI-PCG, CVPR 2025) drops the per-instance surrogate and trains a small conditional diffusion model to predict parameters directly.

### Cited Findings
**Game-character auto-creation (closest prior art)**
- F2P (Shi et al., ICCV 2019) builds an "Imitator" G(x) that mimics the game engine (facial parameters → rendered face) and a feature extractor F(y) that defines the similarity space. Because the engine is not differentiable, the imitator lets parameters be "optimized by gradient descent" in a neural-style-transfer framework. — [arXiv 1909.01064](https://arxiv.org/abs/1909.01064); [CVF](https://openaccess.thecvf.com/content_ICCV_2019/html/Shi_Face-to-Parameter_Translation_for_Game_Character_Auto-Creation_ICCV_2019_paper.html)
- F2P has two losses. A "discriminative loss" (identity) and a "facial content loss", which is pixel-wise error on features from a face semantic-segmentation model and constrains the shape and placement of face components. The method shipped in a game and "has been used by players over 1 million times." — [CVF ICCV 2019](https://openaccess.thecvf.com/content_ICCV_2019/html/Shi_Face-to-Parameter_Translation_for_Game_Character_Auto-Creation_ICCV_2019_paper.html); [arXiv](https://arxiv.org/abs/1909.01064)
- Fast & Robust F2P (AAAI 2020) has four networks: an imitator, a translator T that predicts facial parameters, a face-recognition network that encodes the input into embeddings, and a face-segmentation network for position-sensitive features. It uses "170,000 pairs of continuous facial parameters and rendered facial images" to train the imitator and pre-train the translator. A single forward pass maps face embeddings to parameters, about 1000× faster than iterative F2P. Training is self-supervised with a "recursive consistency" term (the rendered result's facial representation should match the input's) and needs no ground truth or user interaction. — [arXiv 2008.07132](https://arxiv.org/abs/2008.07132); [AAAI](https://ojs.aaai.org/index.php/AAAI/article/view/5537)
- PokerFace-GAN covers neutral-face game-character auto-creation, i.e. removing expression before parameter fitting (title-level only). — [arXiv 2008.07154](https://arxiv.org/abs/2008.07154)
- MeInGame (AAAI 2021, NetEase Fuxi) has three parts: a 3DMM→game-mesh shape-transfer algorithm, low-cost texture acquisition, and a pipeline for training game-face reconstruction networks. Code and data are public. — [arXiv 2102.02371](https://arxiv.org/abs/2102.02371); [GitHub FuxiCV/MeInGame](https://github.com/FuxiCV/MeInGame)
- T2P (CVPR 2023) works on a bone-driven face model with continuous parameters (bone positions) and discrete ones (hairstyles). It uses CLIP plus neural rendering to search both in "a unified framework" and claims to be the first to optimise discrete and continuous parameters together. — [arXiv 2303.01311](https://arxiv.org/abs/2303.01311); [CVF](https://openaccess.thecvf.com/content/CVPR2023/papers/Zhao_Zero-Shot_Text-to-Parameter_Translation_for_Game_Character_Auto-Creation_CVPR_2023_paper.pdf)
- AgileAvatar uses "cascaded domain bridging" with mixed continuous/discrete avatar parameters. (1) It stylises the selfie into an avatar-like portrait and normalises expression. (2) It fits parameters to that stylised target through a differentiable imitator. (3) A "cascaded relaxation-and-search" pipeline finds a valid discrete avatar vector the engine can render. — [arXiv 2211.07818](https://arxiv.org/abs/2211.07818); [SIGGRAPH history listing](https://history.siggraph.org/?p=215758)
- SwiftAvatar (AAAI 2023) uses dual-domain generators with shared latent codes to produce paired (realistic face, avatar render) images. GAN inversion of engine renders links the latents to avatar vectors, which yields synthetic (avatar vector, realistic face) pairs. Semantic augmentation adds diversity, and a lightweight estimator trained on the pairs is validated on two avatar engines. — [arXiv 2301.08153](https://arxiv.org/abs/2301.08153); [AAAI](https://ojs.aaai.org/index.php/AAAI/article/view/25753)
- NEW: "Unsupervised Cross-Domain Regression for Fine-grained 3D Game Character Reconstruction" (arXiv, Dec 2024) is an end-to-end regressor aimed at the real→game domain gap. It adds a contrastive loss for instance-wise disparity and an auxiliary 3D identity-aware extractor. — [arXiv 2412.10430](https://arxiv.org/abs/2412.10430)
- CK3's own portrait system stores ethnicity blends, facial features, age and body type in per-character DNA that children inherit (GDC talk on the DNA-based portrait system). — [GDC Vault](https://www.gdcvault.com/play/1027354)

**Procedural materials/content and other non-differentiable generators**
- "Node Graph Optimization Using Differentiable Proxies" (Hu, Guerrero, Hasan, Rushmeier, Deschaintre, SIGGRAPH 2022) replaces each non-differentiable material-graph node (e.g. brick-pattern generators driven by discrete parameters) with a differentiable neural proxy. This allows end-to-end gradient optimisation of the whole graph against a target photo, beyond earlier approaches limited to differentiable filter nodes. — [arXiv 2207.07684](https://arxiv.org/abs/2207.07684); [Yale](https://graphics.cs.yale.edu/publications/node-graph-optimization-using-differentiable-proxies)
- NEW: DI-PCG (CVPR 2025) treats procedural parameters as the denoising target of a lightweight diffusion transformer conditioned on an image. It covers six Infinigen generators with 48/19/12/14/9/15 parameters, has 7.6M network parameters, trained in about 30 GPU-hours, generalises to in-the-wild images, and also accepts sketches. — [arXiv 2412.15200](https://arxiv.org/abs/2412.15200); [CVF CVPR 2025](https://openaccess.thecvf.com/content/CVPR2025/html/Zhao_DI-PCG_Diffusion-based_Efficient_Inverse_Procedural_Content_Generation_for_High-quality_3D_CVPR_2025_paper.html)
- Learning to Paint (ICCV 2019) trains a differentiable neural renderer for strokes that runs efficiently on GPU and drives a model-based DDPG agent. It notes that earlier work interacting with non-differentiable painting simulators "failed to provide detailed feedback". — [arXiv 1903.04411](https://arxiv.org/abs/1903.04411)
- SPIRAL (ICML 2018) does the opposite: no surrogate. An RL agent emits programs for a black-box graphics engine, and a discriminator's output is the reward. This was the key to progress, though it needed a distributed RL setup. — [PMLR](https://proceedings.mlr.press/v80/ganin18a.html)
- L-GSO (NeurIPS 2020) iteratively fits deep generative surrogates of a stochastic, non-differentiable simulator in local neighbourhoods of parameter space and uses them for gradients. It beats BO, numerical optimisation and score-function estimators when the simulator's parameter dependence lies on a low-dimensional submanifold. — [arXiv 2002.04632](https://arxiv.org/abs/2002.04632); [NeurIPS](https://papers.nips.cc/paper/2020/hash/a878dbebc902328b41dbf02aa87abb58-Abstract.html)
- "Learning to Simulate" (ICLR 2019) tunes a non-differentiable simulator's parameters with policy gradient, using downstream-model validation accuracy as reward. This is a REINFORCE-on-black-box precedent. — [arXiv 1810.02513](https://arxiv.org/abs/1810.02513)
- NEW: SMIRK (CVPR 2024) replaces the differentiable rasteriser in self-supervised 3D-face fitting with a neural rendering module (geometry render plus sparsely sampled input pixels → face image). This "bridges the domain gap between the input and the synthesized output" and lets the model augment training with novel expressions. — [arXiv 2404.04104](https://arxiv.org/abs/2404.04104); [CVF](https://openaccess.thecvf.com/content/CVPR2024/papers/Retsinas_3D_Facial_Expressions_through_Analysis-by-Neural-Synthesis_CVPR_2024_paper.pdf)
- NEW (title only): "Learning Instance-Specific Parameters of Black-Box Models Using Differentiable Surrogates" (2024). — [arXiv 2407.17530](https://arxiv.org/abs/2407.17530)

### Inferences
- The project's imitator (337 params → 192², about 29k renders) is about 6× smaller in data than Fast F2P's 170k pairs. At about 1000 renders/min, 170k renders take about 3 h and 1M about 17 h. Data volume is cheap, so it is the first thing to scale (inference from cited numbers plus the stated render rate).
- The NetEase line shipped a translator (amortised) model, not per-image imitator optimisation. Per-image surrogate optimisation was mainly a stepping stone to supervision. That matches the project's goal of "DNA in seconds".
- For discrete genes (hair, beard, brows), the published options are: (a) relax-then-search (AgileAvatar); (b) joint continuous/discrete search with a neural renderer plus CLIP (T2P); (c) DI-PCG-style direct prediction. Since real renders are cheap, the simplest robust variant is to predict the top-k discrete options and score each one in the real renderer (see Q3).
- SMIRK's idea transfers directly: an imitator that sees some appearance information from the target cannot "hide" geometry errors in colour. Separately supervising geometry (landmarks/segmentation of renders) from texture reduces the shortcuts available to the optimiser.

### Gaps
- I could not get exact imitator architectures, resolutions or dataset sizes for F2P (ICCV 2019) because full text was blocked. Background knowledge suggests a DCGAN-like generator trained on tens of thousands of random renders, but this is unverified.
- No published work found on CK3 specifically, or on any Paradox/Clausewitz-engine photo→DNA inversion. The search for "Crusader Kings 3 DNA from photo AI" returned only forum threads and the GDC talk.
- No paper reports blind human-judge outcomes for imitator-optimised vs real-render-verified results. The project's observation (recognizer gains, no human-visible gain) appears not to be addressed directly in this literature.

## 2. How do people prevent the optimiser from exploiting surrogate errors? Which methods are cheapest and most effective?

### Takeaway
The offline model-based optimisation (MBO) literature calls this "objective hacking": optimisation drifts off the data manifold where the learned model overestimates the score. The standard fixes, from cheap to more involved:
- keep the search near the data (boundary/prior losses, trust regions, limited ascent steps, multiple restarts)
- train the surrogate to be conservative or smooth off-manifold (COMs, RoMA)
- penalise ensemble disagreement (MOPO-style)
- the strongest fix: verify with the real simulator and aggregate data (DAgger-like)

The project faces a second, separate exploit. Optimising against an ensemble of face-recognition nets creates features that transfer to held-out recognizers (an ensemble-transfer adversarial effect). So holdout-recognizer gains are not evidence of human-visible likeness.

### Cited Findings
- NEW: The comprehensive offline-MBO review (Kim, Gu, Yuan, Yun, Liu, Bengio, Chen; arXiv 2503.17286, TMLR 2026) says extrapolating beyond the offline data "suffers from significant epistemic uncertainty, which may trigger objective hacking wherein the model exploits inaccuracies". It also surveys generative-model-guided approaches (VAE/GAN/autoregressive/diffusion). — [arXiv 2503.17286](https://arxiv.org/abs/2503.17286); [TMLR listing](https://mlanthology.org/tmlr/2026/kim2026tmlr-offline)
- COMs (Trabucco et al., ICML 2021) train one model with "an augmented regression objective that penalizes overestimation of the performance on off-manifold designs", outperforming the best prior method by 1.3–2× on some tasks. — [PMLR](http://proceedings.mlr.press/v139/trabucco21a/trabucco21a.pdf)
- Design-Bench (ICML 2022) is the standard benchmark suite for offline MBO, including high-dimensional continuous tasks such as robotics controllers. — [PMLR](https://proceedings.mlr.press/v162/trabucco22a/trabucco22a.pdf)
- RoMA (NeurIPS 2021) calls the core problem "adversarially optimized inputs… where the DNN highly overestimates the true objective". It counters this with robust pre-training plus adaptation of the proxy around the current candidates, using a local smoothness prior, and is best on 4 of 6 tasks. — [arXiv 2110.14188](https://arxiv.org/abs/2110.14188); [NeurIPS](https://papers.nips.cc/paper_files/paper/2021/file/24b43fb034a10d78bec71274033b4096-Paper.pdf)
- Neural-Adjoint (Ren et al., NeurIPS 2020) is essentially the project's current method: train a forward surrogate and gradient-descend on its inputs from many random starts. Adding a simple "boundary loss" that keeps inputs inside the training-data range "dramatically improves" results. On the Tandem model (a forward surrogate plus an inverse net, structurally like F2P's translator) the boundary loss cut error "by at least 2.5 orders of magnitude". — [arXiv 2009.12919](https://arxiv.org/abs/2009.12919); [NeurIPS](https://proceedings.nips.cc/paper_files/paper/2020/file/007ff380ee5ac49ffc34442f5c2a2b86-Paper.pdf)
- MOPO (NeurIPS 2020) optimises against reward penalised by estimated model error. Ensemble-disagreement penalties (e.g. max deviation of member means from the ensemble mean) and learned-variance penalties both beat no penalty "significantly". — [arXiv 2005.13239](https://arxiv.org/abs/2005.13239)
- NEW (titles/abstracts only): newer offline-MBO directions include learning-to-rank surrogates (ICLR 2025), robust guided diffusion for offline black-box optimisation (2024) and policy-guided gradient search (2024). — [ICLR 2025 PDF](https://proceedings.iclr.cc/paper_files/paper/2025/file/768c19273e20fa09147885d03da7550f-Paper-Conference.pdf); [arXiv 2410.00983](https://arxiv.org/abs/2410.00983); [arXiv 2405.05349](https://arxiv.org/abs/2405.05349)
- Ensemble attacks transfer: an image that is adversarial for several models "is more likely to transfer" to others, and ensemble-based optimisation made even targeted adversarial examples transfer to an unseen black-box system (Clarifai). — [Liu et al., ICLR 2017, arXiv 1611.02770](https://arxiv.org/abs/1611.02770)
- NEW: "Rethinking Model Ensemble in Transfer-based Adversarial Attacks" (ICLR 2024) links transferability to loss-landscape flatness and closeness to each member's optimum. Making ensemble optimisation smoother/flatter, as augmentation does, can increase transfer. — [ICLR 2024](https://proceedings.iclr.cc/paper_files/paper/2024/hash/53fe824f289060ce705ed7c01dae59d2-Abstract-Conference.html)
- Adversarially robust networks have "perceptually aligned gradients". Optimising inputs against standard nets gives "grainy" non-semantic results, while robust nets yield images that actually look like the target. — [Santurkar et al., NeurIPS 2019 text](https://object.cloud.sdsc.edu/v1/AUTH_da4962d3368042ac8337e2dfdd3e7bf3/ml-papers-txt/NeurIPS/2019/image_synthesis_with_a_single_robust_classifier__680ae8da.txt); [ACCV 2022, arXiv 2106.06927](https://arxiv.org/abs/2106.06927); [robust segmentation, arXiv 2204.01099](https://arxiv.org/abs/2204.01099)
- DreamSim (NeurIPS 2023) is a perceptual metric fitted to human similarity judgments. It captures layout, pose and semantic content better than LPIPS-style or large-vision-model baselines. — [NeurIPS 2023](https://proceedings.neurips.cc/paper_files/paper/2023/hash/9f09f316a3eaf59d9ced5ffaefe97e0f-Abstract.html)
- DAgger iteratively collects data under the current learned policy's own distribution, aggregates it, and retrains. It comes with no-regret guarantees for the induced distribution shift. — [arXiv 1011.0686](https://arxiv.org/abs/1011.0686)

### Inferences
**Diagnosis of the project's failure (inference).** Holdout top-1 rising from 47% to 82% while blind judges saw no change fits two documented exploits stacking:
- **Imitator-error exploitation.** Classic offline-MBO objective hacking: the imitator is wrong off-manifold and the optimiser walks there.
- **Recognizer exploitation.** Ensemble-optimised identity cues transfer to held-out face-recognition (FR) models (Liu 2017), so a held-out FR model is not an independent judge.

So "holdout FR top-1" should be retired as the main metric. Measure on real game renders, with a human-aligned check: small blinded 2AFC panels, and possibly DreamSim or a scorer calibrated on human pairwise judgments.

**Cheapest effective safeguards, in order (inference, combining the cited methods):**
1. **Real-render verification of every final answer.** Run K restarts or variants (e.g. 32–128) of imitator-gradient optimisation per photo, render all K in the game (a few seconds at 1000/min), and pick the winner by the real-render score. This is Neural-Adjoint's multi-start idea, with the cheap real simulator replacing the surrogate for selection. Report the real-render score, never the imitator score.
2. **Stay near the data.** Use a Neural-Adjoint boundary loss or a prior/density penalty on DNA (e.g. distance to the training distribution or a learned DNA prior), limit optimisation steps and step size, and start from the translator's prediction. This acts as a trust region.
3. **Ensemble imitators, about 3–5 cheap heads or seeds.** Add a MOPO-style penalty on per-pixel or per-embedding disagreement, and use the same disagreement to pick which proposals to render (Q4).
4. **Conservative/smooth surrogate training.** Add COMs-style negatives: generate optimiser-ascended DNA, render it for real, and train on it. In the project's setting this simply means "render the adversarial proposals". Optionally add RoMA-style smoothness regularisation.
5. **Make the objective itself less hackable.**
   - Use adversarially robust or human-aligned feature extractors, or add geometric terms: landmark/segmentation agreement in both photo and render, like F2P's facial-content loss.
   - Cap the identity-loss contribution.
   - Check the final pick against a held-out, differently trained similarity (human-calibrated, not just another ArcFace).

### Gaps
- No found study quantifies how many active-correction rounds are needed to close the surrogate-exploitation gap for image renderers specifically.
- No found paper evaluates robust-feature or DreamSim identity losses for game-character likeness. Their use here is a reasoned extrapolation.
- The Design-Bench numbers (1.3–2×) are for proteins/materials/robotics tasks. How they carry over to face rendering is unknown.

## 3. How effective are gradient-free methods (CMA-ES, BO, ES, REINFORCE, learning to optimise) at about 50–150 mixed dimensions with about 1000 evaluations/minute?

### Takeaway
With 60k real evaluations per hour, the project is far beyond the budgets where GP Bayesian optimisation shines (hundreds to a few thousand evaluations). Its cubic GP cost and sequential overhead make it a poor fit, except in trust-region or batch forms (TuRBO, Bounce, CASMOPOLITAN). Batch evolution strategies (CMA-ES, with the "margin" fix for integer/categorical genes) are the natural real-renderer optimiser at 50–150 dimensions. The most sample-efficient hybrid uses the imitator only for gradient directions (Guided ES) or local surrogates (L-GSO) while the real renderer decides. Pure REINFORCE has worked (SPIRAL, Learning to Simulate) but needed large distributed RL or simple parameter spaces.

### Cited Findings
- Benchmark on BBOB at 10–60 dimensions: vanilla BO beats CMA-ES "for small dimensions and low budget". High-dimensional BO variants beat both, and the gap widens with dimension. BO is superior "for limited evaluation budgets". — [arXiv 2303.00890](https://arxiv.org/abs/2303.00890); [ACM](https://dl.acm.org/doi/10.1145/3670683)
- NEW: "Vanilla Bayesian Optimization Performs Great in High Dimensions" (Hvarfner, Hellsten, Nardi, ICML 2024) shows that scaling the GP lengthscale prior with dimensionality makes standard BO competitive with or better than specialised high-dimensional BO. — [PMLR](https://proceedings.mlr.press/v235/hvarfner24a.html); [arXiv 2402.02229](https://arxiv.org/abs/2402.02229)
- NEW (titles): "Standard Gaussian Process is All You Need for High-Dimensional BO" (2024); NeurIPS 2024 survey and benchmark of high-dimensional BO; "Understanding High-Dimensional BO" (ICML 2025). — [arXiv 2402.02746](https://arxiv.org/abs/2402.02746); [NeurIPS 2024 D&B](https://papers.nips.cc/paper_files/paper/2024/file/fe0007fcfd707673660ec0f9014bc48e-Paper-Datasets_and_Benchmarks_Track.pdf); [arXiv 2502.09198](https://arxiv.org/abs/2502.09198)
- CMA-based BO hybrids (CMA-BO, CMA-TuRBO, CMA-BAxUS) beat standard BO and TuRBO "by significant margins" on 100–500-dimensional problems. — [Lacuna summary of "High-dimensional BO via Covariance Matrix Adaptation Strategy"](https://lacuna.tiptreesystems.com/work/high-dimensional-bayesian-optimization-via-covariance-matrix-adaptation-strategy/wrk_66d0470d13e6e55064bef8dc70b7296e) (secondary source; primary not fetched)
- TuRBO (NeurIPS 2019) runs several local GP trust regions with Thompson-sampled batches. It targets "high-dimensional problems with several thousand observations", where global BO over-explores. — [arXiv 1910.01739](https://arxiv.org/abs/1910.01739); [NeurIPS](https://proceedings.nips.cc/paper_files/paper/2019/file/6c990b7aca7bc7058f5e98ea909e924b-Paper.pdf)
- Mixed spaces:
  - CASMOPOLITAN (ICML 2021) keeps separate continuous (hyper-rectangle) and categorical (Hamming-ball) trust regions.
  - Bounce uses nested embeddings plus trust regions for combinatorial and mixed spaces, with batch parallelism.
  - NEW: MOCA-HESP (2025) meta-high-dimensional BO for mixed spaces.

  — [arXiv 2102.07188](https://arxiv.org/abs/2102.07188); [arXiv 2307.00618](https://arxiv.org/abs/2307.00618); [arXiv 2508.06847](https://arxiv.org/abs/2508.06847)
- CMA-ES with Margin (2022) fixes plain CMA-ES's stagnation on integer/binary variables (variance collapsing below discretisation granularity) by lower-bounding marginal probabilities. A (1+1) elitist variant followed (2023), and code is public. — [arXiv 2205.13482](https://arxiv.org/abs/2205.13482); [arXiv 2305.00849](https://arxiv.org/abs/2305.00849); [GitHub EvoConJP/CMA-ES_with_Margin](https://github.com/EvoConJP/CMA-ES_with_Margin)
- Guided ES (ICML 2019) uses cheap but biased "surrogate gradients" to define a low-dimensional guiding subspace and concentrates finite-difference random search there. This fits the case "imitator gradient is correlated with, but not equal to, the true gradient". — [PMLR](https://proceedings.mlr.press/v97/maheswaranathan19a.html)
- L-GSO (NeurIPS 2020) beats BO, numerical differentiation and score-function (REINFORCE-style) estimators when parameter dependence is effectively low-dimensional. — [arXiv 2002.04632](https://arxiv.org/abs/2002.04632)
- REINFORCE-type approaches on black-box renderers: SPIRAL needed distributed RL with a discriminator reward ([PMLR](https://proceedings.mlr.press/v80/ganin18a.html)). Learning to Simulate used policy gradient over simulator parameters ([arXiv 1810.02513](https://arxiv.org/abs/1810.02513)). Learning to Paint found a differentiable neural renderer more effective than interacting with non-differentiable simulators ([arXiv 1903.04411](https://arxiv.org/abs/1903.04411)).
- Amortised "learning to optimise" predicts solutions for repeated similar problems and can be orders of magnitude faster than per-instance optimisation. — [Amos, "Tutorial on amortized optimization", FnTML 2023, arXiv 2202.00665](https://arxiv.org/abs/2202.00665)

### Inferences
**Budget arithmetic (inference).**
- CMA-ES at n ≈ 100–150 continuous genes typically needs on the order of 10³–10⁴+ evaluations to make solid progress. Full covariance learning scales roughly with n², which is background knowledge, not verified here.
- At 1000 evaluations/min that is about 1–15 min per face. That is fine for a few hundred "gold" targets but too slow to label tens of thousands of photos.
- So use real-renderer CMA-ES (a) as the gold-standard per-photo optimiser for evaluation and for a high-quality distillation set, and (b) as a final refinement started from the amortised prediction, with a small sigma and a few hundred to a thousand evaluations (seconds to about 1 min).

**Recommended hybrid.**
- **Warm start.** Start from the translator output.
- **Continuous genes.** Use sep-CMA-ES or full CMA-ES with large batches (the population can match the render batch size). Alternatively use Guided ES, with the imitator's gradient as the guiding subspace.
- **Discrete genes.** Use CMA-ES-with-Margin, or enumerate the top-k hair/beard/brow options per photo. These are small categorical sets, and each candidate costs only about 60 ms of render time.
- **Objective.** The objective must still be robust (Q2), because real-renderer optimisation removes imitator exploitation but not recognizer exploitation.

**Why not GP BO here.** GP BO and TuRBO/Bounce are better suited to budgets of a few hundred evaluations. The project's bottleneck is the hackability of the objective, not sample count.

### Gaps
- I found no benchmark of CMA-ES vs BO vs ES specifically on rendered-image objectives at about 100–300 dimensions with budgets of about 10⁴ evaluations. The BBOB study stops at 60 dimensions.
- The "CMA-based BO beats TuRBO at 100–500 dimensions" claim comes from a secondary summary page; I did not verify the primary paper.

## 4. Best practices for active-learning / data-aggregation loops between surrogate and real renderer

### Takeaway
The principled template is DAgger-style aggregation. Repeatedly render the points the current optimiser or translator actually visits, add them to the training set, and retrain on the union. Choose extra samples by ensemble disagreement, refit locally around the current optimum (L-GSO), and keep doing it until the surrogate-vs-real gap on optimiser proposals stops shrinking. One round, as the project tried, is generally not enough, because each retrain moves the optimiser to new blind spots.

### Cited Findings
- DAgger: train, run the current learner to collect data from its own induced distribution, aggregate, retrain on everything, and repeat. This has no-regret guarantees under distribution shift. — [arXiv 1011.0686](https://arxiv.org/abs/1011.0686)
- L-GSO re-fits generative surrogates in a local neighbourhood of the current parameters at each step, reusing simulator samples. It is a local, iterative surrogate–simulator loop rather than one global model. — [arXiv 2002.04632](https://arxiv.org/abs/2002.04632)
- RoMA adapts the proxy model specifically around the current set of candidate solutions. — [arXiv 2110.14188](https://arxiv.org/abs/2110.14188)
- Ensemble disagreement (std across members) is used as the epistemic-uncertainty proxy for adaptive sampling with neural surrogates and is called robust in sequential settings. — [arXiv 2604.19027](https://arxiv.org/abs/2604.19027) (2026, neural-operator surrogate; NEW)
- Ensemble-free batch-mode deep active learning for surrogate regression (student–teacher) reduces the compute cost of deep ensembles. — [arXiv 2211.10360](https://arxiv.org/abs/2211.10360)
- MBRL precedent: MBPO and MOPO pair learned dynamics ensembles with uncertainty-aware use of model rollouts. — [MOPO arXiv 2005.13239](https://arxiv.org/abs/2005.13239)

### Inferences
**Concrete loop for the project (inference).** Each round uses about 20–60k renders, roughly 20–60 min of game time.
- **Proposal sampling, about 40% of the round.** Run the current pipeline (translator → imitator refinement) on a fixed pool of real photos and render the proposals plus several perturbations.
- **Disagreement sampling, about 30%.** Generate many candidate DNAs and render the ones with the highest ensemble-imitator disagreement, measured in identity-embedding space and pixel space.
- **Coverage sampling, about 30%.** Use prior or random samples that respect the engine's active-blend-shape limit (≈128), so the base distribution stays valid.
- **Retrain and track.** Retrain the imitator and translator on everything. The key metric is the real-vs-imitator identity-embedding gap on that round's proposals, not on random validation data.
- **Stopping rule.** Stop when the gap plateaus, or when human 2AFC preference for the new round's outputs over the previous round's stops improving.

**Practical points.**
- Version the dataset by engine layout, so a layout change such as the 128-active-shape cap invalidates or relabels old data and does not silently corrupt it.
- Encode hard engine constraints in the sampler and the optimiser (e.g. top-k sparsity projection on blend shapes), so neither the surrogate nor the optimiser can propose configurations the engine silently ignores.

### Gaps
- The literature I found gives no empirical rule for the number of rounds or the sampling-mix ratios for neural-renderer surrogates. The 40/30/30 mix and per-round sizes above are suggestions, not sourced.
- I found no head-to-head comparison of uncertainty sampling vs optimiser-proposal sampling for image-generating surrogates.

## 5. Amortised inference (params = f(image)) from synthetic renders, and domain adaptation from synthetic renders to real photos

### Takeaway
Amortised prediction trained on simulator output is mainstream:
- the simulation-based inference (SBI) tooling and amortised-optimisation surveys
- Fast F2P's translator (about 1000× faster)
- DI-PCG's 7.6M-parameter diffusion predictor
- Microsoft's synthetic-only face pipelines

The domain gap is handled in four main ways:
- domain-invariant intermediate inputs (face-recognition embeddings, dense landmarks, segmentation)
- highly varied, realistic synthetic data and domain randomisation
- pixel-level translation (CycleGAN/CyCADA, stylisation as in AgileAvatar, dual-domain GAN pairs as in SwiftAvatar)
- self-supervised render-and-compare consistency (Fast F2P's recursive consistency, SMIRK's neural renderer)

### Cited Findings
- SBI and amortisation: SBI needs only simulations, no simulator gradients, parallelises massively, and amortises inference across observations without new simulations. The `sbi` toolkit implements NPE/SNPE, NLE and NRE. — NEW: [sbi reloaded, arXiv 2411.17337](https://arxiv.org/abs/2411.17337); [sbi docs](https://sbi.readthedocs.io/)
- Amortised optimisation tutorial (Amos, FnTML 2023): learned predictors of optimisation solutions, with code. — [arXiv 2202.00665](https://arxiv.org/abs/2202.00665); [GitHub](https://github.com/facebookresearch/amortized-optimization-tutorial)
- Fast F2P's translator maps face-recognition embeddings (a domain-invariant input) to parameters. It is trained self-supervised with recursive consistency on 170k engine renders and is about 1000× faster than iterative fitting. — [arXiv 2008.07132](https://arxiv.org/abs/2008.07132)
- NEW: DI-PCG predicts procedural parameters with an image-conditioned diffusion transformer (7.6M parameters, about 30 GPU-hours) and generalises to in-the-wild images. A diffusion head also represents multimodal or ambiguous solutions. — [CVF CVPR 2025](https://openaccess.thecvf.com/content/CVPR2025/html/Zhao_DI-PCG_Diffusion-based_Efficient_Inverse_Procedural_Content_Generation_for_High-quality_3D_CVPR_2025_paper.html)
- "Fake It Till You Make It" (ICCV 2021) builds a procedurally generated parametric 3D face plus a hand-crafted asset library. It renders realistic, diverse synthetic faces so that landmark and face-parsing models trained on synthetic data alone match real-data accuracy in the wild, without domain adaptation. — [CVF](https://openaccess.thecvf.com/content/ICCV2021/html/Wood_Fake_It_Till_You_Make_It_Face_Analysis_in_the_ICCV_2021_paper.html)
- "3D Face Reconstruction with Dense Landmarks" (ECCV 2022) trains on about 100k synthetic images and predicts 10× more landmarks than usual, then fits a morphable model to them. It reached state of the art in-the-wild at over 150 FPS on one CPU thread. This is the "domain-invariant intermediate representation, then fit" pattern. — [Microsoft Research](https://www.microsoft.com/en-us/research/publication/3d-face-reconstruction-with-dense-landmarks/); [ECVA](https://www.ecva.net/papers/eccv_2022/papers_ECCV/html/6228_ECCV_2022_paper.php)
- Domain randomisation (Tobin et al., 2017): randomising textures, lighting and distractors in simulation let detectors trained only on synthetic images transfer to real ones. — [arXiv 1703.06907](https://arxiv.org/abs/1703.06907)
- CyCADA (ICML 2018) does pixel- and feature-level cycle-consistent adaptation with a task-consistency loss. Synthetic→real segmentation per-pixel accuracy went from 54% to 82%. — [arXiv 1711.03213](https://arxiv.org/abs/1711.03213)
- AgileAvatar stylises real selfies into the avatar domain first, then fits parameters. SwiftAvatar synthesises paired realistic-face/avatar data with dual-domain GANs and trains a light estimator. The Dec-2024 cross-domain regressor uses contrastive and identity-aware losses. — [arXiv 2211.07818](https://arxiv.org/abs/2211.07818); [arXiv 2301.08153](https://arxiv.org/abs/2301.08153); [arXiv 2412.10430](https://arxiv.org/abs/2412.10430)
- NEW: SMIRK's neural renderer plus render-and-compare training bridges the input↔synthesis gap for self-supervised 3D-face regression. — [arXiv 2404.04104](https://arxiv.org/abs/2404.04104)
- MeInGame transfers 3DMM shape to game meshes. An existing photo→3DMM regressor can be followed by a learned 3DMM→game-parameter map. — [arXiv 2102.02371](https://arxiv.org/abs/2102.02371)

### Inferences
**Recommended amortised architecture for the project (inference, assembled from the above).**
- **Inputs (domain-invariant, computed from both photos and renders with the same frozen models).**
  - Several face-recognition embeddings.
  - Dense 2D landmarks or a 3DMM fit (e.g. from an off-the-shelf photo→3DMM/landmark model).
  - A face-parsing mask, plus coarse skin/hair colour statistics.

  The predictor never sees raw pixels, which removes most of the render↔photo style gap.
- **Supervised stage.** Train f(features) → DNA on 0.5–1M real game renders. These are cheap, at about 8–17 h of rendering.
  - Use a parameter loss.
  - Add a cross-entropy head for discrete hair/beard/brow choices.
  - Optionally use a diffusion or mixture head (DI-PCG style), since likeness→DNA is many-to-one.
- **Self-supervised stage on real photos.** Fast F2P's recursive consistency: photo → DNA → imitator render → features ≈ photo features. Keep it conservative, with an ensemble penalty and a boundary loss, and periodically replace imitator renders with real renders of the predicted DNA (Q4 loop).
- **Gold distillation set.** For a few hundred to a few thousand diverse photos, run real-renderer CMA-ES (Q3) with a robust, human-calibrated objective. Use the results both as supervised targets and as the evaluation set.
- **Optional domain bridging.** Use pixel-level translation of CK3 renders toward photo-realism, or of photos toward CK3 style (CycleGAN or modern image-to-image diffusion, SwiftAvatar/AgileAvatar style). This can create more (photo-like, DNA) pairs. Verify that identity is preserved through the translation.
- **Inference cost.** Seconds per photo: feature extraction plus one forward pass, plus an optional short real-render refinement or top-k re-rank (Q2 safeguard 1).

### Gaps
- No found study compares embedding/landmark-input regressors with raw-pixel regressors for game-character parameter prediction in terms of human-judged likeness.
- I found no published numbers on how many real-photo self-supervised steps a translator tolerates before it starts exploiting the imitator. This is presumably the same objective-hacking risk as in Q2.
- Details such as CyCADA's exact task and dataset (GTA→Cityscapes per-pixel accuracy) and the Dense Landmarks data size come from search snippets and were not checked against full text.
