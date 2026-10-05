# 3D face identity reconstruction and facial attribute estimation from one or a few photos: keeping individual distinctiveness (2022–2026)

Research date: 2026-10-05. Access in this session: arxiv.org, openaccess.thecvf.com, now.is.tue.mpg.de, huggingface.co, semantic scholar and project pages (*.github.io) were blocked by the egress proxy. Facts about papers therefore come from (a) web-search result snippets of those pages, marked "(snippet)", and (b) GitHub READMEs and LICENSE files, which I read directly through raw.githubusercontent.com. Licence statements come from the actual LICENSE/README files unless marked otherwise. The web-search budget ran out near the end, so a few items are left as gaps and not filled from memory.

Context: the CK3 pipeline is MediaPipe → MICA (ArcFace → FLAME identity) → analysis-by-synthesis onto a FLAME-2023-Open basis (42 identity axes). Measured problems: MICA β std is about 0.45 against a prior of 1.0; MediaPipe captures 40–70% of the deviation from the mean (eyes about 20%); smiles leak into the lips; eyelid type and lip fullness are poorly measured; flash or lighting corrupts skin tone.

## Q1. Best current methods for identity-accurate face shape from single images and small photo sets, and what NoW, REALY, Stirling and NeRSemble-SVFR show

### Takeaway
On neutral-shape benchmarks the 2023–2026 front-runners are TokenFace (ICCV'23, NoW median 0.76 mm; no code located), FlowFace (CVPR'24, 0.87 mm; dense per-vertex alignment with multi-view fitting), Pixel3DMM (ICLR'26; matches FlowFace on NoW, about 15% better on posed faces; code CC BY-NC) and SHeaP (ICCV'25; 0.93 mm NoW validation; easy to run; CC BY-NC). MICA (0.90 mm) is still near the top of the neutral-shape benchmarks. None of these benchmarks directly measures whether a method keeps a face's distinctiveness. The most transferable idea for this project is the newer "dense screen-space prior + 3DMM fitting" family (FlowFace, Pixel3DMM, RealDenseFace, Wood et al.'s 703 dense landmarks). It replaces sparse MediaPipe landmarks with per-vertex correspondences and fits shape with weak priors. It can also fit jointly across several photos.

### Cited Findings
**Benchmarks**
- NoW: 2054 images of 100 subjects, with ground-truth scans. It evaluates neutral shape under varying viewing angle, lighting and occlusion. The prediction is rigidly aligned to the scan, with rotation, translation **and scaling**, using 7 landmarks, then refined by scan-to-mesh distance. — [NoW evaluation repo](https://github.com/soubhiksanyal/now_evaluation); dataset size from [search snippet of NoW leaderboard](https://now.is.tue.mpg.de/nonmetricalevaluation.html)
- NoW non-metrical leaderboard (snippet, "last updated June 19, 2024"): TokenFace 0.76 mm median (rank 1), FlowFace 0.87 mm (rank 2), MICA 0.90 mm (rank 3), DECA 1.09 mm (rank 8). A separate metrical leaderboard also exists. — [NoW non-metrical leaderboard (snippet)](https://now.is.tue.mpg.de/nonmetricalevaluation.html); [metrical leaderboard](https://now.is.tue.mpg.de/metricalevaluation.html)
- REALY (ECCV 2022) is a region-aware benchmark. It reports NMSE separately for nose, mouth, forehead and cheek, aligns with 85 keypoints, and uses bidirectional evaluation. Its data is now distributed through the Headspace dataset agreement. It also introduced the HIFI3D++ 3DMM. — [REALY repo](https://github.com/czh-98/REALY); [paper](https://arxiv.org/abs/2203.09729)
- Pixel3DMM introduced a new benchmark with diverse expressions, viewpoints and ethnicities. It is the first to evaluate both posed and neutral geometry, and is hosted as the "NeRSemble SVFR" single-view face reconstruction benchmark. — [Pixel3DMM arXiv (snippet)](https://arxiv.org/abs/2505.00615); [NeRSemble SVFR benchmark](https://kaldir.vc.cit.tum.de/nersemble_benchmark/benchmark/svfr) (page blocked, existence from search)

**Methods (newest first)**
- **RealDenseFace** (arXiv 2608.09238, Aug 2026; newest): a network predicts two dense UV-space maps, a UV-to-image correspondence map and a relative-depth map. A custom Gauss-Newton solver then fits the 3DMM in a few iterations. It supports single-image, offline-sequence and online settings, runs at 84 FPS on an RTX 4090, and reports state-of-the-art results on NeRSemble SVFR. It explicitly targets the "tens of seconds per image" cost of earlier dense-prior fitting. — [arXiv PDF (snippet)](https://arxiv.org/pdf/2608.09238). I did not find code or licence.
- **Pixel3DMM** (Giebenhain, Kirschstein, Rünz, Agapito, Nießner; ICLR 2026): ViTs built on DINO features predict per-pixel surface normals and FLAME UV coordinates, and these constrain FLAME fitting. Training used 3 high-quality 3D datasets registered to FLAME (>1,000 identities, 976K images). On NoW it "achieves the same metrics as FlowFace … but performs worse than TokenFace". On FaceScape and the new benchmark it "significantly outperforms TokenFace", and it is more than 15% better on posed-expression geometry. — [arXiv (snippet)](https://arxiv.org/abs/2505.00615); [ICLR 2026 poster](https://iclr.cc/virtual/2026/poster/10009198)
- Pixel3DMM code is CC BY-NC 4.0. It needs a FLAME account, PyTorch3D, nvdiffrast and CUDA 11.8 in the reference environment, and it **uses MICA in preprocessing**. It supports FLAME2023_no_jaw and has "online" and "joint" tracking stages, so it can work on multiple frames. — [GitHub README](https://github.com/SimonGiebenhain/pixel3dmm); [LICENSE](https://github.com/SimonGiebenhain/pixel3dmm/blob/master/LICENSE)
- **SHeaP** (Schoneveld et al., ICCV 2025): predicts FLAME together with 2D Gaussians rigged to the mesh, and is trained self-supervised on 2D video only. It reportedly beats prior self-supervised methods on NoW (neutral) and on a new non-neutral benchmark. — [CVF (snippet)](https://openaccess.thecvf.com/content/ICCV2025/html/Schoneveld_SHeaP_Self-Supervised_Head_Geometry_Predictor_Learned_via_2D_Gaussians_ICCV_2025_paper.html)
- SHeaP repo: FLAME-parameter inference only, 224×224 head crops, needs only `torch>=2.0`. It ships two models: "paper", which is best on NoW (validation median 0.93 mm), and "expressive", which was "trained for longer with less regularisation and tends to be more expressive". Licence is CC BY-NC 4.0, and it uses FLAME2020. — [GitHub](https://github.com/nlml/sheap) (README + LICENSE.txt)
- **FlowFace** (Taubner et al., CVPR 2024): stage 1 is a ViT plus a recurrent refinement network that predicts the screen-space 2D position of **every 3DMM vertex**. It is trained on high-quality 3D scan annotations, not on weak or synthetic labels. Stage 2 jointly fits the 3DMM **across multiple views**. It was evaluated on NoW in single-view and multi-view settings. — [CVPR paper (snippet)](https://openaccess.thecvf.com/content/CVPR2024/papers/Taubner_3D_Face_Tracking_from_2D_Video_through_Iterative_Dense_UV_CVPR_2024_paper.pdf); [arXiv](https://arxiv.org/abs/2404.09819)
- **TokenFace** (ICCV 2023): a transformer with separate tokens per facial component, plus temporal transformers. It is trained on hybrid 2D+3D data and reported state of the art "by a large margin" on NoW and Stirling. — [CVF (snippet)](https://openaccess.thecvf.com/content/ICCV2023/html/Zhang_Accurate_3D_Face_Reconstruction_with_Facial_Component_Tokens_ICCV_2023_paper.html)
- **3DDFA-V3** (CVPR 2024 Highlight): the Part Re-projection Distance Loss turns facial-part segmentation into 2D point sets and matches their distribution with grid-anchor statistics. Its REALY overall score is about 1.435, with regions ranging from about 1.07 to 1.86 (snippet). Code is MIT, but the model is BFM-based (35,709-vertex BFM topology, as in Deep3D and HRN). It outputs 68/106/134 landmarks and 8-part segmentation, offers a MobileNet-V3 variant and a CPU renderer, and the reference environment is torch 1.12.1 with cu102. — [CVF (snippet)](https://openaccess.thecvf.com/content/CVPR2024/html/Wang_3D_Face_Reconstruction_with_the_Geometric_Guidance_of_Facial_Part_CVPR_2024_paper.html); [arXiv html (snippet)](https://arxiv.org/html/2312.00311v3); [GitHub + MIT LICENSE](https://github.com/wang-zidu/3DDFA-V3)
- **HRN** (CVPR 2023, Alibaba DAMO): a hierarchical representation network that was "top-1 on REALY" as of March 2023. It has a **multi-view mode (MV-HRN)**, which fits several images of one subject in about 1 minute, while single-view inference takes under 1 s. Licence is Apache-2.0, and training code is not released. — [GitHub + LICENSE](https://github.com/youngLBW/HRN)
- **MICA** (ECCV 2022): ArcFace (ResNet100 trained on Glint360K) feeds a FLAME neutral shape regressor. It was trained on about 2,300 subjects from 8 merged 3D datasets, tested on Stirling and NoW, and needs insightface antelopev2/buffalo_l models. Licence is MPI non-commercial scientific research. — [GitHub + LICENSE](https://github.com/Zielon/MICA); [paper (snippet)](https://arxiv.org/pdf/2204.06607)
- **SMIRK** (CVPR 2024) targets expression fidelity through a neural renderer, not a differentiable renderer. Its shape encoder is **pre-trained "using only the extracted landmarks and the output of MICA"**. Code is MIT, but it needs a FLAME account. — [GitHub README + LICENSE](https://github.com/georgeretsi/smirk); [CVPR paper](https://openaccess.thecvf.com/content/CVPR2024/papers/Retsinas_3D_Facial_Expressions_through_Analysis-by-Neural-Synthesis_CVPR_2024_paper.pdf)
- DECA and EMOCA (FLAME regressors from MPI) are both under MPI non-commercial research licences. — [DECA](https://github.com/yfeng95/DECA); [EMOCA](https://github.com/radekd91/emoca)
- **Dense landmarks** (Wood et al., Microsoft, ECCV 2022): 703 landmarks covering the whole head, including eyes and teeth. The model is trained only on synthetic renders with perfect labels, and fitting a 3DMM to these landmarks gave state-of-the-art in-the-wild monocular reconstruction. The authors show that dense landmarks are "an ideal signal for integrating face shape information across frames", in monocular and multi-view settings, at more than 150 FPS on one CPU thread. — [Project](https://microsoft.github.io/DenseLandmarks/); [ECCV](https://www.ecva.net/papers/eccv_2022/papers_ECCV/html/6228_ECCV_2022_paper.php)
- **VGGHeads** (2024): a 1M-image fully synthetic, diffusion-generated dataset. One model detects heads and regresses 3D head meshes in a single step. Code is MIT, and ONNX/PyTorch weights are on HF. — [GitHub + LICENSE](https://github.com/KupynOrest/head_detector)
- **NPHM** (CVPR 2023) is a neural (SDF-based) parametric head model, not a mesh 3DMM. Code is MIT, but the dataset has a separate licence. — [GitHub](https://github.com/SimonGiebenhain/NPHM)
- **Metrical Photometric Tracker** (MICA authors) is an analysis-by-synthesis FLAME tracker for video and image sequences. — [GitHub](https://github.com/Zielon/metrical-tracker)
- A 2026 arXiv paper on profile-specific 3DMM regression notes that recent single-image methods "commonly regress 3DMM/FLAME parameters directly from RGB input". I could not read its details. — [arXiv 2605.01746 (snippet)](https://arxiv.org/pdf/2605.01746)

### Inferences
- NoW and REALY report absolute surface error after a **similarity (scaled) alignment**. A method that pulls shape toward the mean loses little on this metric, because most of a face's surface is close to average anyway. So a low NoW score does not show that a method keeps distinctiveness, and none of the benchmarks I found reports a "fraction of deviation-from-mean recovered" statistic like the one this project measured (40–70%). The project's own render-based test (ground-truth β → render → estimate) is the right instrument and should be kept as the acceptance test for any replacement.
- The project's main gap is MediaPipe capturing only about 20% of eye deviation. The most promising swap is a **dense per-vertex correspondence prior** in place of sparse landmarks: Pixel3DMM's UV and normal maps, or the same idea in FlowFace and RealDenseFace. With per-vertex targets, the analysis-by-synthesis fit is data-dominated rather than prior-dominated, and these methods are built for multi-view or multi-frame joint fitting. Pixel3DMM is the only one of the three with public code. It is FLAME-based (direct fit to the project's basis) but CC BY-NC, which is acceptable for a hobby or non-commercial project.
- SHeaP is the lowest-friction drop-in FLAME regressor (torch ≥ 2.0 only, so it is likely to run on an RTX 5080/Blackwell without old CUDA pins). Its "expressive" checkpoint, trained with less regularisation, is worth A/B-testing against MICA on the project's shrinkage test. Neither checkpoint is documented as fixing shrinkage.
- SMIRK's shape branch is distilled from MICA, so it is expected to inherit MICA's shrinkage. Use it for expression and jaw estimation, not identity.
- 3DDFA-V3 and HRN are BFM-based. To feed the FLAME→CK3 transfer they would need a BFM→FLAME conversion or would serve only as auxiliary signals. 3DDFA-V3's 8-part segmentation and 134 landmarks are, however, MIT-licensed and could act as extra 2D constraints, especially on lip and eye contours.
- Reference environments pin old CUDA: cu102 (3DDFA-V3), cu117 (SMIRK), cu118 (Pixel3DMM) and CUDA 10 (FFHQ-UV). An RTX 5080 (Blackwell) generally needs a recent CUDA 12.x PyTorch build, so expect to rebuild PyTorch3D and nvdiffrast. This is background knowledge, not verified in this session.

### Gaps
- Exact NoW numbers (median/mean/std) for Pixel3DMM, SHeaP test set, SMIRK, EMOCA, 3DDFA-V3, HRN and Stirling/ESRC results could not be read, because the leaderboard and arXiv were blocked.
- NeRSemble SVFR leaderboard values (neutral vs posed) could not be read.
- I found no public code or weights for TokenFace, FlowFace, Wood et al. dense landmarks or RealDenseFace (unconfirmed, not verified absent).
- Licence of FLAME 2023 Open (believed to be CC-BY-4.0) was not verified in this session.

## Q2. Methods that address shrinkage or regression-to-the-mean ("identity strength"), and the statistically right correction

### Takeaway
No recent 3D-face paper I found targets "shrinkage of identity coefficients" directly. The levers in the literature are indirect: dense, data-dominated fitting (Pixel3DMM, FlowFace), weaker regularisation (SHeaP "expressive"), perceptual shape critics (Otto et al. 2023), multi-view joint fitting (FlowFace, MV-HRN, Wood et al.), and fully Bayesian 3DMM inference (Schönborn et al.). The statistical fix is a measurement-error correction calibrated on the project's own synthetic renders. Estimate each axis's attenuation slope and noise from (true β, measured β) pairs, average photos to cut noise, then de-attenuate (divide by the slope). Be aware that a Bayesian posterior mean deliberately shrinks again, so it is the wrong target when the goal is a population with the right spread.

### Cited Findings
- "Regression dilution / attenuation": noise in the regressor biases slopes toward zero, and more noise means more attenuation. — [Wikipedia: Regression dilution](https://en.wikipedia.org/wiki/Regression_dilution)
- The reliability ratio λ is true-signal variance divided by observed variance, and dividing the attenuated slope by λ gives an approximately unbiased slope. Regression calibration replaces the noisy predictor with E[X | observed]. SIMEX adds increasing simulated noise and extrapolates back to zero noise. — [MetricGate: attenuation bias correction](https://metricgate.com/docs/attenuation-bias-correction/); [MetricGate: measurement error correction](https://metricgate.com/docs/measurement-error-correction/) (secondary tutorial sources; the concepts are standard)
- Errors-in-variables methods (Deming regression, total least squares) model noise on both axes. — [MetricGate EIV](https://metricgate.com/docs/errors-in-variables-regression/)
- Fully probabilistic 3DMM inference: Schönborn et al. (IJCV 2017) infer the **posterior distribution** of 3DMM parameters from a single image using Metropolis-Hastings propose-and-verify sampling. — [Schönborn et al. IJCV 2017 PDF](https://shapemodelling.cs.unibas.ch/gravis-literature/publications/2017/2017_Schoenborn_IJCV.pdf)
- **Perceptual Shape Loss** (Otto et al., Disney Research and ETH, Pacific Graphics 2023): a discriminator-style critic that scores how well a *shaded render of the geometry alone* matches the photo, without estimating albedo or lighting. It can be added to optimisation energies or to regressor training and improved state-of-the-art results. — [arXiv](https://arxiv.org/abs/2310.19580); [Eurographics DL](https://diglib.eg.org/handle/10.1111/cgf14945)
- SHeaP ships a less-regularised "expressive" model, which "tends to be more expressive", alongside the NoW-optimal "paper" model. This is an explicit trade-off between benchmark score and expressiveness. — [SHeaP README](https://github.com/nlml/sheap)
- Multi-view and multi-frame aggregation as an identity-fusion mechanism:
  - FlowFace jointly fits one 3DMM across multiple views. — [CVPR'24](https://openaccess.thecvf.com/content/CVPR2024/papers/Taubner_3D_Face_Tracking_from_2D_Video_through_Iterative_Dense_UV_CVPR_2024_paper.pdf)
  - Wood et al. show dense landmarks integrate shape across frames and views. — [DenseLandmarks](https://microsoft.github.io/DenseLandmarks/)
  - MV-HRN fits multi-view photos of one subject. — [HRN](https://github.com/youngLBW/HRN)
  - Pixel3DMM has "joint" tracking over frames. — [Pixel3DMM](https://github.com/SimonGiebenhain/pixel3dmm)
  - Avat3r takes about 4 images with inconsistent expressions. — [Avat3r (snippet)](https://arxiv.org/abs/2502.20220)
- ArcFace, MICA's input, is "invariant to illumination, expression, rotation, occlusion, and camera parameters", which is why MICA uses it. — [MICA paper (snippet)](https://arxiv.org/pdf/2204.06607)

### Inferences
- **Why MICA shrinks (hypothesis):** a regressor trained with an L2-type loss outputs the conditional mean E[β | ArcFace]. When the embedding carries only part of the shape information (ArcFace is tuned for discrimination, not geometry), that conditional mean is pulled toward the population mean in exactly the directions the embedding underdetermines. That is regression to the mean by construction. Training on about 2,300 subjects limits how well rarer shapes (and under-represented ethnic shapes) are covered. Both points are inferences, not stated in the sources found.
- **Recommended calibration recipe (project-specific):**
  1. Use the existing render harness. Sample true β ~ N(0, I) on the 42 axes (plus the vanilla-gene and eyelid targets), render varied pose, lighting and expression, and run the pipeline to get measured β̂.
  2. Per axis i (or as a full matrix), regress β̂ on β: β̂ ≈ A·β + b + ε, with Cov(ε) = Σ. A matrix A also captures cross-talk between axes.
  3. **Unbiased (de-attenuated) estimate:** β* = A⁻¹(β̂ − b). This restores the population spread (std ≈ 1) but amplifies noise by 1/a_ii.
  4. **Bayesian posterior mean** with prior N(0, I): E[β | β̂] = Aᵀ(AAᵀ + Σ)⁻¹(β̂ − b). This is MSE-optimal per person but **shrinks again**, which is exactly the "average face" problem. Use it only as a sanity bound.
  5. For a population that looks right, use a variance-matching compromise. Scale the posterior so that the predicted population std matches the prior std (the "constrained Bayes" idea; see Gaps). Equivalently, apply the de-attenuated estimate clipped at the CK3 gene ranges.
  6. **Multiple photos:** averaging N photos' β̂ (inverse-variance weighted) cuts the noise term Σ/N, but **does not remove the attenuation A**, because that bias is systematic. Average first, then apply A⁻¹. The noise amplification of A⁻¹ also shrinks with N, so multi-photo input makes de-attenuation safe. This fits the project's observation that multi-photo input "helps somewhat" without fixing shrinkage.
  7. Make A depend on the measurement channel: MediaPipe eye axes at about 20% recovery need roughly a ×5 gain, so their noise must be estimated carefully, or those axes should be replaced by a better sensor (Q3) before de-attenuating.
- The calibration is only valid if the synthetic renders resemble real photos (lighting, sensor, expression). A domain gap in the renders would make A wrong on real data. SIMEX-style checks (add synthetic noise or perturbation and watch the trend) or a small set of real people with known 3D scans can validate it.
- A cheap way to bring in "identity strength" without retraining is ArcFace self-consistency. Render the fitted CK3/FLAME face, embed it with ArcFace, and compare with the photos' mean embedding. Increase the deviation magnitude while similarity improves. This is an inference; there is a domain gap between game renders and photos.

### Gaps
- I found no 2023–2026 paper that explicitly measures or corrects "identity coefficient shrinkage" in FLAME regressors (such as MICA's β variance). The literature search was cut short by the search budget.
- The "constrained Bayes" (ensemble-variance-matching) estimator is attributed in my background knowledge to T. A. Louis (JASA 1984) and M. Ghosh (1992). It could not be verified with a source in this session.
- I found no calibrated-uncertainty FLAME regressor (one that outputs per-axis variance) with public code; the 2017 MCMC approach is BFM-era.

## Q3. Dense landmark and face-parsing models more accurate than MediaPipe for eyelids, lips and eye shape, plus fine-grained attribute classifiers

### Takeaway
Better 2D sensors exist, but none outputs "eyelid type". The strongest options are:
- Sapiens/Sapiens2: 243 face keypoints out of 308. Sapiens is CC BY-NC; Sapiens2 (Apr 2026) uses a custom Meta licence.
- FaRL via the `facer` library: MIT. LaPa and CelebAMask-HQ parsing, 300W/WFLW landmarks, and CelebA attributes.
- SegFace and FaceXFormer: MIT. Transformer parsing and multi-task analysis.
- 3DDFA-V3: 134 landmarks plus part segmentation, MIT.
- Pixel3DMM: per-pixel FLAME UV and normals, which is effectively a dense landmark field.

For eyelid type (monolid / inner double / double) I found no public dataset or pretrained classifier. A small custom classifier on frozen FaRL, DINO or Sapiens features, trained on a few hundred self-labelled eye crops, is the practical route.

### Cited Findings
- **Sapiens** (Meta, ECCV 2024 best-paper candidate): pretrained on 300M human images, with tasks for 2D pose, part segmentation, depth and normals. Licence is **CC BY-NC 4.0**. — [GitHub + LICENSE](https://github.com/facebookresearch/sapiens)
- The Sapiens keypoint set has 308 keypoints, **243 of them facial**, compared with the usual 68. — [LearnOpenCV overview (secondary)](https://learnopencv.com/sapiens-human-vision-models/); [HF model card](https://huggingface.co/facebook/sapiens-pose-1b-torchscript)
- **Sapiens2** (released 24 April 2026): pretrained on 1B human images. It covers 308-keypoint pose, 29-class body-part segmentation, normals, pointmaps and matting, in 0.4B–5B sizes (0.1B–5B per one source), and runs standalone with torch and safetensors. Sapiens2-5B reportedly reaches 82.3 mAP on 308-kp in-the-wild (snippet). Licence is a custom "Sapiens2 License" granting "non-exclusive, worldwide, non-transferable and royalty-free limited license". I did not see a non-commercial clause in a grep, but this needs verification. — [GitHub README + LICENSE.md](https://github.com/facebookresearch/sapiens2); [arXiv 2604.21681 (snippet)](https://arxiv.org/pdf/2604.21681)
- **FaRL** (Microsoft, CVPR 2022 representation, MIT): face parsing F1-mean 93.88 on LaPa (94.04 with ep64) and 89.56 on CelebAMask-HQ. Face alignment NME 2.93 on 300W (inter-ocular), 3.96 on WFLW and 0.943 on AFLW-19 (diag). — [FaRL GitHub + MIT LICENSE](https://github.com/FacePerceiver/FaRL)
- **facer** (MIT) wraps FaRL parsing (LaPa/CelebM at 448 px), alignment (300W/WFLW/AFLW19) and a **CelebA attribute model at 92.06% accuracy**. FaRL parsing is also available in `batch-face` for batching. — [facer GitHub](https://github.com/FacePerceiver/facer)
- **SegFace** (AAAI 2025, MIT): a transformer with class-specific tokens for long-tail classes. Mean F1 is 93.03 on LaPa and 88.96 on CelebAMask-HQ, and a MobileNetV3 variant runs at about 96 FPS. — [GitHub + LICENSE](https://github.com/Kartik-3004/SegFace)
- **FaceXFormer** (ICCV 2025, MIT): one transformer for 9 tasks (parsing, landmarks, head pose, attributes, age/gender/race, expression, visibility) at 33 FPS. — [GitHub + LICENSE](https://github.com/Kartik-3004/facexformer)
- **STAR loss** (CVPR 2023) is a landmark-detection loss that reduces semantic ambiguity along contours, with code on GitHub. Its licence was not checked. — [GitHub](https://github.com/ZhenglinZhou/STAR)
- 3DDFA-V3 outputs 68/106/134 landmarks and an 8-part segmentation derived from its 3D fit (MIT code, BFM). — [GitHub](https://github.com/wang-zidu/3DDFA-V3)
- Wood et al. predict 703 dense landmarks, including the eye region, trained on synthetic data. — [DenseLandmarks](https://microsoft.github.io/DenseLandmarks/)
- MediaPipe is Apache-2.0. — [GitHub LICENSE](https://github.com/google-ai-edge/mediapipe)
- Search for monolid/double-eyelid classifiers returned mainly medical eyelid-lesion work plus a hobby Roboflow "eye shape" dataset that includes a monolid class. I found no academic benchmark. — [Roboflow Universe eye-shape](https://universe.roboflow.com/face-hqu83/eye-shape); [eyelid lesion DL (unrelated)](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC10011394/)

### Inferences
- Standard parsing label sets (LaPa about 11 classes; CelebAMask-HQ 19 classes, including upper and lower lip as separate classes) segment the **eye opening and lips** but **not the eyelid crease**. Lip thickness, measured as vermilion height over mouth width from upper- and lower-lip masks, should therefore be measurable much more directly from FaRL or SegFace masks than from MediaPipe contour points. The lip-class detail is background knowledge about these datasets and was not fetched.
- CelebA's 40 binary attributes (via facer/FaRL) include attributes such as "Big_Lips" and "Narrow_Eyes" (from memory, not fetched this session). Usable as weak extra signals, but they are coarse and annotation-noisy.
- **Eyelid-type pipeline:** crop each eye at high resolution using Sapiens or FaRL landmarks. Hand-label a few hundred crops into monolid, inner/partial double and double, taking care to balance ethnicity. Train a linear or MLP probe on frozen DINOv2, FaRL or Sapiens features. The project has only 8 eyelid genes, so a 3-way plus confidence output maps directly onto them. Sapiens's 243 facial keypoints may also include eyelid-contour points, but whether the crease is annotated is unknown (gap).
- For eye shape (palpebral fissure width, height, tilt and canthal angle), measure directly from high-resolution dense landmarks or parsing masks and fit the CK3 eye genes to those measurements. That avoids pushing eye shape through the global FLAME β, where MediaPipe recovers only about 20% of the deviation.
- Sapiens and Pixel3DMM are CC BY-NC, and Sapiens2's licence needs checking; for a hobby project that is fine. FaRL, facer, SegFace, FaceXFormer, VGGHeads and MediaPipe are MIT or Apache.

### Gaps
- I found no public pretrained eyelid-type (monolid/double) or lip-fullness regressor with a citable accuracy.
- I could not confirm whether the Sapiens 243 face keypoints include an eyelid-crease contour, or their per-region accuracy compared with MediaPipe's 478-point mesh.
- No head-to-head accuracy figures between MediaPipe Face Mesh and FaRL, Sapiens or dense landmarks on eye and lip regions were found.

## Q4. Expression neutralisation (neutral identity from smiling photos) and illumination/albedo estimation for skin tone

### Takeaway
For neutralisation, the practical approach is to fit identity and expression **jointly** with a strong expression model and keep only identity. SMIRK, EMOCA, Pixel3DMM or SHeaP estimate the FLAME expression and jaw; the project then refits the CK3 identity with that expression held fixed, or down-weights the lips and mouth when photos are smiling. ArcFace-based identity (MICA, Arc2Face) is designed to be expression-invariant, and Arc2Face's 2025 Expression Adapter can synthesise a *neutral* image of the same identity as an extra input, though that is hallucinated. For skin tone:
- TRUST (ECCV 2022, non-commercial, FLAME-based) is the established unbiased-albedo baseline.
- HUST (ICCV 2025) reports the best FAIR-benchmark ITA error and bias, but its code release is unconfirmed.
- FFHQ-UV (MIT code, CVPR 2023) provides an evenly lit UV texture prior and RGB fitting.

### Cited Findings
- SMIRK reconstructs "extreme, asymmetric, and subtle expressions". It replaces the differentiable renderer with a neural renderer that regenerates the face from the predicted mesh and sparse input pixels, and it is trained on LRS3, MEAD, CelebA and FFHQ. — [CVPR 2024 paper](https://openaccess.thecvf.com/content/CVPR2024/papers/Retsinas_3D_Facial_Expressions_through_Analysis-by-Neural-Synthesis_CVPR_2024_paper.pdf); [GitHub](https://github.com/georgeretsi/smirk)
- Pixel3DMM is more than 15% better on **posed-expression** geometry, and its benchmark scores both posed and neutral geometry. That makes it the best-documented option for separating expression from identity in one fit. — [arXiv (snippet)](https://arxiv.org/abs/2505.00615)
- SHeaP reports gains on a non-neutral-expression benchmark and in emotion classification, which signals expressive meshes. — [CVF (snippet)](https://openaccess.thecvf.com/content/ICCV2025/html/Schoneveld_SHeaP_Self-Supervised_Head_Geometry_Predictor_Learned_via_2D_Gaussians_ICCV_2025_paper.html)
- MICA relies on ArcFace's invariance to expression, illumination and pose to predict a **neutral** metrical shape. — [MICA (snippet)](https://arxiv.org/pdf/2204.06607)
- **Arc2Face** (ECCV 2024 oral; code MIT; built on Stable Diffusion; trained on WebFace42M) generates images of a subject from its ArcFace embedding alone. Its **Expression Adapter** (Oct 2025) generates "any subject under any facial expression" with blendshape guidance. Weights are on HF. — [GitHub README + LICENSE](https://github.com/foivospar/Arc2Face)
- 3D-only neutralisation research exists, such as a GCN autoencoder with a GAN that maps an expressive latent to a neutral one (2021), and an information-bottleneck VAE with separate neutral and expressive decoders (WACV 2022). Both operate on 3D meshes and were validated on CoMA and BU-3DFE. — [arXiv 2104.10273](https://arxiv.org/pdf/2104.10273); [WACV 2022](https://openaccess.thecvf.com/content/WACV2022/html/Sun_Information_Bottlenecked_Variational_Autoencoder_for_Disentangled_3D_Facial_Expression_Modelling_WACV_2022_paper.html)
- **TRUST** (ECCV 2022) names and quantifies racial bias in albedo estimation. It introduced the FAIR synthetic benchmark (ITA-based skin-tone error plus bias score) and resolves the light/albedo ambiguity with "scene disambiguation" cues. It reported a 57% lower total score (35% lower average ITA error, 77% lower bias) than the prior state of the art. It is DECA/FLAME-based, and its licence is **non-commercial research**. — [GitHub README + LICENSE](https://github.com/HavenFeng/TRUST)
- **HUST** (ICCV 2025) learns a high-fidelity texture codebook (VQGAN) from large RGB face sets, then adapts from texture to the albedo domain unsupervised. On FAIR it reports the lowest average ITA error (11.20) and bias score (1.58). The authors "state their code, models, and training data will be made publicly available" (snippet). — [CVF](https://openaccess.thecvf.com/content/ICCV2025/html/Ran_HUST_High-Fidelity_Unbiased_Skin_Tone_Estimation_via_Texture_Quantization_ICCV_2025_paper.html); [arXiv 2406.13149 (snippet)](https://arxiv.org/html/2406.13149v1)
- FitMe and Relightify learn large facial-reflectance priors with StyleGAN and latent diffusion respectively, and are cited by HUST as strong albedo reconstructors. — [HUST arXiv (snippet)](https://arxiv.org/html/2406.13149v1)
- **FFHQ-UV** (CVPR 2023): more than 50k UV textures with even lighting, neutral expression and cleaned facial regions, built through StyleGAN-based normalisation. It includes an RGB-fitting pipeline and a FLAME conversion recipe, and was mirrored on HF in April 2026. Code licence is MIT, and the reference environment is CUDA 10.0 / Python 3.7. — [GitHub + LICENSE](https://github.com/csbhr/FFHQ-UV)

### Inferences
- **Neutralisation recipe for CK3:**
  1. Run an expression-capable estimator (Pixel3DMM tracker, SMIRK or SHeaP-expressive) to get FLAME expression and jaw per photo.
  2. In the analysis-by-synthesis fit, optimise the 42 identity axes **shared across photos** together with **per-photo** expression and pose, so the expression absorbs the smile.
  3. Weight mouth and lip-region residuals by an expression-intensity score, or drop them for smiling photos.
  4. Prefer neutral-mouth photos for the lip genes.
  Multi-photo joint fitting with shared identity is the standard way expression leakage is suppressed in multi-view methods (FlowFace, Wood et al.).
- Arc2Face-generated neutral images can serve as extra pseudo-photos, but they come from ArcFace embeddings and so carry the same "identity-only, detail-poor" information MICA uses. Their shape detail (lips, eyelids) is likely regressed toward a diffusion prior, so treat them as low-weight evidence, not ground truth. The same caution applies to multi-view hallucination methods (FaceLift, Arc2Avatar).
- **Skin tone:** for CK3's limited skin-colour gene, a full albedo map is unnecessary. A robust approach is to estimate albedo per photo (TRUST, or HUST if released), convert the cheek and forehead mean to ITA or Lab, and take the **median across photos**. Down-weight flash photos, detectable through specular highlights and a strong frontal-light estimate from the SH lighting. TRUST's FLAME basis matches the project, and its non-commercial licence is acceptable for a hobby use.

### Gaps
- I could not confirm whether HUST code or weights are released, or Relightify/FitMe code licences.
- I found no image-based "neutralise identity from a smiling photo" method with public code from 2023–2026 beyond the joint identity+expression fitting in the reconstructors above.
- I found no quantitative comparison of TRUST and HUST on real (non-synthetic) flash photos.

## Q5. Which have usable code and weights on a single consumer GPU, and their licences; plus Gaussian-avatar and generative methods

### Takeaway
Everything below runs on a single consumer GPU for inference, but older repos pin CUDA 10–11.8 and need porting for an RTX 5080.
- Practical, permissive (MIT/Apache) for this project: 3DDFA-V3 (BFM), HRN with MV-HRN (BFM), FaRL/facer, SegFace, FaceXFormer, VGGHeads, Arc2Face code, LAM, GAGAvatar, FFHQ-UV, MediaPipe.
- Non-commercial but FLAME-native and most relevant: Pixel3DMM, SHeaP, MICA, DECA/EMOCA, TRUST, Sapiens.
- SMIRK code is MIT.
- FLAME itself needs a FLAME account.

Gaussian avatar-from-photo methods (LAM, GAGAvatar, Avat3r, FaceLift, Arc2Avatar) produce renderable heads, not 3DMM identity coefficients. They are at most auxiliary, as generators of extra views or as FLAME-tracked outputs, and their multi-view hallucinations can themselves regress to the mean.

### Cited Findings
| Method | Licence (code / weights) | Notes |
|---|---|---|
| Pixel3DMM | CC BY-NC 4.0; FLAME account | CUDA 11.8 reference; uses MICA in preprocessing — [repo](https://github.com/SimonGiebenhain/pixel3dmm) |
| SHeaP | CC BY-NC 4.0; FLAME2020 | torch ≥ 2.0, 224 px crops — [repo](https://github.com/nlml/sheap) |
| MICA | MPI non-commercial | insightface models — [repo](https://github.com/Zielon/MICA) |
| SMIRK | MIT code; FLAME needed | shape encoder pre-trained from MICA output — [repo](https://github.com/georgeretsi/smirk) |
| DECA / EMOCA | MPI non-commercial | — [DECA](https://github.com/yfeng95/DECA), [EMOCA](https://github.com/radekd91/emoca) |
| 3DDFA-V3 | MIT (BFM model has its own terms) | torch 1.12 cu102 reference; MobileNet variant — [repo](https://github.com/wang-zidu/3DDFA-V3) |
| HRN / MV-HRN | Apache-2.0 | about 1 min multi-view fit; no training code — [repo](https://github.com/youngLBW/HRN) |
| Deep3DFaceRecon_pytorch | MIT | BFM-based baseline — [repo](https://github.com/sicxu/Deep3DFaceRecon_pytorch) |
| VGGHeads | MIT; weights on HF | single-step detection + head mesh — [repo](https://github.com/KupynOrest/head_detector) |
| NPHM | MIT code; dataset separately licensed | — [repo](https://github.com/SimonGiebenhain/NPHM) |
| Sapiens | CC BY-NC 4.0 | — [repo](https://github.com/facebookresearch/sapiens) |
| Sapiens2 | custom Meta "Sapiens2 License" (royalty-free limited grant; terms to verify) | — [repo](https://github.com/facebookresearch/sapiens2) |
| FaRL / facer | MIT | — [FaRL](https://github.com/FacePerceiver/FaRL), [facer](https://github.com/FacePerceiver/facer) |
| SegFace, FaceXFormer | MIT | — [SegFace](https://github.com/Kartik-3004/SegFace), [FaceXFormer](https://github.com/Kartik-3004/facexformer) |
| TRUST | non-commercial research | — [repo](https://github.com/HavenFeng/TRUST) |
| FFHQ-UV | MIT code | — [repo](https://github.com/csbhr/FFHQ-UV) |
| Arc2Face | MIT code; weights on HF (weights licence not checked) | — [repo](https://github.com/foivospar/Arc2Face) |
| FaceLift | Apache-2.0 code; **weights under Adobe Research License** | — [repo](https://github.com/weijielyu/FaceLift) |
| LAM | Apache-2.0 | — [repo](https://github.com/aigc3d/LAM) |
| GAGAvatar | MIT | — [repo](https://github.com/xg-chu/GAGAvatar) |
| MediaPipe | Apache-2.0 | — [repo](https://github.com/google-ai-edge/mediapipe) |

- **LAM** (SIGGRAPH 2025, Alibaba Tongyi): one image produces an animatable Gaussian head in one forward pass. FLAME canonical points act as transformer queries, and animation uses LBS plus corrective blendshapes. The successor MeshLAM (CVPR 2026) had its report released in April 2026. — [GitHub](https://github.com/aigc3d/LAM); [arXiv 2502.17796 (snippet)](https://arxiv.org/html/2502.17796v1)
- **GAGAvatar** (NeurIPS 2024) reconstructs controllable 3D head avatars from single images. — [GitHub](https://github.com/xg-chu/GAGAvatar)
- **Avat3r** (ICCV 2025): an animatable large reconstruction model that builds a head avatar from about **4 images**, using DUSt3R position maps and Sapiens features. It is trained with inconsistent-expression inputs, so it handles phone captures and monocular video frames, and the whole pipeline runs in a few minutes on a single consumer GPU. — [arXiv (snippet)](https://arxiv.org/abs/2502.20220); [ICCV poster](https://iccv.thecvf.com/virtual/2025/poster/2551). I did not find a code repo.
- **FaceLift** (ICCV 2025, UC Merced and Adobe): single image → multi-view diffusion → GS-LRM Gaussian head. Checkpoints download automatically from HF, and training data is not released. — [GitHub](https://github.com/weijielyu/FaceLift)
- **Arc2Avatar** (CVPR 2025): 3DGS plus score distillation guided by Arc2Face, from as little as one image. It keeps dense correspondence with the FLAME template, so it can be deformed by FLAME blendshapes. — [arXiv 2501.05379 (snippet)](https://arxiv.org/pdf/2501.05379). **ID-to-3D** (2024) is a related ID-guided SDS head method. — [arXiv 2405.16570](https://arxiv.org/pdf/2405.16570)

### Inferences
- **Suggested upgrade path, in priority order, to test against the project's render-based shrinkage metric:**
  1. Add the per-axis A-matrix calibration and multi-photo average-then-de-attenuate step (Q2). This is cheap and can be done today.
  2. Replace or augment MediaPipe constraints with dense per-vertex correspondences from Pixel3DMM (FLAME UV + normals), jointly over all photos with shared identity and per-photo expression.
  3. A/B MICA against SHeaP "expressive" as the initialiser or prior mean.
  4. Measure eyes and lips directly from FaRL, SegFace or Sapiens masks and landmarks, and add a custom eyelid-type probe.
  5. Get skin tone from the cross-photo median of TRUST or HUST albedo.
- Gaussian-avatar and diffusion methods (LAM, Avat3r, FaceLift, Arc2Avatar) are not a direct replacement, because the CK3 output must be blend-shape weights. LAM and Arc2Avatar are FLAME-anchored, but their identity detail lives mostly in Gaussians or texture, not FLAME β. FaceLift and Arc2Face-generated novel views could be fed to a multi-view FLAME fitter, but hallucinated views encode a generative prior and may reintroduce averaging.
- The Pixel3DMM install also pulls in MICA, so moving to Pixel3DMM does not remove the MICA dependency. MICA would serve only as initialisation, which the dense fitting then corrects.

### Gaps
- Per-method VRAM and runtime figures on consumer GPUs were not documented in most READMEs; Pixel3DMM's README gives none.
- I could not verify code availability for Avat3r, TokenFace, FlowFace, RealDenseFace or HUST.
- I did not check the HF weight licences for Arc2Face, VGGHeads and the Sapiens2 checkpoints; model cards were not reachable.
