# Photo-to-Game-Parameter Literature and Industry Practice (2017 to 2026), with Lessons for a CK3 DNA Generator

> **Verification legend (read first).** In this session WebFetch was blocked for arxiv.org, openaccess.thecvf.com, ojs.aaai.org, ar5iv, Semantic Scholar, ResearchGate, GDC Vault, epicgames.com and every other paper mirror. Only web-search snippets and public GitHub raw files could be read. Each finding is tagged:
> - **[V]**: verified in this session from a search-result snippet or a GitHub README.
> - **[R]**: recalled from prior knowledge of the paper and *not* re-verified this session. The canonical URL is still given. Treat these details (numbers especially) as needing a check before anyone relies on them.
> - **NEW** marks 2024 to 2026 work.
>
> Project context: CK3 Ruler Designer DNA (~100 morph genes, blend shapes capped at ~128 per head, decal genes, discrete assets). The renderer is non-differentiable. The project already has a landmark-based differentiable geometry replica, a neural imitator trained on ~29k renders (and exploited by the identity loss), CLIP attribute classifiers and an LLM A/B judge. Current quality is about 4-5/10.

---

## Q1. Key papers on face-to-parameter / photo-to-avatar for game character creators, and how each dealt with the non-differentiable engine

### Takeaway
The field moved through four stages: (1) per-photo gradient search through a learned "imitator" of the engine (Wolf 2017, F2P 2019); (2) feed-forward "translators" trained self-supervised with the imitator in the loop, mapping a pose-robust face embedding to parameters (Fast F2P 2020, TPAMI 2022); (3) domain-bridging, which either stylizes the photo into engine style first (AgileAvatar 2022) or synthesizes realistic-face/parameter pairs with dual GANs (SwiftAvatar 2023); (4) **NEW** large-pretrained-encoder feed-forward translators that accept any photo style or text (T2P 2023, ICE 2024, T2PE 2025, EasyCraft CVPR 2025). EasyCraft reports 87% user preference over F2P. NetEase is the only group that has publicly deployed this at scale (Justice / 逆水寒, >10M uses). Its current production approach is the feed-forward one, not per-photo optimization.

### Cited Findings

**Precursor: Unsupervised Creation of Parameterized Avatars (Wolf, Taigman, Polyak; ICCV 2017)**
- [V] Defines the "Tied Output Synthesis" problem: map an input image to a tied pair (parameter vector, engine-rendered image) so that the rendered image resembles the input. No supervision with matching input/output pairs. A GAN implements a discrepancy-based generalization bound. Applied to automatic avatar creation. — [arXiv 1704.05693](https://arxiv.org/abs/1704.05693); [CVF](https://openaccess.thecvf.com/content_iccv_2017/html/Wolf_Unsupervised_Creation_of_ICCV_2017_paper.html)
- [R] The non-differentiable graphics engine is replaced in training by a learned differentiable approximation of it. Identity is preserved with a fixed face-descriptor network ("f-constancy"). This is the conceptual ancestor of the "imitator". — [arXiv 1704.05693](https://arxiv.org/abs/1704.05693)

**F2P: Face-to-Parameter Translation for Game Character Auto-Creation (Shi, Yuan, Fan, Zou, Shi, Liu; NetEase Fuxi; ICCV 2019)**
- [V] Formulates character creation as a facial-similarity measurement plus parameter search over many physically meaningful parameters. A generative network ("imitator") imitates the engine's rendering so that the parameters can be optimized by gradient descent in a neural-style-transfer framework. Deployed in a new game and used more than 1 million times. — [CVF](https://openaccess.thecvf.com/content_ICCV_2019/html/Shi_Face-to-Parameter_Translation_for_Game_Character_Auto-Creation_ICCV_2019_paper.html); [arXiv 1909.01064](https://arxiv.org/abs/1909.01064)
- [V] Parameter dimension: 264 (male) and 310 (female). Of these, 208 are continuous (e.g. eyebrow length/width/thickness) and the rest are discrete (hairstyle, eyebrow style, beard style, lipstick style). — [arXiv 1909.01064 (search snippet)](https://arxiv.org/html/1909.01064)
- [V] The game is NetEase's MMO *Justice* (逆水寒). — [GitHub yiyuan1991/Face-to-Parameter README](https://github.com/yiyuan1991/Face-to-Parameter)
- [V] Per-photo optimization runs at ~1 Hz (per the AAAI 2020 follow-up's comparison table). — [AAAI 2020 paper via search](https://ojs.aaai.org/index.php/AAAI/article/view/5537)
- [R] The imitator is a DCGAN-like generator trained on engine renders of randomly sampled parameter vectors (on the order of 20k renders; exact count unverified). The input photo is aligned with facial landmarks before optimization. — [arXiv 1909.01064](https://arxiv.org/abs/1909.01064)
- [V] A GDC talk exists: "Face-to-Parameter Translation via Neural Network Renderer" (Tianyang Shi, NetEase Fuxi), on making the engine differentiable with a neural network. — [GDC Vault 1026581](https://www.gdcvault.com/play/1026581/Face-to-Parameter-Translation-via) (year not verified)

**Fast and Robust F2P (Shi, Zou, Yuan, Fan; AAAI 2020, pp. 1733–1740)**
- [V] Reformulates creation as *prediction*. A "facial parameter translator" maps face-recognition embeddings to parameters in a single forward pass and is trained under a self-supervised paradigm (imitator in the loop). It is "three magnitudes faster" than F2P and "shows better robustness (especially for pose changes)". It replaced F2P in *Justice*. — [GitHub FuxiCV/Face-to-Parameter-V2](https://github.com/FuxiCV/Face-to-Parameter-V2); [AAAI](https://ojs.aaai.org/index.php/AAAI/article/view/5537)
- [V] Reported numbers: LFW face-verification accuracy between photo and created character is 0.9402, against F2P 0.6977 and 3DMM-CNN 0.9235. Speed is ~10^3 Hz, against F2P ~1 Hz and 3DMM-CNN ~10^2 Hz. — [AAAI via search](https://ojs.aaai.org/index.php/AAAI/article/view/5537); [arXiv 2008.07132](https://arxiv.org/abs/2008.07132)
- [R] Training uses an identity loss in embedding space, a facial content (segmentation) loss and a "loop" consistency term: re-embedding the rendered output should reproduce the same prediction. — [arXiv 2008.07132](https://arxiv.org/abs/2008.07132)

**Neutral Face Game Character Auto-Creation via PokerFace-GAN (Shi et al., NetEase; 2020)**
- [V] Exists as arXiv 2008.07154. It targets creating *neutral-expression* game characters from photos that show expressions and poses. — [arXiv 2008.07154](https://arxiv.org/abs/2008.07154)
- [R] Uses a differentiable character renderer and adversarial training to disentangle expression and pose from identity parameters, so smiles are not baked into the neutral face. Published at ACM MM 2020 (venue unverified). — [arXiv 2008.07154](https://arxiv.org/abs/2008.07154)

**Neural Rendering for Game Character Auto-creation (Shi, Zou, Shi, Yuan; IEEE TPAMI 44(3):1489–1502, 2022)**
- [V] Journal consolidation of the line. Self-supervised learning with differentiable neural rendering and an imitator, so that characters are optimized end-to-end by gradient descent. Outputs physically meaningful parameters that users can fine-tune by hand. "Applied to two games providing over 10 million times of online services." — [BUAA PDF listing](https://levir.buaa.edu.cn/static/pdfs/2022_tianyang_shi_neural.pdf); [S-Logix summary](https://slogix.in/machine-learning/neural-rendering-for-game-character-auto-creation/)

**MeInGame: Create a Game Character Face from a Single Portrait (Lin, Yuan, Zou; NetEase Fuxi; AAAI 2021)**
- [V] Predicts facial **shape and texture** from one portrait and claims to integrate into most 3D games. Contributions: (1) low-cost facial texture acquisition, (2) a shape-transfer algorithm moving a 3DMM mesh into game mesh topology, (3) a training pipeline for game-face reconstruction networks. It also builds a race- and gender-balanced shape+texture dataset from in-the-wild images. — [arXiv 2102.02371](https://arxiv.org/abs/2102.02371); [AAAI PDF](https://cdn.aaai.org/ojs/16106/16106-13-19600-1-2-20210518.pdf)
- [V] Code builds on Basel Face Model 2009, Deep3DFaceReconstruction and a modified PyTorch3D (differentiable rasterizer). A pre-trained model ("celeba_hq_demo") is released. An "RGB 3D Face Dataset" is available under a signed license. — [GitHub FuxiCV/MeInGame](https://github.com/FuxiCV/MeInGame)
- [R] The output is a game-topology mesh plus texture, not slider values. This only works in engines that accept per-vertex shape and a custom texture. — [arXiv 2102.02371](https://arxiv.org/abs/2102.02371)

**AgileAvatar: Stylized 3D Avatar Creation via Cascaded Domain Bridging (2022)**
- [V] Self-supervised creation of stylized 3D avatars with mixed continuous and discrete parameters. Pipeline: (1) a modified portrait-stylization model translates the selfie into a *stylized avatar rendering* that serves as the target, (2) a **differentiable imitator** trained to mimic the avatar engine is used to fit parameters to that target, (3) a **cascaded relaxation-and-search** pipeline handles discrete parameters. The output is a user-editable, animatable 3D model (e.g. for personalized emoji). — [arXiv 2211.07818](https://arxiv.org/abs/2211.07818); [SIGGRAPH history](https://history.siggraph.org/?p=215758)
- [V] Motivation: earlier self-supervised methods "fall short" for stylized avatars because of a large style domain gap between photos and engine renders. — [arXiv 2211.07818](https://arxiv.org/abs/2211.07818)
- [R] Authors are ByteDance / UC Santa Cruz (not Snap). Venue is SIGGRAPH Asia 2022. Related ByteDance patents exist: "Cascaded domain bridging for image generation". — [Google Patents US12260485](https://patents.google.com/patent/US12260485)

**SwiftAvatar: Efficient Auto-Creation of Parameterized Stylized Character on Arbitrary Avatar Engines (AAAI 2023; BUPT and Douyin Vision/ByteDance)**
- [V] Unsupervised avatar-vector estimation fails because of the realistic-to-stylized domain gap. SwiftAvatar instead uses **dual-domain generators** (a realistic-face GAN and an avatar GAN sharing latent codes). **GAN inversion of engine-rendered avatar images** links the latent codes to avatar vectors, which yields synthetic pairs of (avatar vector, realistic face). Semantic augmentation increases diversity. A lightweight estimator is then trained supervised on these pairs. Evaluated on two avatar engines. — [arXiv 2301.08153](https://arxiv.org/abs/2301.08153); [AAAI](https://ojs.aaai.org/index.php/AAAI/article/view/25753); [Metaphysic explainer](https://metaphysic.ai/post/creating-better-avatars-with-a-dual-domain-approach)
- [V] The paper argues that purely supervised approaches need costly manual labelling that does not transfer across engines, because avatar-vector definitions differ per engine. — [arXiv 2301.08153 (search snippet)](https://arxiv.org/pdf/2301.08153)

**T2P: Zero-Shot Text-to-Parameter Translation for Game Character Auto-Creation (Zhao, Li, Hu, Li, Zou, Shi, Fan; NetEase Fuxi; CVPR 2023)**
- [V] Creates characters from arbitrary text without photos. Uses pre-trained CLIP and neural rendering. Continuous parameters are searched by fine-tuning a translator and discrete parameters by evolution search, in one framework. Claims to outperform SOTA text-to-3D methods objectively and subjectively. — [CVF PDF](https://openaccess.thecvf.com/content/CVPR2023/papers/Zhao_Zero-Shot_Text-to-Parameter_Translation_for_Game_Character_Auto-Creation_CVPR_2023_paper.pdf); [arXiv 2303.01311](https://arxiv.org/abs/2303.01311)
- [V] NetEase says the "text face-creation" (文字捏脸) research is applied in *Justice Mobile* (逆水寒手游). — [游戏日报 2023-03-14](https://news.yxrb.net/2023/0314/1372.html)

**NEW: T2PE: Zero-shot text-to-parameter realtime translation for game character auto-creation and identity consistency editing (Neurocomputing, 2025)**
- [V] Reports **26x faster inference** and **+3.3% CLIP score** on lower-TFLOPS devices compared with T2P. Adds an editing method that changes a character from new text while preserving its other features (identity consistency). — [ScienceDirect S0925231225004515](https://www.sciencedirect.com/science/article/abs/pii/S0925231225004515); [project page](https://tonystark-jun.github.io/T2PE/)

**NEW: ICE: Interactive 3D Game Character Editing via Dialogue (NetEase Fuxi + NUS + UQ; arXiv 2403.12667, 2024)**
- [V] Multi-round, dialogue-based character editing. An LLM Instruction Parsing Module turns dialogue into per-round edit prompts. A **Semantic-guided Low-dimension Parameter Solver (SLPS)** first localizes the parameters relevant to a fine-grained edit, then optimizes them **in a low-dimensional space "to avoid unrealistic results"**. — [arXiv 2403.12667](https://arxiv.org/html/2403.12667v3)
- [V] ICE notes that popular RPG creators expose "hundreds of adjustable parameters including bone positions and various makeup options". — [arXiv 2403.12667](https://arxiv.org/html/2403.12667v3)

**NEW: EasyCraft: A Robust and Efficient Framework for Automatic Avatar Crafting (Wang, Chen, Zhang, Zhao, Li, Zhang, Hu, Yu; NetEase Fuxi + Univ. of Queensland; CVPR 2025, arXiv 2503.01158)**
- [V] End-to-end **feed-forward** framework that accepts both images and text. Step 1: self-supervised pretraining of a universal **ViT encoder** on a large dataset of photos in *various styles*. Step 2: an **engine-specific translator** (the frozen ViT encoder plus a trainable parameter-generation module) converts facial images of *any style* into crafting parameters. The self-supervised encoder gives photos and other styles a unified feature distribution. Text input is supported as well. — [arXiv 2503.01158](https://arxiv.org/abs/2503.01158); [CVF PDF](https://openaccess.thecvf.com/content/CVPR2025/papers/Wang_EasyCraft_A_Robust_and_Efficient_Framework_for_Automatic_Avatar_Crafting_CVPR_2025_paper.pdf)
- [V] Tested on **two RPG games** with SOTA results claimed. In a user study **87% of users preferred EasyCraft vs 11% for F2P**. — [arXiv 2503.01158 (search snippet)](https://arxiv.org/abs/2503.01158)
- [V] NetEase Fuxi announced it as its CVPR 2025 "前馈捏脸" (feed-forward face-crafting) paper. — [Fuxi news](https://fuxi.163.com/database/1633); [InfoQ mirror](https://xie.infoq.cn/article/a3ede452806b272fd048dbfca)

**Adjacent "photo to avatar" lines that do not output sliders (useful as components)**
- [V] AvatarMe (CVPR 2020) and AvatarMe++ (TPAMI 2021), Imperial College: single-image reconstruction of photorealistic renderable 3D faces with diffuse/specular BRDF, via 3DMM fitting and rendering-aware GANs. Follow-ups are FitMe (CVPR 2023) and Relightify (ICCV 2023). These produce meshes and texture maps, not game parameters. — [GitHub lattas/AvatarMe](https://github.com/lattas/AvatarMe)
- [V] **NEW** Text2Avatar (arXiv 2401.00711, Jan 2024): text to realistic 3D human avatars. A discrete codebook is the intermediate representation that disentangles attributes. Pseudo-data comes from a pretrained unconditional 3D avatar generator to make up for scarce 3D data. — [arXiv 2401.00711](https://arxiv.org/abs/2401.00711v1)
- [V] **NEW** CharGen (arXiv 2509.25058, 2025): attribute-specific Concept Sliders (LoRA) plus StreamDiffusion for interactive portrait editing. This is image-space only, not engine parameters. — [arXiv 2509.25058](https://arxiv.org/html/2509.25058)
- [V] AvatarizeMe (STAG 2023, Eurographics): fast tool that turns selfies into animatable lifelike avatars using ML. — [EG Digital Library](https://diglib.eg.org/items/b4e01093-dc84-42e3-9c96-e87724463644/full)

### Inferences
- **What each approach did about the non-differentiable engine:**
  - learned imitator plus per-photo gradient descent (Wolf 2017, F2P 2019);
  - imitator used only at *training* time, with a feed-forward translator at inference (Fast F2P, TPAMI 2022, T2PE);
  - imitator plus relaxation-and-search after stylizing the photo into engine style (AgileAvatar);
  - **no imitator in the loss at all**: GAN inversion of real engine renders to build supervised pairs (SwiftAvatar);
  - a frozen, style-agnostic SSL encoder plus a parameter head trained per engine (EasyCraft; its exact training signal was not verified);
  - evolutionary/black-box search for the discrete part (T2P).
  
  Nobody in this literature uses RL. The "search" components are evolutionary/discrete only. Note that SwiftAvatar's and EasyCraft's inference paths never backpropagate image-space identity losses through an imitator.
- **Main lesson for CK3:** the project's ~29k (DNA, render) pairs are exactly the asset that SwiftAvatar and EasyCraft-style pipelines exploit. Train an **inverse network render→DNA with a loss in parameter space** (plus a geometry loss through the existing differentiable head replica). Put it on top of a **frozen self-supervised encoder** (EasyCraft idea; DINOv2/MAE-class or face-recognition backbones are candidates) so that real photos and CK3 renders share a feature space. This is structurally immune to the imitator-exploitation problem, because the network is never rewarded for fooling an identity net. Imitator or identity losses can then be added as a small fine-tuning term with parameter-prior regularization.
- **Bridging the domain gap:** use either AgileAvatar's "stylize first" step (translate the photo into a CK3-render-looking image, e.g. img2img with a LoRA trained on the 29k renders, then apply the render→DNA network) or SwiftAvatar's dual-StyleGAN pairing. A StyleGAN2 at 256–512 px can be fine-tuned from FFHQ on 29k renders on one RTX 5080, with latent correspondence kept via shared/fine-tuned weights. That synthesizes realistic faces for any DNA and so creates (photo, DNA) training pairs.
- **Pose-robust embeddings** (Fast F2P's key gain: LFW 0.70 → 0.94) suggest that the input to the translator should be identity/shape features that do not change with pose. Raw pixels or landmarks from a single posed photo are a worse input.
- MeInGame-style geometry transfer is the closest analogue to the project's existing differentiable head replica. The literature suggests reconstructing a full 3D face first and then fitting morph weights in 3D, rather than fitting 2D landmarks directly. Identity-aware shape regressors such as MICA (not researched this session) are candidates for that first step.

### Gaps
- Full texts could not be read, so the following were not verified: exact imitator architectures, render counts, loss weights, and quantitative tables beyond those quoted for F2P, Fast F2P, PokerFace-GAN, AgileAvatar and SwiftAvatar.
- EasyCraft's exact training signal for the parameter head (supervised on engine renders? imitator loss?) and the identities of its two RPG games were not verified.
- No 2025–2026 academic paper from groups other than NetEase/ByteDance on *realistic* game sliders was found. The search budget was exhausted before diffusion-based "image-to-slider" works could be probed further.
- No paper using RL for face-to-parameter was found. This may simply reflect the limited search.

---

## Q2. Which methods handle discrete parameters (hairstyles, accessories) together with continuous sliders, and how?

### Takeaway
Three techniques recur. (a) Continuous **relaxation**: one-hot to softmax, optimized jointly through an imitator (F2P 2019). (b) **Relaxation followed by discrete search** in a cascade (AgileAvatar 2022). (c) **Evolutionary search** for the discrete part alternating with gradient or translator updates for the continuous part (T2P 2023). Feed-forward estimators (SwiftAvatar, EasyCraft) can predict discrete choices as classification heads trained on paired data. T2P's "first to handle both" claim is contradicted by F2P and AgileAvatar.

### Cited Findings
- [V] F2P encodes discrete parameters (hairstyle, eyebrow style, beard style, lipstick style) as one-hot vectors concatenated with the 208 continuous ones. Because one-hot codes are hard to optimize, it **smooths them with softmax** and optimizes everything through the imitator. — [arXiv 1909.01064 (search snippet)](https://arxiv.org/html/1909.01064)
- [V] AgileAvatar: the avatar vector mixes continuous (e.g. head length) and discrete (e.g. hair type) parameters over predefined 3D assets. A **cascaded relaxation-and-search** pipeline ensures effective optimization of the discrete parameters. — [arXiv 2211.07818](https://arxiv.org/abs/2211.07818)
- [V] T2P searches continuous facial parameters by fine-tuning a translator and **discrete parameters by evolution search**, scored with CLIP. It claims to be "the first method that can handle the optimization of both discrete and continuous parameters". — [CVF CVPR 2023](https://openaccess.thecvf.com/content/CVPR2023/papers/Zhao_Zero-Shot_Text-to-Parameter_Translation_for_Game_Character_Auto-Creation_CVPR_2023_paper.pdf); [ScienceDirect T2PE summary](https://www.sciencedirect.com/science/article/abs/pii/S0925231225004515). **This claim is contradicted** by F2P 2019 (softmax relaxation) and AgileAvatar 2022 (relaxation-and-search), both above.
- [V] SwiftAvatar's estimator outputs whole "avatar vectors" for arbitrary engines, trained on synthesized pairs. The snippets do not confirm how it splits discrete and continuous outputs. — [arXiv 2301.08153](https://arxiv.org/abs/2301.08153)
- [V] NetEase Fuxi describes production parameter systems as many continuous sliders per facial region (e.g. inner/middle/outer eyebrow position, thickness, angle, colour) plus discrete options (eyebrow shapes, makeup). Top mobile games have "大几百个" (several hundred) facial parameters. — [GameLook, Dec 2025, Fuxi talk at China Game Industry Annual Conference](http://www.gamelook.com.cn/2025/12/584761/); [Fuxi mirror](https://fuxi.163.com/database/3001)
- [V] Tencent's *Moonlight Blade Mobile* (天涯明月刀手游) exposes over 200 adjustable parameters on 48 facial bones. This illustrates the scale of continuous-only spaces in Chinese MMOs. — [17173](https://news.17173.com/content/04042019/103853310.shtml)

### Inferences
- For CK3 the discrete set (hair, beard, eyebrow asset, eye colour, possibly ethnicity-template decals) is small enough to **enumerate**. Following the AgileAvatar/T2P pattern, a robust recipe is:
  1. predict the top-k discrete choices with classifiers (CLIP attribute heads already exist; or nearest-neighbour retrieval of the photo embedding against renders of each asset);
  2. for each candidate combination, fit or predict the continuous morph genes conditioned on it;
  3. pick the best combination with a held-out scorer.
  
  Softmax relaxation through an imitator is the riskiest option, because the imitator may render "blends" of hairstyles that the engine cannot produce.
- Discrete assets mostly affect hair and silhouette, while identity mostly lives in continuous shape. So fit continuous genes with hair masked out, or with a hair-agnostic face crop, to stop hair choice from contaminating the shape fit.
- A feed-forward model can output both: a regression head for the ~100 morph genes and decal intensities, plus softmax classification heads for each asset slot, trained on the 29k (DNA, render) pairs, where the labels are known exactly.

### Gaps
- Exact evolutionary-search settings in T2P and the relaxation schedule in AgileAvatar (temperatures, number of cascade stages) were not verified.
- No source covers handling a hard engine cap like CK3's ~128 active blend shapes. Sparsity-constrained fitting (L1 / top-k on gene activations) is an inference, not a literature finding.

---

## Q3. Loss functions / likeness measures, and reported failure modes (adversarial exploitation, "average face")

### Takeaway
The canonical likeness objective since F2P combines a **global identity loss** (a face-recognition embedding such as LightCNN-29v2) with a **local "facial content" loss** computed on features of a face-parsing (segmentation) network. Text and multi-style methods replace or add **CLIP** similarity. Evaluation uses face-verification accuracy between photo and render (LFW protocol) and pairwise user-preference studies. The documented failure modes are:
- domain gap between photos and engine renders, which breaks self-supervised fitting on stylized engines;
- pose and expression leakage, which motivated Fast F2P and PokerFace-GAN;
- unrealistic parameter combinations when optimizing in the full space, which motivated ICE's low-dimensional solver.

None of the sources read explicitly documents adversarial exploitation of the identity network, but the project's own experience matches the motivation for the feed-forward and paired-data designs.

### Cited Findings
- [V] F2P uses two losses: a **"discriminative loss"** for global facial appearance and a **"facial content loss"** for local details. The identity network is **LightCNN-29v2**, giving 256-d embeddings. — [arXiv 1909.01064 (search snippet)](https://arxiv.org/html/1909.01064)
- [R] F2P's facial content loss compares feature maps of a face semantic-segmentation network (ResNet-based, trained on face-parsing data), weighted by part probability maps. Its ablation shows that each loss alone is worse than the combination. — [arXiv 1909.01064](https://arxiv.org/abs/1909.01064)
- [V] Fast F2P's metric is LFW-style face verification between the photo and the created character: **0.9402** for Fast F2P vs **0.6977** for F2P vs 0.9235 for 3DMM-CNN. The per-photo optimization method (F2P) scores markedly worse on identity verification than the embedding-driven translator. — [AAAI 2020 via search](https://ojs.aaai.org/index.php/AAAI/article/view/5537)
- [V] Fast F2P was explicitly motivated by robustness to **pose changes**. — [GitHub FuxiCV/Face-to-Parameter-V2](https://github.com/FuxiCV/Face-to-Parameter-V2)
- [V] PokerFace-GAN addresses **expression/pose leakage**: photos with expressions should still yield neutral-face characters. — [arXiv 2008.07154](https://arxiv.org/abs/2008.07154)
- [V] SwiftAvatar: unsupervised estimation methods "often fail because of the domain gap between realistic faces and stylized avatar images". — [arXiv 2301.08153](https://arxiv.org/abs/2301.08153)
- [V] AgileAvatar: self-supervised approaches fall short on stylized avatars because of the large style domain gap. It counters this by stylizing the selfie first, so the imitator-based loss compares like with like. — [arXiv 2211.07818](https://arxiv.org/abs/2211.07818)
- [V] ICE's SLPS optimizes only the localized, relevant parameters in a **low-dimensional space to avoid unrealistic results**, an explicit acknowledgment that unconstrained optimization in the full slider space gives implausible faces. — [arXiv 2403.12667](https://arxiv.org/html/2403.12667v3)
- [V] T2P and T2PE measure text-to-character likeness with **CLIP score** (T2PE: +3.3%) and subjective studies. — [ScienceDirect T2PE](https://www.sciencedirect.com/science/article/abs/pii/S0925231225004515)
- [V] EasyCraft evaluates by user preference (**87% vs 11% against F2P**). — [arXiv 2503.01158](https://arxiv.org/abs/2503.01158)
- [V] MeInGame works in geometry and texture space (3DMM fitting plus differentiable rendering with PyTorch3D) rather than through an imitator of the game renderer. — [GitHub FuxiCV/MeInGame](https://github.com/FuxiCV/MeInGame)

### Inferences
- **The project's imitator exploitation is the expected outcome of the F2P recipe** when the identity term dominates and the imitator is queried off its training distribution. The literature's countermeasures, in order of how directly they apply to CK3:
  1. **Change the training signal.** Supervise in parameter space on real (DNA, render) pairs and on synthetic (photo, DNA) pairs (SwiftAvatar, EasyCraft-style), so there is no adversarial path at all.
  2. **Pair identity with a part-wise content loss** (F2P's segmentation-feature loss) and landmark/3D geometry losses through the *exact* differentiable geometry replica, which cannot be fooled the way an imitator can. Keep the identity weight small.
  3. **Constrain the parameter space** to a low-dimensional plausible manifold (ICE's SLPS). For CK3, a PCA/VAE fitted on community presets or on the game's own ethnicity templates is one option.
  4. **Close the imitator loop.** Periodically render the optimizer's outputs in the real game and add them to imitator training (DAgger-style hard-negative mining; this is an inference, not from the sources). Use an imitator ensemble and penalize its disagreement as an uncertainty term.
  5. **Score with a different recognizer than the one optimized** (for example, train on LightCNN/ArcFace but evaluate with another network and the LLM judge), so that exploitation shows up in evaluation.
- **On "average face" output:** none of the sources read reported regression to the mean explicitly. However, feed-forward regressors trained with an L2 parameter loss are known to shrink towards the dataset mean. Fast F2P's identity/loop losses and EasyCraft's user-study gains suggest that identity-aware losses (or per-parameter classification over binned values) are needed in addition to parameter regression.
- **Adopt a Fast-F2P-style automatic metric.** Report face-verification TAR/accuracy between the photo and the actual in-game render with a held-out recognizer, alongside the GPT/Claude judge. This makes progress measurable without the judge's variance.
- **Neutralize expression and normalize pose before the shape fit** (PokerFace-GAN). Using an expression-free 3DMM identity shape as the target directly addresses the "smile baked into morphs" failure.

### Gaps
- No source read in this session explicitly reports identity-network adversarial exploitation or "average face" failures for these game methods. Both are inferred from general knowledge and from the project's own experience.
- Exact loss weights, and whether EasyCraft uses identity/perceptual losses at all, were not verified.

---

## Q4. What commercial games and tools do in practice, and public talks on their pipelines

### Takeaway
Two industrial families exist. (1) **Parameter-space photo-to-slider** for slider-based MMO character creators is publicly documented only by **NetEase** (Justice PC / 逆水寒手游 since ~2018–2019, >10M uses, later text and voice input). (2) **Mesh/scan-space likeness** comes from sports games and digital-human tools: EA Game Face (FIFA 12–16, photo upload), NBA 2K Companion-app face scan (multi-view phone capture), EA FC 27 "PhotoFit" (leaked, 2026), and Epic MetaHuman (Mesh to MetaHuman / identity solve from footage = template-mesh fitting). These produce meshes and textures, not slider values, so they are less transferable to CK3. Roblox's Photo-to-Avatar (alpha) generates a **2D stylized preview first** and the 3D avatar second. That is the same "stylize first, then fit" idea as AgileAvatar. No public source was found for Black Desert, The Sims, Ubisoft, miHoYo or a Tencent photo-to-slider feature.

### Cited Findings
**NetEase (Justice / 逆水寒, Justice Mobile)**
- [V] F2P was deployed in *Justice* and then replaced by Fast F2P. The TPAMI version reports two games and >10M online uses. — [GitHub Face-to-Parameter-V2](https://github.com/FuxiCV/Face-to-Parameter-V2); [TPAMI 2022 PDF](https://levir.buaa.edu.cn/static/pdfs/2022_tianyang_shi_neural.pdf)
- [V] Fuxi has developed image face-crafting algorithms since **2018**: one input image yields "stable" in-game parameters that look very similar to the reference. — [GameLook Dec 2025 (Fuxi vision lead Li Lincheng's talk at the 2025 China Game Industry Annual Conference)](http://www.gamelook.com.cn/2025/12/584761/); [Toutiao](https://www.toutiao.com/zixun/7494839603103451147/)
- [V] *Justice Mobile* features include text face-crafting (players type classical-poetry or wuxia descriptions) and, from Dec 2025, "voice face-crafting" (声音捏脸), where voice features become modelling parameters. — [QQ News 2025-12-03](https://news.qq.com/rain/a/20251203A083ZF00); [游戏日报](https://news.yxrb.net/2023/0314/1372.html)
- [V] Press coverage: SCMP, "NetEase can use AI to turn you into a game avatar with one selfie" (2019). Synced Review: "Chinese Gaming Giant NetEase Leverages AI to Create 3D Game Characters from Selfies". TechXplore covered MeInGame (Mar 2021). — [SCMP](https://www.scmp.com/abacus/games/article/3041702/netease-can-use-ai-turn-you-game-avatar-one-selfie); [Synced/Medium](https://medium.com/syncedreview/chinese-gaming-giant-netease-leverages-ai-to-create-3d-game-characters-from-selfies-487f8de0f0c7); [TechXplore](https://techxplore.com/news/2021-03-meingame-deep-method-videogame-characters.html)
- [V] GDC talk: "Face-to-Parameter Translation via Neural Network Renderer" (Tianyang Shi, NetEase Fuxi). — [GDC Vault](https://www.gdcvault.com/play/1026581/Face-to-Parameter-Translation-via)

**EA**
- [V] EA SPORTS Game Face: web upload of photos (front/side) to build a 3D head, which users can then tweak (nose, eyes, mouth, hair, facial hair). It first appeared around FIFA 12 and was dropped after FIFA 16. — [TouchArcade](https://toucharcade.com/games/ea-sports-game-face-3d-avatar); [Dexerto](https://www.dexerto.com/ea-sports-fc/ea-fc-27-leak-reveals-you-can-scan-yourself-into-the-game-for-first-time-in-a-decade-3394992/)
- [V] **NEW (2026, leak/rumor, unconfirmed by EA)** EA FC 27 "PhotoFit": scan your face with a phone (via a barcode/app flow) to generate a matching 3D head. The scan is a starting point that remains editable. — [Dexerto](https://www.dexerto.com/ea-sports-fc/ea-fc-27-leak-reveals-you-can-scan-yourself-into-the-game-for-first-time-in-a-decade-3394992/); [Players.com.ua](https://players.com.ua/en/news/ea-fc-27-leak-scanning-own-faces-to-return-to-the-game-for-the-first-time-in-10-years/)
- [V] **NEW (GDC 2024)** EA SEED, "From Photo to Expression: Generating Photorealistic Facial Rigs" (Hau Nghiep Phan, ML Summit). Covers FaceRig (EA's in-house rig), FaceMixer (generates blendshapes from a neutral face), FaceOptim (blendshapes/animation from scans or 4D) and **FaceBot** (generates photoreal face textures and meshes with expressions that feed the other tools). — [GDC Vault 1034555](https://gdcvault.com/play/1034555/); [EA SEED blog](https://www.ea.com/seed/news/generating-photorealistic-facial-rigs)

**2K (NBA 2K)**
- [V] The NBA 2K26 face scan runs through the NBA 2K Companion App: **13 captures** while slowly rotating the head (~30°, at most 45°), in even natural light, with the phone ~18" away. The app produces a 3D face model for MyPLAYER. — [Destructoid](https://www.destructoid.com/how-to-scan-your-face-for-nba-2k26-myplayer/); [Yahoo Tech](https://tech.yahoo.com/gaming/articles/scan-face-nba-2k26-myplayer-185527093.html)

**Epic MetaHuman**
- [V] Mesh to MetaHuman / footage workflow: create a MetaHuman Identity, track frames, then run an **Identity Solve** ("fitting the Template Mesh topology onto the target mesh volume"). The template is submitted to the MetaHuman backend, and the facial rig is generated with the "From Identity" tool. In **UE 5.6** this happens in the in-engine MetaHuman Creator via "Conform from Identity". Footage requires compatible capture devices. — [Epic docs: Mesh to MetaHuman](https://dev.epicgames.com/documentation/metahuman/mesh-to-metahuman); [From video footage](https://dev.epicgames.com/documentation/metahuman/from-video-footage); [Identity asset](https://dev.epicgames.com/documentation/metahuman/metahuman-identity-asset)

**Roblox**
- [V] **NEW** Photo-to-Avatar (alpha) API: `AvatarCreationService:RequestAvatarGenerationSessionAsync()`, then a selfie prompt (`PromptSelectAvatarGenerationImageAsync`), then `GenerateAvatar2DPreviewAsync` (**a 2D preview avatar image**), then a full 3D avatar built via HumanoidDescription. Sessions limit previews and "Allowed3DGenerations". The user also supplies a text prompt. — [Roblox Creator Docs: avatar generation](https://create.roblox.com/docs/avatar/avatar-generation)

**Ready Player Me**
- [V] Selfie-to-avatar is described (in a secondary article) as an ML model producing a 3D model from one photo using a 3DMM approach. RPM avatars ship with a blend-shape facial rig including visemes. — [80.lv](https://80.lv/articles/creating-3d-avatars-from-a-single-selfie/); [RPM docs: morph targets](https://docs.readyplayer.me/ready-player-me/~/changes/L1uyUyvU6GAHHzgm2k6O/api-reference/avatars/morph-targets/oculus-ovr-libsync). *Caveat: the search summary may have conflated RPM with the AvatarizeMe paper. Treat the 3DMM attribution as unverified.*

**Krafton inZOI**
- [V] Create a Zoi offers 250+ options, numeric sliders plus a clay-like Sculpt mode, and local AI texture generation from prompts. A "3D Printer" turns photos into *objects/accessories*. **No photo-to-face feature was confirmed.** The claim that the face model is "built on a deep learning algorithm trained on thousands of real human morphologies" comes from a low-authority guide site. — [PlayNews guide](https://www.playnews.gg/en/guides/inzoi-2026-mastering-create-a-zoi-and-generative-ai); [inZOI wiki: Creative Studio](https://thegameswiki.com/inzoi/wiki/creative-studio)

**Tencent / Naraka**
- [V] *Moonlight Blade Mobile*: 200+ parameters on 48 bones, with no photo AI found. *Naraka: Bladepoint* (永劫无间, NetEase/24 Entertainment) is known for its deep face-crafting (bone move/rotate/scale plus makeup, with presets), but the search found **no authoritative source for a photo-to-face AI** in Naraka. — [17173](https://news.17173.com/content/04042019/103853310.shtml); [Toutiao on Naraka](https://www.toutiao.com/zixun/7534536124174419995)

### Inferences
- The only shipped system matching CK3's situation (fixed slider/bone parameter space plus discrete assets, with the engine untouched) is NetEase's. Its production arc went from per-photo optimization to a feed-forward translator, and by 2025 to a feed-forward model on a self-supervised universal encoder (EasyCraft). That supports the project's plan for a pre-trained few-second model over per-photo iteration.
- Sports games achieve recognizability by **capturing more data** (13 phone views in NBA 2K; front and side photos in Game Face) and writing **mesh and texture** directly. CK3 cannot accept free meshes. The transferable part is to **use multiple photos when available** and fuse them into one 3D identity estimate before fitting DNA. The project brief already allows "one or a few photos".
- MetaHuman's identity solve is template-mesh fitting. Fitting CK3's morph basis to a reconstructed 3D face is the same operation in a much lower-dimensional basis, and the project's differentiable geometry replica is the right tool for it.
- Roblox's 2D-preview-first flow and AgileAvatar's stylization step are the same pattern: generate a target image in the engine's own style, then solve parameters against that target. For CK3, a CK3-style img2img generator (trained on the 29k renders) followed by a render→DNA inverse network is an industry-validated architecture.

### Gaps
- No public technical talk was found on how NBA 2K, EA Game Face/PhotoFit, Ready Player Me or Roblox map photos to their internal parameters (blend shapes vs free mesh).
- **No information was found** on photo-to-character features or pipelines at Black Desert (Pearl Abyss), The Sims (EA Maxis), Ubisoft, miHoYo/HoYoverse, or a Tencent photo-to-slider product. Apple Memoji, Snap Bitmoji and Meta Avatars selfie-suggestion pipelines could not be researched because the search budget ran out.
- The GDC year of Tianyang Shi's talk and its content are unverified.
- [R, unverified] Ready Player Me's corporate status (reports of an acquisition and a developer-service shutdown around late 2025 / early 2026) could not be checked.

---

## Q5. Work that learns from user-created avatars/presets (human photo↔avatar pairs) or from user selections

### Takeaway
Very little published work trains on human-made photo↔avatar pairs. The main argument against it is that **human annotators are inconsistent and every engine needs new labels**: SwiftAvatar says so explicitly, and the "Tag-based annotation creates better avatars" paper (2023) is built on this premise and proposes tagging instead of direct slider labelling. Production systems instead learn from engine renders (self-supervised or synthetic pairs). User feedback enters as **interactive editing** (ICE's multi-round dialogue, T2PE's identity-consistent text edits) and as editable outputs, not as training signal in any published work found.

### Cited Findings
- [V] SwiftAvatar argues that supervised photo→avatar-vector learning requires "large amounts of labeled data collection and manual labeling", which is "laborious, expensive", and not generalizable across engines because avatar-vector definitions differ per engine. Hence it synthesizes pairs. — [arXiv 2301.08153](https://arxiv.org/pdf/2301.08153)
- [V] "Tag-based annotation creates better avatars" (arXiv 2302.07354, 2023) appears in searches on avatar annotation. — [arXiv 2302.07354](https://arxiv.org/pdf/2302.07354)
  - [R] Its abstract states that renderers such as Bitmoji, MetaHuman and Google Cartoonset expose 20+ parameters, some with hundreds of options, so annotators labelling pairwise photo→avatar data have the same difficulty as users. This causes high label noise, and each new renderer or version needs thousands of new pairs. Annotating **tags** (attributes) instead gives higher annotator agreement, more consistent model predictions, and cheap extension to new renderers. — [arXiv 2302.07354](https://arxiv.org/abs/2302.07354)
- [V] AgileAvatar emphasizes that manual avatar creation (selecting and adjusting continuous and discrete parameters) is laborious and time-consuming for average users. Its output is user-editable afterwards. — [arXiv 2211.07818](https://arxiv.org/abs/2211.07818)
- [V] ICE: users refine the character over multiple rounds of dialogue, with an LLM parsing intent and SLPS editing only the relevant parameters. — [arXiv 2403.12667](https://arxiv.org/html/2403.12667v3)
- [V] T2PE: editing with new text while preserving other features. — [ScienceDirect](https://www.sciencedirect.com/science/article/abs/pii/S0925231225004515)
- [V] F2P/TPAMI: outputs are physically meaningful parameters that users can manually fine-tune. — [TPAMI 2022 summary](https://slogix.in/machine-learning/neural-rendering-for-game-character-auto-creation/)
- [V] NetEase Fuxi's Justice Mobile lets players craft from text, and later voice, as alternative inputs to photos. No user-selection learning loop is described. — [QQ News](https://news.qq.com/rain/a/20251203A083ZF00)
- [V] WildAvatar (CVPR 2025) is a web-scale (10,000+ subjects) in-the-wild dataset for 3D human avatar creation from YouTube. It is relevant for diverse real-face training data, not for parameter labels. — [WildAvatar](https://wildavatar.github.io/)

### Inferences
- **Community CK3 DNA presets of real people are valuable, but mainly as a prior and an evaluation set, not as primary supervision.** Following the label-noise argument above:
  1. fit a low-dimensional prior (PCA/VAE) over preset DNAs and use it to regularize the solver or translator (ICE-style low-dimensional optimization);
  2. use (photo, preset) pairs as a *validation* benchmark that measures "community-preset-maker level";
  3. if they are used for fine-tuning, use them as soft targets with low weight, or convert them into *tag/attribute* targets (e.g. "nose: long/hooked") rather than raw gene values.
- Rather than collect raw DNA labels, the project could follow the tag-based idea: have the LLM judge or CLIP produce **attribute tags** for photos (jaw width, nose bridge shape, eye spacing) and train or condition the DNA predictor on those tags. The project's CLIP attribute classifiers already move in this direction.
- A cheap way to learn from user selections (an inference, not found in the literature): show N candidate DNAs (top-k discrete combinations × samples from the posterior) and log which one the user picks. Over time these preference pairs can train a reward model or re-rank the translator's outputs. ICE's multi-round editing is the closest published analogue.

### Gaps
- No published paper was found that trains a photo→game-slider model on **player-created presets** (e.g. NetEase using millions of player-made faces from Justice). Whether NetEase does this internally is unknown.
- The details of "Tag-based annotation creates better avatars" (authors, venue, numbers) were not verified because the fetch was blocked. The summary above is [R].
- No source on preference learning or RLHF from user selections for character creators was found within the search budget.
