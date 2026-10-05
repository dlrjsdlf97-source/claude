# Hair reconstruction, hair-card generation and hairstyle matching for photo-to-CK3 characters

Research conditions (read first): searched 2026-10-05. arxiv.org, huggingface.co, openaccess.thecvf.com, *.is.tue.mpg.de, docs.blender.org, dev.epicgames.com, 80.lv, cgchannel.com, opengameart.org and the MakeHuman site were all blocked by the egress proxy. github.com pages *could* be fetched, so code licences below were checked on GitHub where possible. Everything else comes from search-result snippets (titles, abstracts, venue listings). Details that come only from a snippet are marked "(snippet)". Details I could not verify are marked "(unverified)". Items from 2025–2026 are marked **[NEW 2025]** / **[NEW 2026]**.

---

## Q1. Single-view hair modelling/reconstruction: key papers, do they output strands usable for card conversion, code/weights, licences?

### Takeaway
The field has moved from database retrieval (2015–16) to volumetric CNN/GAN (2018–19), implicit fields (2022), intermediate strand/depth maps (2023), and now diffusion or transformer strand generators trained on large synthetic datasets (DiffLocks 40K hairstyles, CVPR 2025; Im2Haircut, ICCV 2025; HairLRM, SIGGRAPH 2026). Most modern methods output ~10K–100K 3D polyline strands (Alembic/.hair/.data/.blend), which can be turned into cards. However, almost every released codebase or set of weights is **non-commercial/research-only** (DiffLocks, HairStep, GaussianHaircut, HAAR). Perm's *code* is MIT, but its single-view pipeline is not openly released. Curly/kinky hair is still a weak point for everything except DiffLocks.

### Cited Findings
**Database/retrieval era**
- Hu, Ma, Luo, Li, "Single-view hair modeling using a hairstyle database", ACM TOG 34(4) 125, SIGGRAPH 2015. Given a photo plus user strokes, it searches a 3D hairstyle database, combines several matching examples into one hairstyle, then synthesises strands by optimising 2D similarity, physical plausibility and local orientation coherency. — [SIGGRAPH history archive](https://history.siggraph.org/?p=112766)
- "AutoHair: fully automatic hair modeling from a single image" (ACM TOG 2016) was the first fully automatic single-image method. It uses a hierarchical DNN for hair segmentation and **growth-direction estimation**, plus data-driven hair matching against a large set of 3D exemplars. Run over Internet photos, it produced a database of ~50K 3D hair models and supports "hair-aware image retrieval". — [SIGGRAPH history archive](https://history.siggraph.org/?p=102839); [White Rose eprint](https://eprints.whiterose.ac.uk/134268/)

**CNN/GAN volumetric era**
- "HairNet: Single-View Hair Reconstruction using Convolutional Neural Networks" (arXiv 1806.07467; USC VGL project page). — [USC VGL](https://vgl.ict.usc.edu/Research/HairNet/); [alphaxiv](https://alphaxiv.org/abs/1806.07467) (no further details verified)
- Saito et al. 2018, "3D hair synthesis using volumetric variational autoencoders", is patented (US 11074751). A small unofficial re-implementation (volumetric VAE plus an image-to-latent CNN) exists on GitHub. — [patent PDF](https://image-ppubs.uspto.gov/dirsearch-public/print/downloadPdf/11074751); [DefneGulmez/single-view-3d-hair-reconstruction](https://github.com/DefneGulmez/single-view-3d-hair-reconstruction)
- Hair-GANs (Meng Zhang et al., arXiv 1811.06229, published 2019) maps 2D orientation maps plus a bust depth map to a 3D volumetric occupancy+orientation field, which is then used to guide strand synthesis. The network code is at github.com/MengZephyr/HairGANs. — [arXiv PDF](https://arxiv.org/pdf/1811.06229); [GitHub](https://github.com/MengZephyr/HairGANs)

**Implicit / intermediate-representation era**
- NeuralHDHair (Wu, Ye, Yang, Fu, Zhou, Zheng; CVPR 2022) uses an IRHairNet to infer 3D geometric features and a GrowingNet to grow strands in parallel. HairStep (2023) reported that the official code had not been released and re-implemented it as "NeuralHDHair*". — [CVF open access](https://openaccess.thecvf.com/content/CVPR2022/html/Wu_NeuralHDHair_Automatic_High-Fidelity_Hair_Modeling_From_a_Single_Image_Using_CVPR_2022_paper.html); [HairStep arXiv](https://arxiv.org/pdf/2303.02700)
- HairStep (CVPR 2023 Highlight) uses a strand map plus a depth map as its intermediate representation. Its HiSa/HiDa dataset has 1,250 annotated portraits, and it provides a full single-image → 3D hair pipeline that relies on SAM and 3DDFA_V2. **Licence:** CC BY-NC 4.0. The "HiSa & HiDa dataset and pre-trained checkpoints … available for non-commercial research purposes only." A separate differentiable strand renderer (orientation/depth maps from 3D strands) was also released. — [GitHub README](https://github.com/GAP-LAB-CUHK-SZ/HairStep); [LICENSE](https://github.com/GAP-LAB-CUHK-SZ/HairStep/blob/main/LICENSE)

**Video / multi-view strand reconstruction (not single image, but relevant as offline asset tools)**
- Neural Haircut (Sklyarova et al., ICCV 2023, Samsung Labs) does prior-guided strand reconstruction. Its strand prior is reused by HAAR. Licence not verified (the GitHub URL returned 404 to the fetcher). — [CVF supplemental](https://openaccess.thecvf.com/content/ICCV2023/supplemental/Sklyarova_Neural_Haircut_Prior-Guided_ICCV_2023_supplemental.pdf)
- MonoHair (CVPR 2024 oral) reconstructs from a monocular video in two stages: an exterior pass (Patch-based Multi-View Optimization) and interior inference. It outputs connected strands in `.hair` files, was tested on an RTX 3090Ti with CUDA 11.3, and uses Instant-NGP and tiny-cuda-nn. The repo includes a LICENSE.txt, but I could not read its terms. — [CVF](https://openaccess.thecvf.com/content/CVPR2024/html/Wu_MonoHair_High-Fidelity_Hair_Modeling_from_a_Monocular_Video_CVPR_2024_paper.html); [GitHub](https://github.com/KeyuWu-CS/MonoHair)
- GaussianHaircut (ECCV 2024) represents hair as strand-aligned 3D Gaussians fitted from a monocular video and takes "a few hours on a single GPU". **Licence:** CC BY-NC-SA 4.0, plus the 3DGS licence for the parts derived from it. It uses Blender 3.6 for the final visualisation. — [ECCV paper page](https://ecva.net/papers/eccv_2024/papers_ECCV/html/2505_ECCV_2024_paper.php); [GitHub](https://github.com/eth-ait/GaussianHaircut)
- GaussianHair (arXiv 2402.10483) models strands as connected cylindrical Gaussians, captured from handheld smartphone video. HairGS (arXiv 2509.07774, **[NEW 2025]**, BMVC 2025 archive) reconstructs multi-view strands and typically finishes within one hour. — [arXiv 2402.10483](https://arxiv.org/pdf/2402.10483); [HairGS arXiv](https://arxiv.org/abs/2509.07774); [BMVC 2025 PDF](https://bmva-archive.org.uk/bmvc/2025/assets/papers/Paper_1220/paper.pdf)

**Parametric / generative strand models (2024–2026)**
- **Perm** (C. He et al., ICLR 2025; arXiv 2407.19451) is a learned parametric hair model. It uses a PCA strand representation in the frequency domain that disentangles global structure (θ) from local curl (β). **Code licence: MIT.** Training data is Hair20k (~20k hairstyles; USC-HairSalon augmented with horizontal flips). It supports interpolation between two hairstyles (θ, β or joint modes) and exports `.abc` and `.data` strands. Single-view results are released "uncurated"; the authors say "curly/kinky hairstyles are currently not well reconstructed". Models trained on Adobe-internal data and the single-view pipeline are available only by contacting the authors. It needs CUDA ≥11.3 and gcc ≤10. — [GitHub](https://github.com/c-he/perm); [arXiv](https://arxiv.org/abs/2407.19451v4)
- **HAAR** (Sklyarova et al., CVPR 2024) is text-conditioned diffusion over latent "hair maps" of guide strands, upsampled to dense hairstyles with up to ~100K strands. It takes seconds per hairstyle, compared with hours for SDS methods such as TECA. It also offers image-to-hairstyle and text editing. **Licence:** scientific research only. The README lists an "NVIDIA A100 40GB GPU" requirement and does not mention a Blender exporter. — [project](https://haar.is.tue.mpg.de/); [GitHub](https://github.com/Vanessik/HAAR)
- **DiffLocks** (Rosu et al., CVPR 2025) **[NEW 2025]** trains an image-conditioned diffusion transformer on a 40K-hairstyle synthetic dataset. It predicts a scalp texture of per-strand latent codes, which decode directly to ~100K strands with no post-processing, and it is the first method to reconstruct afro and highly curled hair from a single image. It exports `.blend` (Blender 4.1.1) and Alembic (`--export_alembic`) and needs the NATTEN and FlashAttention-2 CUDA kernels. **Licence:** custom non-commercial scientific research licence (Meshcapade) that allows "non-commercial scientific research, non-commercial education, or non-commercial artistic projects" and prohibits commercial use. — [CVPR poster](https://cvpr.thecvf.com/virtual/2025/poster/33017); [GitHub](https://github.com/Meshcapade/difflocks); [LICENSE](https://github.com/Meshcapade/difflocks/blob/main/LICENSE)
- **Im2Haircut** (Sklyarova et al., ICCV 2025) **[NEW 2025]** builds a transformer hair prior trained on synthetic plus real data, then fine-tunes a Gaussian-splatting reconstruction on one or more images to produce strand geometry. Code availability and licence could not be verified because the project page was blocked. — [arXiv PDF](https://arxiv.org/pdf/2509.01469); [ICCV poster](https://iccv.thecvf.com/virtual/2025/poster/2401)
- **TANGLED** **[NEW 2025]** (snippet) is a latent diffusion model conditioned on multi-view lineart. It ships a "MultiHair" dataset of 457 hairstyles annotated with 74 attributes, with an emphasis on culturally significant and complex styles. arXiv ID and venue are unverified. — [catalyzex listing](https://catalyzex.com/author/Zijun%20Zhao) (snippet)
- **HairLRM** (arXiv 2606.15238, ACM SIGGRAPH 2026 Conference Paper) **[NEW 2026]** uses a Large Reconstruction Model mesh as a structural anchor and a "Dual Orientation AutoEncoder" that lifts coarse geometry into strands. It targets failures on ponytails (occlusion) and curls (directionality). A related paper, "Strand-based Hairstyle Generation via Large Reconstruction and Multimodal Models" (arXiv 2608.13679), is known by title only. — [arXiv](https://arxiv.org/abs/2606.15238); [HKBU listing](https://scholars.hkbu.edu.hk/en/publications/hairlrm-strand-based-hair-modeling-via-large-reconstruction-model/); [arXiv 2608.13679](https://arxiv.org/pdf/2608.13679)
- "HairFormer": I found no verifiable paper by that exact name. Searches returned DiffLocks/Im2Haircut and an unidentified arXiv 2505.06166. — [search result](https://arxiv.org/pdf/2505.06166v1) (title not confirmed)

**Datasets underlying these models**
- USC-HairSalon: 343 synthetic hairstyles with up to 10,000 strands each, aligned to a template bust (as described in the Neural Haircut supplemental). Its licence was not found. — [Neural Haircut supplemental](https://openaccess.thecvf.com/content/ICCV2023/supplemental/Sklyarova_Neural_Haircut_Prior-Guided_ICCV_2023_supplemental.pdf)
- DiffLocks dataset: 40K hairstyles, each ~100K strands, with a rendered RGB image and metadata. — [GitHub](https://github.com/Meshcapade/difflocks)

### Inferences
- **Usable for card conversion.** Every strand method above (Perm, HAAR, DiffLocks, MonoHair, GaussianHaircut, HairStep) outputs 3D polylines that Blender can import as Curves (Alembic, or `.blend` directly for DiffLocks). They are therefore valid inputs to any strands→cards converter (Q2). The output is realistic, physically grown hair, not CK3's painterly medieval card style, so a conversion step must also adapt the textures (Q2/Q6).
- **Licensing for a free hobby mod.** DiffLocks' licence explicitly permits "non-commercial artistic projects", which plausibly covers a free CK3 mod. Whether redistributing *generated assets* inside a mod is covered is not stated; this needs a read of the full licence text. HairStep (CC BY-NC) and GaussianHaircut (CC BY-NC-SA) also allow non-commercial use. HAAR ("scientific research purposes only") is the most restrictive. Perm's MIT code is the most permissive, but its pretrained weights come from OneDrive and their data provenance traces to USC-HairSalon, whose licence is unknown.
- **RTX 5080 (16 GB, Blackwell) feasibility.** Several repos pin old toolchains: Perm needs CUDA 11.3 and gcc ≤10; MonoHair CUDA 11.3; GaussianHaircut CUDA 11.8; HairStep PyTorch 1.9/CUDA 11.1. Blackwell generally needs CUDA 12.8+ builds of PyTorch and custom kernels, so expect rebuild work (this is general knowledge, not verified per repo). HAAR's stated A100-40GB requirement suggests it may not fit in 16 GB without changes. DiffLocks depends on NATTEN and FlashAttention-2 builds for the GPU.
- **Most practical per-photo use.** Running a single-image reconstructor (DiffLocks, or HairStep-style strand/depth maps) on the photo gives a 3D hair *target*: silhouettes from several views, depth, and flow direction. Library items and shape genes can then be scored against that target, which is more informative than the front-view silhouette alone. This is probably more useful than converting each user's reconstruction into a new CK3 mesh, which would also need custom per-character mod assets.

### Gaps
- Im2Haircut and Neural Haircut code licences, MonoHair LICENSE terms, USC-HairSalon licence, and the DiffLocks *dataset* licence (whether it differs from the code) were not readable through the proxy.
- No benchmark numbers (e.g., strand precision/recall) were collected; I could not open any paper PDF.
- TANGLED's arXiv ID and venue, and the content of CGHair (Q2), were not verified.

---

## Q2. Hair-card generation/conversion from strands or parameters (papers, engine tools, Blender add-ons, procedural generators)

### Takeaway
Research on automatic strands→cards matured in 2025–2026: Strands2Cards (SIGGRAPH Asia 2025), Auto Hair Card Extraction/"DiffHairCard" (ACM TOG 2025) and CGHair (CVPR 2026). There is also a reverse cards→strands method, HairCS (arXiv, Sept 2026). The production-ready options in practice are UE5's Hair Card Generator and Blender tools: Hair Tool (paid, 30+ procedural presets, texture baking), Daniel Bystedt's free Geometry-Nodes curves→cards setup, and the LS Hair Card Tool. For a 30–50 item library, Blender curves plus a GN or Hair Tool card pipeline is enough. The research methods matter mainly if you want to batch-convert many strand hairstyles automatically.

### Cited Findings
**Research: strands → cards**
- **Strands2Cards: Automatic Generation of Hair Cards from Strands** (K. Tojo, L. Hu, N. Umetani, H. Li; SIGGRAPH Asia 2025 Conference Papers) **[NEW 2025]**:
  - Clusters strands into "wisps".
  - Builds a hairstyle-preserving texture per wisp by skinning-based alignment of its strands into a normalised pose in UV space; textures can be **shared among similar wisps** to save atlas space.
  - Fits polygon strips to the clusters by differentiable rendering of transparent, cluster-coloured coverage masks.
  - Converts a full hairstyle of >100K strands in about 20 s, and outperforms prior approaches on curly and wavy hair.
  - No public code was found.
  — [MBZUAI repository](https://irep.mbzuai.ac.ae/items/042f079d-4815-4160-ac21-34f2df9a4967)
- **Auto Hair Card Extraction for Smooth Hair with Differentiable Rendering** (arXiv 2505.18805; v1 was titled "DiffHairCard") **[NEW 2025]**. Authors: Zhongtian Zheng, Tao Huang, Haozhe Su, Xueqi Ma, Yuefan Shen, Tongtong Wang, Yin Yang, Xifeng Gao, Zherong Pan, Kui Wu (LIGHTSPEED + University of Utah); ACM TOG 2025 per the search snippet. It converts strand hair into a **limited number of cards and textures**. Each strand is encoded as a projected 2D curve in texture space, which makes the conversion end-to-end differentiable. — [arXiv PDF](https://arxiv.org/pdf/2505.18805); [v1 HTML](https://arxiv.org/html/2505.18805v1)
- **CGHair: Compact Gaussian Hair Reconstruction with Card Clustering** (Luo et al., CVPR 2026; arXiv 2604.03716) **[NEW 2026]**. Title and venue are confirmed; content is unverified (pages blocked). The title implies reconstruction into a compact card-clustered Gaussian representation. — [CVF open access](https://openaccess.thecvf.com/content/CVPR2026/html/Luo_CGHair_Compact_Gaussian_Hair_Reconstruction_with_Card_Clustering_CVPR_2026_paper.html); [CVPR 2026 poster](https://cvpr.thecvf.com/virtual/2026/poster/38164)
- **HairCS: Reconstructing Strand-Based Hair from Hair Cards** (arXiv 2609.16465, submitted 15 Sept 2026; University of Utah, LIGHTSPEED, UCLA) **[NEW 2026]**. It converts a card model into strands that grow from the scalp with uniform roots and filled volume. The output is compatible with strand rendering, simulation and grooming modifiers (clump/curl/noise), and it is validated on short, long, curly, bun and ponytail styles. — [arXiv PDF](https://arxiv.org/pdf/2609.16465); [gamedev.net summary](https://gamedev.net/news/5764-haircs-reconstructing-strand-based-hair-from-hair-cards/)

**Engine / DCC tools**
- Unreal Engine has an official "Hair Card Generator for Grooms" (documentation page exists; contents not readable through the proxy) and an 80.lv tutorial. — [Epic docs](https://dev.epicgames.com/documentation/unreal-engine/hair-card-generator-for-grooms-in-unreal-engine); [80.lv](https://80.lv/articles/learn-how-to-use-ue5-s-hair-card-generator)
- **Hair Tool** (Bartosz Styperek, Blender add-on, sold on Gumroad):
  - Generates and models card-based hair.
  - Includes "more than 30 fully procedural hairstyle presets".
  - Curve resampling/decimation, manual or automatic UVs.
  - Bakes normal, AO, diffuse, tangent, ID, root and flow maps, with channel packing.
  - Transfers UVs and vertex weights from the character to the cards; generates bones with jiggle preview.
  - Version 4.x targets Blender 4.2+, 3.x targets 3.6–4.1.
  — [Gumroad listing](https://bartoszstyperek.gumroad.com/l/hairtool) (snippet)
- **Daniel Bystedt's free Geometry Nodes setup** (Blender 3.6+, Gumroad) automatically converts curve hair into cards: strips deformed along the curves, "with the twist aligned to the underlying surface geometry". It can be tuned in the viewport or with sliders. — [CG Channel](https://www.cgchannel.com/?p=156171); [80.lv](https://80.lv/articles/free-hair-cards-from-curves-setup-for-blender) (snippets)
- **LS Hair Card Tool** (paid, Blender 4.2+) is a non-destructive GN-modifier tool that makes cards from curves *or* meshes. **HairTG-Cards 2** (paid) provides a card workflow integrated with Substance 3D Painter and Blender. — [Superhive LS Hair Card Tool](https://superhivemarket.com/products/ls-hair-card-tool); [CG Channel tag page](https://www.cgchannel.com/tag/hair-cards-from-curves/) (snippets)
- Other Blender market tools: "Hair Particle To Hair Card Generator", "Hair Cards To Curves" (reverse conversion) and "Blender Pro Hair Suite". — [Superhive: particle→card](https://superhivemarket.com/products/hair-particle-to-card-generator-for-blender); [Superhive: cards→curves](https://superhivemarket.com/products/hair-cards-to-curves); [Pro Hair Suite docs](https://superhivemarket.com/products/blender-pro-hair-suite-addon/docs) (snippets)

**Procedural / parametric generators**
- Perm offers a *parametric* strand space (θ global structure, β curl) with interpolation between hairstyles (MIT code). It can serve as a continuous "hairstyle generator" whose strands are then carded. — [GitHub](https://github.com/c-he/perm)
- HAAR generates strand hairstyles from text prompts in seconds, but its licence is research-only. — [GitHub](https://github.com/Vanessik/HAAR)
- Hair Tool's 30+ procedural presets are the closest off-the-shelf "parametric hair-card generator" inside Blender. — [Gumroad](https://bartoszstyperek.gumroad.com/l/hairtool) (snippet)

**Industry direction**
- In *Indiana Jones and the Great Circle* (MachineGames, SIGGRAPH 2025 Advances in Real-Time Rendering), most hair is rendered as strands at 60 fps. Hair cards remain only for a couple of corpses and for dogs. — [Advances 2025 slides](https://www.advances.realtimerendering.com/s2025/content/Strand%20Hair%20in%20IJGC%20-%20Final%20Slides%20(Post-Conference).pdf); [course page](https://advances.realtimerendering.com/s2025/) (snippet)

### Inferences
- **For Method (1), procedural generation.** The simplest robust pipeline in Blender is: parametric guide curves (Blender hair Curves + GN, or Hair Tool presets) → interpolate children → cards via Bystedt's GN setup or Hair Tool → **reuse CK3's existing hair texture atlas** rather than baking new ones. Reusing the atlas keeps the in-game hair shader, alpha and hair-colour gene working and the art style consistent. Baking new textures (Hair Tool can) is only needed for curl or afro textures CK3 lacks.
- **Wisp-level texture sharing and clustering.** Strands2Cards' idea is the key principle to copy manually: cluster strands into wisps (k-means on root position plus shape), fit one card per wisp, and assign each card one of a small set of shared atlas tiles. This mirrors how vanilla game hair is built.
- **HairCS (cards→strands) plus Strands2Cards/GN (strands→cards)** suggests a round-trip workflow for Method (3): turn vanilla CK3 cards into editable strands, restyle them in Blender's curve tools, and re-card them. HairCS code was not confirmed, and Blender's "Hair Cards To Curves" add-on is a commercial alternative.
- Paper methods with no public code (Strands2Cards, Auto Hair Card Extraction) are not directly usable. Their published algorithms (wisp clustering, differentiable coverage fitting) could be approximated with Blender GN plus a Python k-means script.

### Gaps
- UE Hair Card Generator settings (card count, LODs, atlas size, texture channels, UE version, plugin status) could not be read from Epic docs.
- Prices and licences of the Blender add-ons could not be confirmed because Gumroad and Superhive were blocked.
- Code availability for Strands2Cards, Auto Hair Card Extraction, CGHair and HairCS is unconfirmed.
- I did not find an older EA/Frostbite or Eurographics card-generation paper with verifiable details. Earlier "real-time hair card" literature remains a gap.
- CK3's hair shader texture layout (which channels hold alpha, flow, root/tip, ID) was not researched. Matching it is required for any new atlas.

---

## Q3. Hairstyle retrieval/matching to a fixed library; taxonomies, attribute datasets, useful descriptors

### Takeaway
Every system that maps photos onto a *fixed* hair library uses some form of **retrieval plus refinement**:
- Hu 2015 and AutoHair: database search on 2D silhouette and orientation, then deformation.
- Pinscreen 2017: CNN attribute classification narrows a polystrip (card) database, then the hair is deformed.
- AgileAvatar: discrete relaxation-and-search through a differentiable imitator.
- Tag-based annotation: region-wise semantic tags shared by photos and assets.
- Make-A-Character 2: direct CNN classification into an asset library.

For a 150–200-item library with continuous shape genes, the strongest published analogue is **coarse tag/attribute filtering → render-and-compare on silhouette plus orientation → continuous parameter fit**, which the project already partly does.

### Cited Findings
- **Pinscreen, "Avatar digitization from a single image for real-time rendering"** (Hu, Saito, Wei, Nagano, Seo et al., ACM SIGGRAPH [Asia] 2017; patent US 10535163):
  - Hair is represented as **polystrips** (polygonal strips, i.e., hair cards), "compatible with existing game engines".
  - Pipeline: find a subset of similar polystrip hairstyles in a large database, choose the most alike, deform it to fit the 2D hair, then run "polystrip patching optimization" to fix collisions and bald spots and apply suitable textures.
  - "Hairstyle retrieval performance is enhanced using a deep convolutional neural network for semantic hair attribute classification."
  — [SIGGRAPH history](https://history.siggraph.org/?p=210642); [Google Patents](https://patents.google.com/patent/US10535163); [USC VGL PDF](https://vgl.ict.usc.edu/Research/3DAvatar/PINSCREEN%203D%20AVATAR%20FROM%20A%20SINGLE%20IMAGE.pdf) (snippets)
- **AgileAvatar** (arXiv 2211.07818; listed in the SIGGRAPH history archive) predicts an "avatar vector" that mixes continuous parameters (e.g., head length) and discrete ones (e.g., hair type). It notes that hairstyle is especially hard because "hair includes hundreds of options with only minor differences". Method: stylise the selfie into the avatar domain, train a **differentiable imitator** of the avatar renderer, then run a **cascaded relaxation-and-search** for the discrete parameters. — [arXiv PDF](https://arxiv.org/pdf/2211.07818); [SIGGRAPH history](https://history.siggraph.org/?p=215758)
- **SwiftAvatar** (arXiv 2301.08153), "Efficient Auto-Creation of Parameterized Stylized Character on Arbitrary Avatar Engines", is related work (title only verified). — [arXiv PDF](https://arxiv.org/pdf/2301.08153)
- **"Tag-based annotation creates better avatars"** (arXiv 2302.07354):
  - Maps both face photos and avatar hairstyles into a semantic **tag space**; the best avatar per photo is retrieved through tags.
  - Tags are region-specific, e.g., "Hair direction on the top of the head" and "Hair curliness level on the side of the head", and were designed iteratively with domain knowledge.
  - Detailed tags gave higher annotator agreement and less label noise.
  - The tags generalised across Google Cartoonset, MetaHuman and NovelAI.
  - Follow-ups: "Tag-Based Annotation for Avatar Face Creation" (arXiv 2308.12642) and "Neurosymbolic Tag-Based Annotation for Interpretable Avatar Creation" (PMLR vol. 284, 2025).
  — [arXiv PDF](https://arxiv.org/pdf/2302.07354); [arXiv 2308.12642](https://arxiv.org/pdf/2308.12642); [PMLR](https://proceedings.mlr.press/v284/liu25a.html)
- **Make-A-Character 2** (arXiv 2501.07870) **[NEW 2025]** uses "a specialized hairstyle classification model powered by convolutional neural networks that directly maps a portrait image to a specific hairstyle within an asset library" (snippet). — [arXiv PDF](https://arxiv.org/pdf/2501.07870)
- AutoHair's data-driven matching depends on predicted **hair segmentation plus hair growth direction (orientation) maps**. Hu 2015 optimises 2D similarity and **local orientation coherency**. Both methods use silhouette plus orientation as the matching descriptor. — [AutoHair](https://history.siggraph.org/?p=102839); [Hu 2015](https://history.siggraph.org/?p=112766)
- HairStep argues that a **strand map (2D orientation) plus depth map** narrows the synthetic-to-real gap better than raw orientation maps. Its released differentiable strand renderer produces orientation and depth maps from 3D strands, which is exactly what you need to render library candidates into the same feature space as the photo. — [GitHub](https://github.com/GAP-LAB-CUHK-SZ/HairStep); [arXiv](https://arxiv.org/pdf/2303.02700)
- Struct2Hair (CASA 2022) proposes "a hair shape descriptor for hairstyle modeling" (title only verified). — [Bournemouth eprint](https://eprints.bournemouth.ac.uk/37986/1/Struct2Hair_CASA2022.pdf)

**Datasets and taxonomies**
- **K-Hairstyle** (KAIST, Nestyle, Brandi; **ICIP 2021**, not ICCV; arXiv 2102.06288) is a Korean hairstyle dataset with hair attributes annotated by Korean expert hairstylists, plus segmentation masks. It is used for dyeing, hairstyle transfer and **hairstyle classification**. Project page: psh01087.github.io/K-Hairstyle. **Image count conflicts:** 500,000 high-resolution images ([arXiv abstract snippet](https://arxiv.org/pdf/2102.06288)) versus 256,679 images ([Papers with Code dataset page](https://astro.paperswithcode.com/dataset/k-hairstyle)), probably because different paper versions report different numbers. — [arXiv](https://www.arxiv.org/abs/2102.06288); [KAIST repository](https://koasas.kaist.ac.kr/handle/10203/290621)
- **Figaro-1k**: 1,050 unconstrained images spread equally over **7 hairstyle classes** (straight, wavy, curly, kinky, braids, dreadlocks, short), released publicly with manual hair masks. — search-result summary citing the [K-Hairstyle paper](https://arxiv.org/pdf/2102.06288) and "Hair detection, segmentation, and hairstyle classification in the wild" ([Image and Vision Computing](https://www.datalearner.com/academic/journal-papers/0262-8856/volumes-and-issues/341/paper-detail/27823)) (snippet)
- **MultiHair** (TANGLED): 457 3D hairstyles with **74 attributes**, emphasising complex and culturally significant styles (snippet). — [catalyzex](https://catalyzex.com/author/Zijun%20Zhao)
- African hairstyle dataset clustering with PCA + k-means (arXiv 2306.06061), title only. — [arXiv PDF](https://arxiv.org/pdf/2306.06061)
- Curly-hair taxonomy for assessing and classifying curly-hair needs (Int. J. Cosmetic Science 2024, Daniels et al.); title only. — [UAL repository PDF](https://ualresearchonline.arts.ac.uk/id/eprint/21508/1/Intern%20J%20of%20Cosmetic%20Sci%20-%202024%20-%20Daniels%20-%20Towards%20a%20taxonomy%20for%20assessing%20and%20classifying%20the%20needs%20of%20curly%20hair%20%20A.pdf)

### Inferences
- **The project's structure is already the right shape.** It has discrete hair meshes plus 5 continuous blend-shape genes, which matches AgileAvatar's mixed discrete/continuous parameter search. Improvements suggested by the literature:
  1. Replace the coarse CLIP attributes (length/bangs/texture) with **region-wise tags** in the style of the tag-based paper. Possible tags: fringe type (none / blunt / see-through / curtain / side-swept), fringe length (above brow / brow / eye), part (centre/side/none), top volume, side coverage of the ears, side length (ear / jaw / shoulder / chest), back length, curl class (straight/wavy/curly/coily), tied state (loose/ponytail/bun/braid). Annotate each of the ~150–200 library items **once** with the same tags; this is cheap with a small library.
  2. Use the tags to prune to the top-k candidates (e.g., 10–20). Then render candidates in-game, as already done, and compare on **(a) hair-mask IoU or Chamfer distance, (b) orientation or strand map similarity** (e.g., Gabor or HairStep-style strand map from the photo versus the rendered card flow), and **(c) region-weighted IoU** that up-weights the forehead/fringe band and the jaw-level side band.
  3. Fit the 5 shape genes by continuous search (CMA-ES or a grid) per candidate, then pick the best (mesh, genes) pair. This is the relaxation-and-search pattern.
- The orientation descriptor matters most for bangs and part direction, which a silhouette cannot see (a side-swept fringe and a blunt fringe can share a silhouette). AutoHair, Hu 2015 and HairStep all rely on it.
- K-Hairstyle is the only large dataset whose label vocabulary is likely to include Korean/idol cuts such as two-block and see-through bangs. It is the best source for training or validating a fringe/cut classifier for the "modern hairstyles" library, but its exact class list was not verified. Figaro's 7 classes are too coarse for cuts and useful only for curl/texture.

### Gaps
- K-Hairstyle's exact class list (e.g., whether "two-block", "see-through bang" or "C-curl" are classes), its access conditions (AI Hub registration, Korean-resident restrictions) and its licence were not verifiable (project page blocked).
- CelebA attribute names (Bangs, Straight_Hair, Wavy_Hair, Bald, Receding_Hairline, hair colours) are known from general background, not verified in this session.
- I found no quantitative comparison of CLIP embeddings versus silhouette versus orientation descriptors for hairstyle retrieval.

---

## Q4. How commercial photo-to-avatar systems handle hair (library size, matching method)

### Takeaway
Shipping photo-to-avatar systems almost never reconstruct hair. They fit the face and then **either let the user pick a hairstyle or classify into a fixed library**. Published examples: EA GameFace maps to "the closest style supported in-game"; Ready Player Me relies on user choice among roughly 200 customisation options; Pinscreen retrieves and deforms from a polystrip database; AgileAvatar searches discrete hair parameters among hundreds of options; Make-A-Character 2 uses CNN classification into an asset library. NetEase's MeInGame (face-to-parameter) covers face shape and texture only.

### Cited Findings
- **EA SPORTS GameFace (UFC):**
  - Users upload photos from several angles to build the face.
  - Hair style and colour are then changed on the website.
  - "Not all hair styles and facial hair are supported on a 1-to-1 appearance from website to in-game"; the game "should determine the closest style that is supported in-game".
  — [EA Answers HQ](https://answers.ea.com/t5/UFC/Info-How-do-I-import-my-GameFace-to-UFC/m-p/3022110) (snippet)
- **Ready Player Me:** a selfie generates the avatar. Users then customise hairstyle, eyebrows, eyes and so on, from about "200 different avatar customization options" in total (early figure). Asset-creation docs exist for custom hairstyles. I found no evidence of automatic hairstyle matching. — [Ryan Schultz blog 2020](https://ryanschultz.com/2020/05/23/ready-player-me-creates-a-3d-avatar-for-mozilla-hubs-from-a-selfie/); [RPM docs: hairstyle](https://docs.readyplayer.me/ready-player-me/customizing-guides/create-custom-assets/hairstyle); [80.lv](https://80.lv/articles/creating-3d-avatars-from-a-single-selfie/) (snippets)
- **MeInGame** (NetEase Fuxi AI Lab + University of Michigan; arXiv 2102.02371) predicts game-character face shape and texture from one portrait. Hairstyle selection is not described as part of the method. — [arXiv](https://arxiv.org/abs/2102.02371v1); [Neurohive](https://neurohive.io/en/news/meingame-neural-network-generates-game-character-from-a-face-image/)
- **Pinscreen** (2017): polystrip hair database plus CNN attribute classification plus deformation and patching, aimed at "gaming and social VR". — [Google Patents](https://patents.google.com/patent/US10535163)
- **AgileAvatar:** hair is one of the discrete parameters, with "hundreds of options with only minor differences". — [arXiv PDF](https://arxiv.org/pdf/2211.07818)
- **Make-A-Character 2** **[NEW 2025]**: CNN hairstyle classifier into an asset library (snippet). — [arXiv PDF](https://arxiv.org/pdf/2501.07870)
- **MetaHuman** (Epic) **[NEW 2025]**: since MetaHuman 5.6 (June 2025), MetaHumans can be used in Unity, Godot, Blender, Houdini, Maya "or any other engine or DCC". Use is free up to US$1M revenue per year; above that a US$1,850 per-seat annual licence applies. MetaHumans, grooms and outfits can be sold on Fab. Free Maya and Houdini plugins export grooms to Alembic. — [CG Channel](https://www.cgchannel.com/2025/06/you-can-now-sell-metahumans-or-use-them-in-unity-or-godot/); [Digital Production](https://digitalproduction.com/2025/06/05/metahumans-graduate-ready-for-unity-godot-and-the-fab-cash-register/); [MetaHuman licence page](https://www.metahuman.com/license)
- Snapmoji (arXiv 2503.11978), "Instant Generation of Animatable Dual-Stylized Avatars", is a Snap paper on stylised avatar generation **[NEW 2025]** (title only; hair handling not verified). — [arXiv HTML](https://arxiv.org/html/2503.11978v1)

### Inferences
- Industry practice confirms the project's plan of matching a **fixed library plus continuous shape parameters**, not per-user reconstruction. GameFace's fallback to the "closest supported style" is the same problem at production scale.
- Library sizes in production stylised-avatar systems are in the **hundreds** (AgileAvatar's "hundreds"; Ready Player Me's "~200 options" across all categories). A CK3 library of ~150 vanilla plus 30–50 modern styles is within the normal range. Coverage of *cut categories* (fringe types, lengths, curl classes) matters more than raw count.

### Gaps
- No primary sources were found for hair handling in Apple Memoji, Bitmoji, Zepeto, NetEase F2P (beyond face), EA GameFace's matching algorithm or library size, or MetaHuman Creator's groom count.
- Ready Player Me's current service status (2025–2026) was not verified.

---

## Q5. Open, reusable hair-card / hair asset collections (free/CC licences) as sources for Method (2)

### Takeaway
Truly open (CC0) *hair meshes* are scarce. The best CC0 source of complete hairstyles is MakeHuman/MPFB. MetaHuman grooms are now licensable outside Unreal (free under US$1M revenue) but are realistic strand grooms needing carding. Most other free packs are **card textures/alphas only** (OwlishMedia CC0 alphas, Shy Neko, Darcy anime cards), useful for building new cards but not complete hairstyles. Research strand datasets (DiffLocks 40K, Hair20k/USC-HairSalon) are non-commercial or of unclear licence.

### Cited Findings
- **MakeHuman / MPFB2:** "All core assets are shared under Creative Commons CC0." The MakeHuman system assets and "MakeHuman Hair 01" hairstyle packs are listed as CC0. A "System Hair Materials 02" pack was added 2026-07-17. — [MakeHuman asset packs](https://static.makehumancommunity.org/assets/assetpacks/index.html); [MakeHuman licence](https://static.makehumancommunity.org/about/license.html); [MPFB assets doc](https://static.makehumancommunity.org/mpfb/docs/assets.html) (snippets)
- **OwlishMedia "Hair Alphas For Days"** (OpenGameArt): CC0, 85 hair alphas (PNG transparent plus B/W), 138.5 MB, 16K+ downloads. — [OpenGameArt](https://opengameart.org/content/hair-alphas-for-days-hair38png) (snippet)
- **Free card texture packs (licence terms not verified beyond "free"):**
  - Shy Neko hand-drawn hair cards (18 base colours, normal, alpha, matcaps, 2K) — [Gumroad](https://shyneko.gumroad.com/l/Haircards)
  - "Free Hair Card Alphas" (16-bit, 4096², 2 sheets × 4 alphas) — [Payhip](https://payhip.com/b/kq6R1)
  - Darcy's free anime hair cards (39 textures) — [Gumroad](https://xonic.gumroad.com/l/NXjjB)
  (snippets)
- **MetaHuman grooms:** usable in other engines and DCCs since 5.6, under the MetaHuman licence; Alembic groom export through the Maya/Houdini plugins. — [CG Channel](https://www.cgchannel.com/2025/06/you-can-now-sell-metahumans-or-use-them-in-unity-or-godot/)
- **Paid stylised libraries (not open, but candidates for kitbash reference):**
  - Danil Goe "200 Stylized Hairstyles" (100 male / 100 female; Blender asset library plus 200 FBX; ~2,000–5,300 vertices each; short/long/spiky/braided/curly/anime) — [Superhive](https://superhivemarket.com/products/stylized-hair-asset-library--100-male--100-female--200-bundle); [Unreal forum](https://forums.unrealengine.com/t/danil-goe-200-stylized-hairstyles-male-female-hair-asset-library/2763432)
  - "HairCraft Library – 20 Asian Short Hairstyles" — [Superhive](https://superhivemarket.com/products/haircraftlibrary)
  (snippets)
- **Research strand datasets:**
  - DiffLocks 40K (~100K strands each, `.blend`/Alembic; non-commercial licence) — [GitHub](https://github.com/Meshcapade/difflocks)
  - Perm's Hair20k (augmented USC-HairSalon; download via OneDrive; repo MIT, data licence unstated) — [GitHub](https://github.com/c-he/perm)
  - USC-HairSalon (343 styles; licence not found) — [Neural Haircut supplemental](https://openaccess.thecvf.com/content/ICCV2023/supplemental/Sklyarova_Neural_Haircut_Prior-Guided_ICCV_2023_supplemental.pdf)

### Inferences
- **Best legal source for Method (2) in a redistributable CK3 mod:** MakeHuman CC0 hairstyles (complete meshes; dated, generic Western styles) and self-made cards using CC0 alphas.
- **MetaHuman grooms** are a plausible source of modern cuts (short fades, bobs, curly styles), now that the licence allows non-Unreal use. Whether redistributing *converted derivatives* in a free Paradox mod is permitted should be checked against the full licence text. Paid libraries (Danil Goe, HairCraft) almost certainly forbid redistributing raw assets; a mod that ships converted meshes may breach their EULAs.
- **DiffLocks' 40K dataset** is the richest source of curly/afro strand hairstyles. Its non-commercial "artistic projects" clause may permit use in a free mod, but redistribution terms need checking.

### Gaps
- Licence wording for the Gumroad/Payhip free packs and the Danil Goe/HairCraft EULAs was not readable.
- Sketchfab CC-BY hair assets, Blender Studio (CC-BY) character hair and VRoid hair were not searched (search budget exhausted).
- The exact list and count of MakeHuman CC0 hairstyles was not verified.

---

## Q6. Synthesis: recommendations for the three production methods and for matching

### Takeaway
Use **Method (3), kitbashing vanilla CK3 cards, as the backbone** for art-style and shader consistency, with the existing 5 shape axes. Use **Method (1), procedural Blender curves → GN or Hair Tool cards using CK3's own atlas,** for modern cuts that vanilla pieces cannot cover (two-block, idol see-through bangs, bobs). Reserve **Method (2), conversion,** for curly/afro and other volumetric styles, sourced from CC0 MakeHuman or (licence permitting) MetaHuman/DiffLocks strands, carded in Blender. For matching, upgrade to **region-wise tags → top-k render-and-compare on silhouette plus orientation/strand maps → continuous gene fit**, optionally with a DiffLocks/HairStep 3D target.

### Cited Findings
- Card conversion from strands is automatable in seconds in research (Strands2Cards: >100K strands in ~20 s; wisp texture sharing). No code was found, so Blender GN tools (free Bystedt setup; Hair Tool with 30+ presets and baking) are the practical route. — [Strands2Cards](https://irep.mbzuai.ac.ae/items/042f079d-4815-4160-ac21-34f2df9a4967); [Hair Tool](https://bartoszstyperek.gumroad.com/l/hairtool); [CG Channel on Bystedt](https://www.cgchannel.com/?p=156171)
- Reverse conversion from cards to strands exists in research (HairCS, Sept 2026) and as a commercial Blender add-on ("Hair Cards To Curves"). This enables restyling vanilla cards as strands. — [HairCS](https://arxiv.org/pdf/2609.16465); [Superhive](https://superhivemarket.com/products/hair-cards-to-curves)
- The only single-image method shown to reconstruct afro and highly curled hair is DiffLocks (non-commercial licence, Blender export). Perm explicitly fails on curly/kinky hair. — [GitHub DiffLocks](https://github.com/Meshcapade/difflocks); [GitHub Perm](https://github.com/c-he/perm)
- Published photo→fixed-library systems combine attribute classification or tags with retrieval and deformation (Pinscreen), or with discrete search through an imitator (AgileAvatar). — [Pinscreen patent](https://patents.google.com/patent/US10535163); [AgileAvatar](https://arxiv.org/pdf/2211.07818); [Tag-based](https://arxiv.org/pdf/2302.07354)

### Inferences
**Method (3), kitbashing vanilla CK3 meshes (recommended first)**
- Pros: identical shader, atlas, alpha sorting and hair-colour gene behaviour, and it matches CK3's medieval look. The existing 5 blend-shape axes can be re-authored on kitbashed meshes with the same tooling already used for vanilla meshes.
- Recipe: build a parts catalogue from vanilla meshes (fringe pieces, crown/top shells, side panels, back/nape pieces, ponytail/bun add-ons) → recombine → fix intersections and card sorting → skin to the CK3 head skeleton → add the 5 shape keys.
- Good for: bobs, layered cuts, side-swept fringes, ponytails.
- Weak for: very short modern cuts (fades, two-block undercuts) and tight curls, where no vanilla pieces exist.

**Method (1), procedural/parametric generator in Blender (for modern cuts)**
- Parameterise a small "hairstyle grammar" of guide-curve templates per region (fringe, top, sides, back), each with sliders (length, lift, part position, fringe density/see-through, curl amplitude and frequency). Output Curves → GN card generator (Bystedt free or Hair Tool) → UVs mapped onto **vanilla CK3 atlas tiles**, or a new atlas baked with Hair Tool if needed.
- Wisp clustering and texture sharing (Strands2Cards idea) keeps the card count close to vanilla budgets.
- The generator can also emit the 5 shape-key variants directly, by re-evaluating the parameters at the gene extremes, so new meshes get the inheritable shape axes for free.
- Best for: two-block, idol see-through bangs, curtain bangs, short crops. Undercuts likely need a short-hair "shell" or texture on the scalp because cards cannot represent a fade well.

**Method (2), converting existing assets (for curly/afro and volume styles)**
- Source priority by licence safety: MakeHuman CC0 → self-built from CC0 alphas → MetaHuman grooms (check redistribution) → DiffLocks/Perm strands (non-commercial; check redistribution) → paid libraries (reference only).
- Pipeline: import strands (Alembic) → decimate to guide strands → cluster into wisps → GN cards → retexture to the CK3 atlas → retarget to the CK3 head → shape keys.
- Expect the most manual cleanup here (style mismatch, alpha sorting, polycount).

**Matching**
1. Hair mask from a face-parsing or segmentation model, plus an orientation/strand map from the photo (HairStep-style).
2. Region-wise tags (fringe type and length, part, side and back length, curl class, tied state) predicted by CLIP or a fine-tuned classifier. K-Hairstyle could support modern Korean cuts if its labels fit; Figaro is useful for curl class.
3. Prune to the top-k library items by tag agreement.
4. Render each candidate in CK3 at the photo's pose and compare region-weighted mask IoU (forehead/fringe band weighted most), orientation-map cosine similarity, and optionally CLIP-image similarity on the hair crop.
5. Optimise the 5 shape genes per candidate (CMA-ES or grid) and pick the best (mesh, genes) pair, as in AgileAvatar's relaxation-and-search.
6. Optional: run DiffLocks on the photo to get a 3D strand target, render it from 2–3 views, and score candidates on multi-view silhouettes. This helps with back length and volume that a frontal photo hides.
- Evaluate with the project's existing 10-point judge. The upper-bound test (pasting real hair raised the score from 4.0 to 4.9) gives the ceiling the library-plus-matching approach is trying to approach.

### Gaps
- None of the recommended pipelines has been tested on CK3's hair shader or atlas format. CK3 hair technical constraints (polycount budget, texture channels, alpha sorting, skeleton) were out of scope and not researched.
- I found no published comparison of kitbashing versus procedural versus conversion for game hair libraries. The ranking above is an inference.
- Licence redistribution questions (MetaHuman derivatives in a Paradox mod; DiffLocks-generated assets) need direct reading of the full licence texts.
