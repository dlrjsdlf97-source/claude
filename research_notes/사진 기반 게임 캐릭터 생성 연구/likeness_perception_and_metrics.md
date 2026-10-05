# Likeness: what makes a stylized face recognizable, and how to measure and learn it (photo ↔ CK3 render)

> Research-process note for the report writer: WebSearch worked until the session's search budget ran out. WebFetch was **blocked** for arxiv.org, openaccess.thecvf.com, PMC, bioRxiv, Semantic Scholar, Frontiers, huggingface.co, web.mit.edu, cs.utexas.edu, people.socsci.tau.ac.il and most university hosts. Only github.com could be fetched. Most details below therefore come from **search-result abstracts/snippets** or **GitHub READMEs**, not full papers. Where a detail comes only from a snippet, it is marked "(snippet)". Items I know from background knowledge but could **not** confirm this session are listed under "Gaps → unverified leads", never as findings. Tags: **[FOUND]** = foundational (pre-2015), **[NEW 2023–26]** = recent computational work.

---

## Q1. Which face cues matter for recognition (configural/holistic, eyebrows, eyes, hair/external features, shape vs pigmentation, contrast), and what happens at low resolution?

### Takeaway
Human identity recognition rests on more than geometry. **Pigmentation/surface reflectance is about as important as 3D shape.** Eyebrows matter as much as or more than eyes. **External features (hair, face outline) dominate for *unfamiliar* viewers**, while familiar viewers rely more on internal features, and changing the hairstyle disrupts recognition holistically. Familiar faces stay recognizable at tiny resolutions (about 7×10 px), which shows that coarse, low-spatial-frequency structure (the hair mass, dark brow and eye regions, overall contrast and colouring) carries much of the identity. All of this fits the project's finding that appearance fixes (brows, eye and lip colour, beard, hair) beat landmark-shape fixes.

### Cited Findings
- **[FOUND] Overview.** Sinha, Balas, Ostrovsky & Russell, "Face Recognition by Humans: Nineteen Results All Computer Vision Researchers Should Know About," *Proc. IEEE* 94(11):1948–1962, Nov 2006, gives 19 human results with pointers for engineers — [IEEE](https://ieeexplore.ieee.org/document/4052483); [PDF](https://inc.ucsd.edu/mplab/~marni/Igert/19results_sinha_06.pdf)
- **[FOUND] Low resolution.** Yip & Sinha (2002): people recognized more than half of an unprimed set of familiar faces at only **7×10 pixels**, and performance reached ceiling at **19×27 pixels**. Tolerance to degradation **increases with familiarity** (snippet) — [Sinha et al. 2006 PDF](https://inc.ucsd.edu/mplab/~marni/Igert/19results_sinha_06.pdf)
- **Spatial frequency.** Psychophysics suggests humans preferentially use a narrow band of low spatial frequencies for faces. A machine study found that face-identity discrimination peaked at about **22×22 px, roughly 8–10 cycles per face**, which "compares favorably to psychophysical results" (snippet) — [arXiv q-bio/0612001](https://arxiv.org/abs/q-bio/0612001); [Univ. Barcelona repository](https://diposit.ub.edu/items/e8185e80-d78c-48aa-bc51-346a87e7fa35/full)
- **[FOUND] Eyebrows.** Sadr, Jarudi & Sinha (2003), "The role of eyebrows in face recognition," *Perception* 32(3):285–293. Removing the eyebrows from **familiar (celebrity)** faces caused a very large, significant drop in recognition, and the drop was **significantly larger than when the eyes were removed**. The authors conclude eyebrows "may be at least as influential as the eyes" — [Sinha lab PDF](https://web-mit-edu.ezproxyberklee.flo.org/sinhalab/Papers/sinha_eyebrows.pdf); a 2024 follow-up comparing human and DCNN feature reliance: [Frontiers Comput. Neurosci. 2024 / PMC11035738](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC11035738/) (content not fetched)
- **[FOUND] Pigmentation vs shape.** Russell (MIT thesis with Sinha) reports that "for face recognition, pigmentation cues are about as important as shape cues." This challenges the assumption that shape dominates. Female faces have more luminance contrast between the eyes/lips and the surrounding skin than male faces, and observers use this contrast for sex classification and attractiveness — [MIT DSpace 1721.1/33734](https://dspace.mit.edu/handle/1721.1/33734)
- **[FOUND] Contrast negation.** Contrast negation impaired face matching **only when the faces varied in pigmentation**. This is evidence that pigmentation is part of the neural representation of identity — [Russell et al., *Perception* (doi 10.1068/p5490)](https://journals.sagepub.com/doi/10.1068/p5490); [preprint](https://web-mit-edu.ezproxyberklee.flo.org/sinhalab/Papers/russell_negation_preprint.pdf)
- **[FOUND] 3D shape vs 2D reflectance.** O'Toole, Vetter & Blanz (1999, *Vision Research*) split laser-scanned heads into shape-normalized faces (each face's own reflectance on the average shape) and reflectance-normalized faces (the average reflectance on each face's own shape). Tested at 0°, 30° and 60° view changes, **both 3D shape and 2D surface reflectance contributed substantially** to recognition — [PDF (Univ. Basel)](https://shapemodelling.cs.unibas.ch/gravis-literature/publications/VisRes99.pdf)
- **[FOUND] Internal vs external features by familiarity.** Ellis, Shepherd & Davies (1979, *Perception*) found an internal-feature advantage for famous faces, but **no difference between internal and external features (hair, chin, outline) for unfamiliar faces**. Young et al. (1985) found internal features were matched faster for familiar faces — [Ellis et al. 1979 PDF](https://moodle2.units.it/pluginfile.php/336418/mod_resource/content/0/Ellis79.pdf). Children rely more on outer features (hair, hairline, jaw) even for people they know — [Cambridge Neuroscience](https://neuroscience.cam.ac.uk/publications/the-development-of-differential-use-of-inner-and-outer-face-features-in-familiar-face-identification)
- **Hair.** Toseeb, Keeble & Bryant (2012, *PLoS ONE*, doi 10.1371/journal.pone.0034144): when hair was the same at learning and test, performance with and without hair did not differ, so internal features suffice for that task. **When the hair state switched between learning and test, accuracy dropped substantially.** The authors read this as holistic representation that includes hair — [PMC3312903](https://pmc.ncbi.nlm.nih.gov/articles/PMC3312903/)
- **[FOUND] Unfamiliar viewers.** Jenkins, White, Van Montfort & Burton (2011, *Cognition* 121(3):313–323): when unfamiliar viewers sorted 20–40 photos of only **2 identities**, they guessed about **7 different people** on average. Within-person photo variability is large, and unfamiliar viewers handle it poorly — [Glasgow ePrints](https://eprints.gla.ac.uk/67032); [PMC5393646](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC5393646/)
- **Critical features (Yovel lab).** Abudarham & Yovel (2016) built a 20-feature face space from human ratings (for example, "which face has thicker lips?"). **Critical features** have high perceptual sensitivity across identities and vary little across views of the same identity, and lip thickness is one example. These features support recognition across images — [Abudarham & Yovel 2016 PDF](https://people.socsci.tau.ac.il/mu/galityovel/files/2024/11/Reverseengineetingthefacespace2016.pdf). Abudarham, Shkiller & Yovel (2019), "Critical features for face recognition," *Cognition* 182:73–83, describe the critical features as a view-invariant subset — [Yovel lab page](https://people.socsci.tau.ac.il/mu/galityovel/?p=276). A related summary says **none of the celebrity faces tested was recognized after 4–5 critical features were replaced** (snippet) — [Yovel lab PDF](https://people.socsci.tau.ac.il/mu/galityovel/files/2024/11/CriticalfeaturesinDPandSRs_July18_final.pdf)
- **Parametric face-space distance predicts human identity judgments.** Jozwik, O'Keeffe, Storrs, Guo, Golan & Kriegeskorte (2022, *PNAS*) collected dissimilarity and same/different-identity judgments for **232 face pairs** sampled in the Basel Face Model (a PCA model of 3D shape and texture). **Euclidean distance in the BFM explained both dissimilarity and identity judgments "surprisingly well"** and was competitive with state-of-the-art DNNs among 16 models — [PMC9271164](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC9271164/)
- **[NEW 2023–26] Featural plus configural similarity metric.** AlignFace (arXiv 2608.14130, 2026) breaks face similarity into attribute-level comparisons, both **featural (e.g., lip colour)** and **configural (e.g., pupillary distance)**. Learned nonlinear functions map attribute similarities to overall similarity and model own-group vs other-group effects. It beats LPIPS and SSIM on human triplet judgments (abstract only, numbers not obtained) — [arXiv PDF](https://arxiv.org/pdf/2608.14130)

### Inferences
- The project's measurements match the literature. Landmarks capture only shape (and only 40–70% of it), but **pigmentation/reflectance is roughly half of identity** (Russell; O'Toole). Brows, iris colour, lip colour/contrast, beard and hair are mostly pigmentation and texture cues, which explains why "appearance fixes gave the biggest wins."
- **Who judges matters.** GPT/Claude judges, and any human who does not know the person, act as **unfamiliar viewers**. Per Ellis et al. and Toseeb et al., such viewers weight **hair and external features** as heavily as internal features. This may be part of why "hair appears to be the biggest remaining cue." For non-celebrities, the end user (who knows the person) is a *familiar* viewer and will weight internal features more. Rec: evaluate with both judge types.
- **Display size.** CK3 portraits are often seen small in the UI. Recognition at about 20–30 px per face (roughly 8–10 cycles per face) depends on low-spatial-frequency structure: hair silhouette and colour, brow darkness and thickness, the eye-region shadow, overall skin tone and the face/jaw outline. Rec: score likeness at the **in-game display size** (for example, blur or downsample both images to about 32–64 px across the face) as well as full resolution. Prioritize cues that survive blurring.
- Rec: give **eyebrow thickness, shape, colour and position** effort equal to the eyes. Treat **skin tone and facial contrast (eyes/lips vs skin)** as identity cues, not cosmetic ones.
- The 2016 Yovel paper says lip thickness is critical. If the full critical-feature list holds (see Gaps), a short prioritized cue checklist for the pipeline would be: hair colour, eye colour, eye shape, eyebrow thickness, lip thickness, with skin tone and eye spacing lower priority.
- The BFM result (Jozwik et al.) suggests a **low-dimensional parametric face space can carry human-relevant identity distance.** CK3's DNA/slider space is such a space. Distances in a well-normalized (PCA/whitened) slider space, built from the ~270 community presets, could be a useful critic feature.

### Gaps
- Could not fetch full texts, so no numbers for eyebrow/eye removal accuracies (Sadr et al.), effect sizes for pigmentation vs shape, or Toseeb accuracy figures.
- No source found that compares recognition of **stylized/painterly** faces at small sizes directly with photographs.
- **Unverified leads (background knowledge, not confirmed this session):**
  - The full Abudarham & Yovel (2016, *J. Vision*) critical/non-critical list, which I recall as critical = lip thickness, **hair colour, eye colour, eye shape, eyebrow thickness**, non-critical = eye distance, face proportion, mouth size, skin colour, nose shape.
  - Sinha & Poggio (1996, *Nature*) "presidential illusion": the same internal features appear as different people under different hair and outline.
  - Yip & Sinha (2002, *Perception*): colour helps recognition mainly when shape cues are degraded.
  - Sinha et al.'s results that configural relations are coded somewhat independently along width and height, and that face shape is encoded "slightly caricatured."
  - All of these should be checked before citing.

---

## Q2. Norm-based face space and caricature research, including selective exaggeration and computational caricature methods

### Takeaway
Caricatures (moving a face away from the average along its own deviation) are usually recognized **at least as well as, and often faster than,** veridical images, and anti-caricatures are recognized worse. The benefit is mostly in **speed**, varies across faces, and comes from **distinctiveness**. Moves in directions orthogonal to the true deviation ("lateral caricatures") change identity, so exaggerating noise is harmful. Deep face recognizers also become more accurate on caricatures, which is a warning that optimizing a recognizer can produce "machine caricatures." Computational caricature methods (WarpGAN, StyleCariGAN, DualStyleGAN) exaggerate **selectively and learn from data**, rather than applying a uniform gain.

### Cited Findings
- **[FOUND] Brennan's caricature generator (1985).** A face is 37 lines on 169 fixed points. Caricatures exaggerate **all metric differences** between a face and a norm, a holistic, uniform exaggeration — [Caricature review (ceon.rs)](https://asistent.ceon.rs/index.php/zrffp/en/article/view/6267)
- **[FOUND] Rhodes, Brennan & Carey (1987), "Identification and Ratings of Caricatures."** Caricatures of familiar faces were **identified faster** than veridical line drawings, which in turn were identified faster than anti-caricatures. **Identification accuracy did not differ** across the three — [Stony Brook record](https://researchconnect.stonybrook.edu/en/publications/identification-and-ratings-of-caricatures-implications-for-mental/)
- **[FOUND] Benson & Perrett (1991)**, "Perception and recognition of photographic quality facial caricatures," extended caricaturing to photographic images (title only; details not obtained) — [ceon.rs review](https://asistent.ceon.rs/index.php/zrffp/en/article/view/6267)
- **[FOUND] Variable benefit.** Caricatures, which increase distinctiveness, are generally recognized at least as well as undistorted images, but they help more for some faces than others. Byatt & Rhodes (1998, *Vision Research* 38:2455–2468) studied own-race vs other-race caricatures (snippet) — [index record](https://acnpsearch.tweb-dev.unibo.it/singlejournalindex/6562235)
- **[FOUND] Lateral caricatures.** These move the face orthogonally to the face's norm-deviation vector. Lateral caricatures have been reported to be **harder to recognize than anti-caricatures** at equal distance (snippet) — [CogSci proceedings paper](https://journalpub.escholarship.org/cognitivesciencesociety/article/30955/galley/20804/download)
- **[FOUND] Conflicting evidence.** Rhodes et al., "Coding spatial variations in faces and simple shapes: a test of two models" (*Vision Research*), compared veridical, caricature, anti-caricature and lateral distortions of famous faces, newly learned faces and simple shapes. The results **favoured absolute (not norm-based) coding** and indicated that "caricatures derive their power from their distinctiveness" — [PDF](https://www.harvardlds.org/wp-content/uploads/2018/05/Rhodes-Coding-spatial-variations-in-faces-and-Simple-shapes-A-test-of-two-models.-.pdf). This conflicts with norm-based interpretations, including the adaptation work below.
- **[FOUND] Norm-based coding via adaptation.** Leopold, O'Toole, Vetter & Blanz (2001), "Prototype-referenced shape encoding revealed by high-level aftereffects," *Nature Neuroscience* 4:89–94 (bibliographic only) — [record](https://itts023d.itts.ttu.edu/OdorDB/Data/70401). Review: "Adaptation and Face Perception: How Aftereffects Implicate Norm-Based Coding of Faces" — [ANU](https://researchportalplus.anu.edu.au/en/publications/adaptation-and-face-perception-how-aftereffects-implicate-norm-ba/)
- **DCNNs and caricature.** Hill, Parde, Castillo, Colón, Ranjan, Chen, Blanz & O'Toole (2019, *Nature Machine Intelligence*; arXiv 1812.10902): in a deep face-recognition network, identity is nested under gender, illumination under identity and viewpoint under illumination. **Network identification accuracy increased with caricature level.** Caricatures help by moving the identity away from other identities and reducing illumination and viewpoint effects — [arXiv PDF](https://arxiv.org/pdf/1812.10902)
- **Facial composites.** Caricaturing composites in a "multi-frame caricature" format is studied as a way to improve recognition of facial composites (title only) — [Napier](https://www.napier.ac.uk/-/media/worktribe/output-188225/understanding-the-multi-frame-caricature-advantage-for-recognising-facial-composites.ashx)
- **WarpGAN** (Shi, Deb & Jain, CVPR 2019). It learns to predict **control points that warp** the photo, plus texture style transfer, with an **identity-preserving adversarial loss** (the discriminator must also tell subjects apart). Exaggeration extent is controllable. Five caricature experts judged that "**only prominent facial features are exaggerated**." Trained on WebCaricature — [CVF](https://openaccess.thecvf.com/content_CVPR_2019/html/Shi_WarpGAN_Automatic_Caricature_Generation_CVPR_2019_paper.html); [code](https://github.com/seasonsh/warpgan)
- **StyleCariGAN** (Jang et al., SIGGRAPH 2021 / ACM TOG). A layer-mixed StyleGAN takes fine (colour/texture) layers from a caricature StyleGAN and keeps coarse layers from the photo StyleGAN. **Shape-exaggeration blocks** then modulate the coarse-layer feature maps to produce exaggeration while "preserving the characteristic appearances of the input," with a controllable exaggeration degree. Trained with WebCaricature — [arXiv PDF](https://arxiv.org/pdf/2107.04331); [GitHub](https://github.com/wonjongg/StyleCariGAN)
- **DualStyleGAN** (Yang, Jiang, Liu & Loy, CVPR 2022). An "intrinsic" style path (the face) and an "extrinsic" style path (the artistic exemplar) hierarchically modulate colour **and structural** style. Trained on small sets: **317 cartoon, 199 caricature and 174 anime images**. In a user study with 27 subjects, the preference scores were 0.93 cartoon, 0.79 caricature and 0.78 anime — [project page](https://mmlab-ntu.com/project/dualstylegan/)

### Inferences
- **Why uniform caricature failed.** A uniform gain exaggerates the landmark estimate's full deviation from the mean. Landmarks capture only 40–70% of the true deviation (eyes about 20%), so much of the measured "deviation" is **error that is partly orthogonal to the true one**. Amplifying it is effectively a *lateral* caricature, which the literature says hurts identity more than an anti-caricature. The caricature benefit is also mainly in RT, not accuracy (Rhodes et al. 1987), so a blind A/B judge would gain little even from a perfect caricature.
- **Selective exaggeration recipe (inferred):**
  1. Estimate each feature's deviation and its *uncertainty*, for example from multiple photos, multiple landmark detectors or test-time augmentation.
  2. Exaggerate only features whose deviation is large relative to the population **and** reliably measured (a signal-to-noise-gated gain, i.e. shrinkage). Population means should come from the same ethnicity and sex as the target, so you exaggerate away from the right norm.
  3. Prioritize "critical" features: brow thickness, lip thickness, eye shape, plus pigmentation.
  4. Cap the gain at a modest level, since WarpGAN-style experts judged that "only prominent features" should be exaggerated.
- The **"ethnic average face"** look is the anti-caricature failure mode: regression to the mean from weak measurements. This is known to reduce recognizability (anti-caricatures are recognized worst). Shrinkage toward the *correct* group norm, combined with gated exaggeration of the confident deviations, is the principled middle ground.
- **ArcFace pitfall.** Hill et al. show deep recognizers reward distinctiveness. Optimizing ArcFace similarity can push renders into recognizer-salient but humanly meaningless directions, effectively "machine caricatures." That fits the project's observation that ArcFace rose without any human gain. Rec: never use a frozen recognizer score alone as the optimization target.
- The ~270 community presets act as **hand-made "caricatures" in CK3 space** by skilled preset makers. Comparing each preset's slider deviation from the ethnic mean with the photo's measured deviation would show directly which features humans exaggerate, which they ignore, and by how much. That is a data-driven, selective exaggeration prior specific to CK3.

### Gaps
- No direct study found of caricature effects for **3D game-engine renders** or painterly styles.
- Could not retrieve the quantitative caricature advantage (for example, % RT gain or the optimal exaggeration level, often cited around +16–50% in the line-drawing literature; **unverified**).
- **Unverified leads:**
  - CariGANs (Cao, Liao & Yuan, SIGGRAPH Asia 2018), with a CariGeoGAN that exaggerates landmarks in PCA space.
  - 3D caricature methods (for example, "Alive Caricature from 2D to 3D," CVPR 2018; Sela et al. 2015 surface caricaturization).
  - "Learning to exaggerate" and exaggeration-intensity papers from 2023–2025.
  - All were not searched or verified because the search budget ran out.

---

## Q3. Models that measure identity similarity across domains (photo vs cartoon/caricature/game avatar), and how well they track humans

### Takeaway
Off-the-shelf photo face recognizers are trained within one domain. Caricature and cartoon identity needs cross-modal training (WebCaricature, CaVINet). Game-avatar papers (Face-to-Parameter and successors) use recognizer embeddings as a **loss**, combined with semantic-segmentation or content losses, not as a validated human-likeness metric. General perceptual metrics (LPIPS, DreamSim) are calibrated to human 2AFC judgments of *generic image* similarity, not identity. Recent work (AlignFace 2026; the BFM study) shows human face similarity is captured best by **attribute- and parameter-level** comparisons. Frontier VLMs still trail humans on similarity triplets and face-recognition benchmarks.

### Cited Findings
- **[FOUND] Game avatars: Face-to-Parameter (F2P)** (Shi, Yuan, Fan, Zou, Shi & Liu, ICCV 2019, NetEase). It optimizes a large set of physically meaningful facial parameters to match a photo using a **"discriminative loss"** (face-recognition features) and a **"facial content loss"** (face semantic segmentation). Because the engine is not differentiable, an **imitator** generator learns to mimic the engine. The method was deployed in a game and used more than 1 million times — [CVF](https://openaccess.thecvf.com/content_ICCV_2019/html/Shi_Face-to-Parameter_Translation_for_Game_Character_Auto-Creation_ICCV_2019_paper.html)
- **"Fast and Robust Face-to-Parameter Translation"** (AAAI 2020) replaces iterative optimization with a single-forward-pass parameter translator — [AAAI](https://ojs.aaai.org/index.php/AAAI/article/view/5537)
- **[NEW 2023–26] EasyCraft** (arXiv 2503.01158, 2025) handles **various facial image styles** with a ViT encoder and a parameter-generation module. It integrates with Stable Diffusion for text-based avatar creation — [arXiv HTML](https://arxiv.org/html/2503.01158v1)
- **Caricature recognition benchmark.** WebCaricature has **6,024 caricatures and 5,974 photos of 252 people**, with landmarks and evaluation protocols. It is harder than earlier sets because of many artistic styles and large intra-person variation — [arXiv PDF](https://arxiv.org/pdf/1703.03230)
- **Cross-modal verification.** CaVINet learns a shared photo/caricature representation. Reported results: **91% verification accuracy on unseen images, 75% on unseen identities, rank-1 of 85% for caricatures and 95% for photos** (snippet; dataset is the authors' CaVI set) — [arXiv PDF](https://arxiv.org/pdf/1807.11688)
- **Toonification / stylization.** DualStyleGAN trains on a few hundred stylized images per style and evaluates by user preference, not by a validated identity metric — [project page](https://mmlab-ntu.com/project/dualstylegan/)
- **[NEW 2023–26] DreamSim** (Fu, Tamir, Sundaram, Chai, Zhang, Dekel & Isola, NeurIPS 2023) is trained on **NIGHTS**, about 20k synthetic image triplets with 2AFC human judgments. It uses an ensemble of **CLIP, OpenCLIP and DINO ViT-B/16** embeddings with **LoRA or an MLP head**. 2AFC agreement: **ensemble 96.2%, OpenCLIP 95.5%, DINO 94.6%, CLIP 93.9%**. It "focuses heavily on foreground objects and semantic content while also being sensitive to color and layout." Single-backbone variants are about 3× faster. The README gives no face-specific validation — [GitHub](https://github.com/ssnl/dreamsim); [NeurIPS](https://proceedings.neurips.cc/paper_files/paper/2023/hash/9f09f316a3eaf59d9ced5ffaefe97e0f-Abstract.html)
- **[FOUND] LPIPS** (Zhang et al. 2018). The "lin" variant learns only a **linear calibration on frozen deep features** from human 2AFC data (BAPPS has 94.7k training triplets, plus JND pairs). It measures low-level patch similarity — [GitHub](https://github.com/richzhang/PerceptualSimilarity)
- **Human face similarity vs models.** BFM (3D shape plus texture PCA) distance was competitive with 16 models including state-of-the-art DNNs for predicting human dissimilarity and same/different judgments — [PMC9271164](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC9271164/)
- **[NEW 2023–26] AlignFace** (2026). Attribute-level featural and configural decomposition beats LPIPS and SSIM on human face triplets — [arXiv PDF](https://arxiv.org/pdf/2608.14130)
- **[NEW 2023–26] VLM similarity gap.** "The Many Senses of Visual Similarity" (2026) built a large human triplet dataset with multiple annotated similarity aspects. Frontier VLMs showed a **considerable gap** to human consensus, and fine-tuning VLMs produces text-prompted similarity metrics (snippet) — [alphaXiv](https://www.alphaxiv.org/abs/2607.18237.md); [HF papers](https://huggingface.co/papers/2607.18237)
- **[NEW 2023–26] VLMs on face tasks.** FaceXBench (2025) has 5,000 multiple-choice questions over 14 face tasks, including recognition in high and low resolution. **No MLLM exceeded 60%**, and the best was Qwen2-VL-72B at 57.86%. GPT-4o and Gemini 1.5 Pro were also tested — [arXiv PDF](https://arxiv.org/pdf/2501.10360). A further 2025 benchmark tests MLLMs with standard face-recognition protocols (details not obtained) — [arXiv 2510.14866](https://arxiv.org/pdf/2510.14866)
- **Recognizer behaviour.** Deep face recognizers become more accurate on caricatures, because exaggeration increases inter-identity distance — [Hill et al.](https://arxiv.org/pdf/1812.10902)

### Inferences
- The project's divergence (InsightFace up, judges flat) is expected:
  1. ArcFace-type models are trained on photos, and CK3 renders are out of domain.
  2. They reward distinctiveness along recognizer-specific directions (Hill et al.).
  3. They are known to discount hairstyle, colour and lighting, which humans, especially unfamiliar viewers, weight heavily (Q1).
- Rec: keep ArcFace or AdaFace as **one feature** in the critic, not as the objective.
- Rec: build a **multi-view feature set** for each (photo, render) pair:
  - (a) cross-domain identity embedding: ArcFace on both images, and optionally ArcFace on a "CK3-ified" photo or a photo-ified render;
  - (b) DINOv2 / CLIP / DreamSim embeddings on **aligned face crops**, on **hair-only crops** (segmentation mask) and on **inner-face crops**;
  - (c) explicit attribute comparisons (AlignFace-style): hair colour, length and style class; eyebrow thickness, colour and arch; iris colour; lip thickness and colour; skin tone (L\*a\*b\*); beard type; face-shape indices;
  - (d) parameter-space distance between the generated preset and the community preset, when one exists.
- CLIP/DINO similarity on whole images will be dominated by the shared CK3 style, background and costume. Mask to the face and hair and **compare render to render** (generated vs community preset of the same person) when possible, which removes the domain gap entirely for the celebrity set.
- For the celebrity set, the **community preset is a "gold" render**. Generated-vs-gold similarity in CK3 render space (DreamSim or DINOv2 on crops, or slider-space distance) is a strong same-domain proxy and can supervise a photo↔render critic for non-celebrities.

### Gaps
- No study found that measures the correlation between ArcFace, CLIP or DINOv2 similarity and **human likeness ratings for stylized or game renders**. This is a key empirical question the project must answer with its own labels.
- iCartoonFace, anime face recognition, and avatar-likeness user-study literature were not searched (budget exhausted).
- **Unverified leads:**
  - DreamBooth (Ruiz et al., CVPR 2023) introduced **DINO** similarity as a subject-fidelity metric, arguing CLIP-I is less sensitive to identity differences within a class. The README fetch did not confirm this.
  - Identity metrics used in ID-preserving diffusion work (InstantID, PhotoMaker, PuLID).

---

## Q4. Training a small learned likeness critic from limited human labels (~150 identities, a few hundred renders), and guarding against reward hacking

### Takeaway
The standard recipe from perceptual metrics (LPIPS-lin, DreamSim) and image reward models (ImageReward, PickScore, HPS v2) is: **frozen strong backbone(s) plus a small learned head (linear, MLP or LoRA)** trained on **pairwise or 2AFC comparisons** with a Bradley–Terry-style logistic loss. Human labels are noisy: single humans agree with consensus only about 65–78% in T2I preference sets, so the label-noise ceiling must be estimated. Optimizing hard against any learned proxy eventually lowers true quality (Goodhart). This is measured as reward-model overoptimization, and it depends on optimizer type, reward-model size, dataset size and KL regularization.

### Cited Findings
- **[FOUND/NEW] Linear head on frozen features works.** LPIPS "lin" adds only a linear calibration on frozen network features, trained on human 2AFC triplets — [GitHub](https://github.com/richzhang/PerceptualSimilarity)
- **[NEW 2023–26] DreamSim** fine-tunes a frozen CLIP + OpenCLIP + DINO ensemble with **LoRA or an MLP head** on about 20k triplets and reaches **96.2%** 2AFC agreement — [GitHub](https://github.com/ssnl/dreamsim)
- **[NEW 2023–26] ImageReward** (NeurIPS 2023): **137k expert comparisons**, a BLIP backbone plus an MLP head. It beats CLIP by 38.6%, Aesthetic by 39.6% and BLIP by 31.6% at preference prediction. Using it as a training signal (**ReFL**), the tuned Stable Diffusion wins **58.4%** of human comparisons against the untuned model — [GitHub](https://github.com/THUDM/ImageReward)
- **[NEW 2023–26] PickScore** fine-tunes **CLIP-ViT-H-14** on Pick-a-Pic: v1 has more than 500k images, v2 more than 1M examples — [GitHub](https://github.com/yuvalkirstain/PickScore)
- **[NEW 2023–26] HPS v2 / v2.1.** HPD v2 has **798k pairwise comparisons over 430k images and 107k prompts**. Preference accuracy on the ImageReward test set / HPD v2 test set:

  | Model | ImageReward test | HPD v2 test |
  |---|---|---|
  | Aesthetic predictor | 57.4% | 76.8% |
  | ImageReward | 65.1% | 74.0% |
  | HPS v1 | 61.2% | 77.6% |
  | PickScore | 62.9% | 79.8% |
  | **Single human** | **65.3%** | **78.1%** |
  | HPS v2 | 65.7% | 83.3% |
  | HPS v2.1 | 66.8% | 84.1% |

  Source: [GitHub](https://github.com/tgxs002/HPSv2)
- **[NEW 2023–26] Reward overoptimization.** Gao, Schulman & Hilton (ICML 2023) used a "gold" reward model as a stand-in for humans and trained proxy models on its labels. Optimizing the proxy, via RL or **best-of-n**, first raises and then lowers the gold score. The functional form differs between RL and best-of-n, the coefficients scale smoothly with reward-model size, and the effect depends on reward-model dataset size, policy size and the **KL penalty** coefficient — [PMLR](https://proceedings.mlr.press/v202/gao23h.html)
- **Low-dimensional, interpretable critics can match DNNs on face similarity.** BFM distance competed with 16 DNN models (Jozwik et al.) — [PMC9271164](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC9271164/). AlignFace learns nonlinear maps from attribute similarities to overall similarity — [arXiv](https://arxiv.org/pdf/2608.14130)

### Inferences (concrete recipe for the project)
- **Data construction from existing assets:**
  - (i) **Within-identity quality pairs.** For the same celebrity, a preset labelled A should beat B, and B should beat C, when judged against that person's photo. This gives roughly a few hundred ordered pairs over about 143 people.
  - (ii) **Identity-matching triplets (cheap, many).** A render of X should be closer to a photo of X than a render of Y is, where Y is a **demographically matched foil** (same sex, ethnicity, age band, ideally similar hair). Matched foils force the critic to learn identity cues, not ethnicity or sex cues. This directly targets the "ethnic average face" failure.
  - (iii) **Generated-vs-community pairs** labelled by humans over time. A small active-learning loop focuses labels where the critic is uncertain.
  - (iv) **Hard negatives** from the project's own pipeline: earlier versions that ArcFace liked but judges did not. These directly teach the critic to ignore recognizer-only gains.
- **Model.** Freeze the backbones (ArcFace/AdaFace, DINOv2, CLIP or DreamSim on aligned face, inner-face and hair crops), compute per-view similarities and **attribute deltas** (hair, brows, eyes, lips, skin, beard), then fit a **logistic / Bradley–Terry model, or a very small MLP**, on those features. With about 150 identities, keep parameters in the tens to hundreds, use strong L2 regularization, and **use leave-identities-out cross-validation** (never split renders of one person across train and test). A LoRA or MLP head on embedding differences, DreamSim-style, is the next step up only if the data grows.
- **Calibration.** Fit temperature scaling or Platt scaling on held-out identities. Report pairwise accuracy against a **label-noise ceiling**: inter-rater agreement on the A/B/C labels, or a human-judge split-half. As the HPS table shows, "single human" accuracy is only 65–78%, so a critic at about 70–80% may already be near the ceiling.
- **Anti-reward-hacking measures (inferred from Gao et al.):**
  - (1) Optimize with **best-of-n reranking** over a constrained, plausible preset space rather than gradient or RL pushing against the critic. Gao et al. found overoptimization depends on the optimizer, and best-of-n with modest n is easy to control.
  - (2) Add a **prior or KL-style penalty** that keeps sliders near the distribution of community presets (a density model over the 270 presets).
  - (3) Use an **ensemble of critics**, for example different backbones and bootstrap folds, and optimize a conservative statistic (mean minus k·std or the minimum).
  - (4) **Hold out a separate judge** (human A/B or a different VLM protocol) that is never optimized against, and track the "proxy up, gold down" turnover.
  - (5) Periodically add the optimizer's own top outputs back into labelling as hard examples.
- Because the critic includes interpretable attribute terms, failures can be diagnosed (for example "hair colour mismatch"), which is useful for a hobby pipeline with limited labels.

### Gaps
- No published likeness/identity reward model for stylized avatars was found.
- No numbers on how few pairs suffice for a reliable head on frozen embeddings in the face domain.
- **Unverified leads:** Coste et al. (ICLR 2024) reward-model ensembles for mitigating overoptimization; the PickScore/ImageReward training losses (InstructGPT/Bradley–Terry-style); few-shot metric-learning baselines (ProtoNets, linear probes on DINOv2). None were verified because searches ran out.

---

## Q5. How reliable are GPT-4V/Claude-style VLM judges for image similarity/likeness, and which human evaluation protocols (lineup, forced choice) to use

### Takeaway
MLLM judges agree with humans best in **pairwise comparison** and diverge more in **absolute scoring and ranking**. They show **position/selection bias**, hallucinate, and are unstable under irrelevant perturbations. On face-specific tasks, current MLLMs are weak (below 60% on FaceXBench). For likeness, VLM judges behave like **unfamiliar human viewers** who lean on hair and external cues. The best protocols are therefore order-swapped pairwise or lineup tasks with demographically matched foils, repeated samples, and a human calibration subset.

### Cited Findings
- **[NEW 2023–26] MLLM-as-a-Judge** (ICML 2024 oral) covers Scoring, Pair Comparison and Batch Ranking with GPT-4V, Gemini, Qwen and LLaVA. MLLMs show "remarkable human-like discernment in **Pair Comparison**" but "**significant divergence from human preferences in Scoring Evaluation and Batch Ranking**," with diverse biases, hallucinations and inconsistencies — [arXiv PDF](https://arxiv.org/pdf/2402.04788v3); [GitHub](https://github.com/Dongping-Chen/MLLM-Judge)
- **[NEW 2023–26] MM-JudgeBias** (2026): many MLLM judges fail to integrate key visual or textual cues and are unstable under semantically irrelevant perturbations — [arXiv PDF](https://arxiv.org/pdf/2604.18164)
- **[NEW 2023–26] Text LLM judges** (Zheng et al., "Judging LLM-as-a-judge with MT-Bench and Chatbot Arena," 2023): GPT-4 and human judges reach **over 80% agreement, the same level as human–human agreement** — [FastChat README](https://github.com/lm-sys/FastChat/blob/main/fastchat/llm_judge/README.md)
- **Position bias.** The judge favours the first or last candidate. Standard mitigation runs each pair twice with positions swapped (blog-level source) — [OneUptime blog](https://oneuptime.com/blog/post/2026-08-31-reduce-position-bias-pairwise-llm-evaluations/view). Systematic study: "Judging the Judges: A Systematic Study of Position Bias in LLM-as-a-Judge" — [arXiv 2406.07791](https://arxiv.org/html/2406.07791v9). Label-free debiasing by recalibrating the prediction distribution (CalibraEval, ACL 2025) — [ACL Anthology](https://preview.aclanthology.org/setup/2025.acl-long.808)
- **[NEW 2023–26] VLM perceptual similarity.** Frontier VLMs show a considerable gap to human consensus on similarity triplets — [alphaXiv](https://www.alphaxiv.org/abs/2607.18237.md). In a material-perception study, GPT-4o correlated best with humans on elementary dimensions (lightness, chromaticity, texture grain) but worse on abstract ones (snippet) — [PMC12178973](https://pmc.ncbi.nlm.nih.gov/articles/PMC12178973)
- **[NEW 2023–26] Face competence.** No MLLM exceeds 60% on FaceXBench's 14 face tasks; the best is 57.86% — [arXiv PDF](https://arxiv.org/pdf/2501.10360)
- **Human-protocol facts:**
  - Unfamiliar viewers are poor at telling identity across photos (2 people sorted as about 7) — [Glasgow ePrints](https://eprints.gla.ac.uk/67032)
  - Unfamiliar viewers weight external features as much as internal ones — [Ellis et al. 1979](https://moodle2.units.it/pluginfile.php/336418/mod_resource/content/0/Ellis79.pdf)
  - Changing hair between study and test cuts recognition — [Toseeb et al. 2012](https://pmc.ncbi.nlm.nih.gov/articles/PMC3312903/)
  - Caricature effects show up in **reaction time** with no accuracy difference — [Rhodes & Brennan 1987](https://researchconnect.stonybrook.edu/en/publications/identification-and-ratings-of-caricatures-implications-for-mental/)
- **Label noise.** In large T2I preference sets, a single human matches the consensus only **65.3% / 78.1%** of the time — [HPS v2 GitHub](https://github.com/tgxs002/HPSv2)

### Inferences (evaluation protocol recommendations)
- **Use forced-choice and lineup tasks, not absolute 1–10 scores.**
  - **Lineup / identification (primary metric):** show one CK3 render and a lineup of K photos (for example K=4–6), the true person plus foils matched on sex, ethnicity, age and **hair colour and length**. Score the hit rate. Chance is 1/K, which gives an interpretable "identifiability" number. Use the reverse direction too (one photo, K renders including other people's renders).
  - **2AFC likeness (secondary metric):** "which render looks more like this person?" for A/B system comparisons.
- **Hair-controlled conditions.** Run lineups with hair-matched foils, or with a hair-masked or bald crop condition. This separates internal-face likeness from hair likeness, because unfamiliar judges, VLM or human, over-weight external features. It also diagnoses the project's "hair is the biggest remaining cue."
- **Familiar vs unfamiliar raters.** Celebrities let you recruit familiar raters (naming or recognition tasks: "who is this?"). Familiar recognition is the gold standard for "looks just like them." For non-celebrities, unfamiliar matching tasks (render vs several *different* photos of the target, to handle within-person variability per Jenkins et al.) are the closest available proxy.
- **VLM judge hygiene:**
  - Always run both orders and count inconsistent verdicts as ties.
  - Use several samples or temperature draws and several prompt phrasings, and aggregate with Bradley–Terry.
  - Present the target with **two or more reference photos**.
  - Ask for pairwise or lineup answers, not 1–10 scores.
  - Randomize image filenames and labels.
  - Report position consistency as a metric.
  - Calibrate the VLM against a small human-labelled subset, and treat it as a feature or teacher for the learned critic rather than as ground truth.
  - Avoid asking a VLM to *name* celebrities: refusals and identification policies confound the measure. Use matching or lineup framing instead.
- **Report confidence intervals.** With about 143 identities, bootstrap over identities. A system difference smaller than about 5 percentage points in 2AFC is unlikely to be resolvable without many more raters or items. This is my estimate, not a sourced figure.
- **Display condition.** Evaluate at in-game portrait size as well as large size (see Q1).

### Gaps
- No study found that measures GPT-4V/Claude reliability specifically for **face likeness** between photos and stylized renders, including test–retest or position consistency for faces.
- No published human-lineup protocol designed for avatar likeness was found (not searched).
- **Unverified leads:** Wang et al. 2023, "Large Language Models are not Fair Evaluators" (position bias; balanced position calibration and multiple-evidence calibration); Zheng et al.'s handling of inconsistent swapped verdicts (the README did not confirm it); the Glasgow Face Matching Test as a rater-screening tool.
