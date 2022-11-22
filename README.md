# Text-Style-Generation

目标：对不同的目标用户可以生成不同的文本，可以是可控制的，简化的，风格迁移的等等（如果有其他的方式也欢迎补充）<br>
主要重点收集2020年后的顶会Paper，例如NLP: EACL, EMNLP, ACL, COLING, NAACL. ML&AI: IJCAL, ICML, AAAI, NeurIPS, ICLR (其他有影响力的Paper也可以补充，但最好是17年后的）<br>


## Controllable Text Generation

## Text Simplification

## Text Style Transfer
1. Expertise Style Transfer: A New Task Towards Better Communication between Experts and Laymen (Cao et al., ACL 2020)
    <br> * Paper: https://aclanthology.org/2020.acl-main.100.pdf
    <br> * Code: https://srhthu.github.io/expertise-style-transfer/
    
## Stylistic Text Generation

## Diffusion Related Model

### Diffusion in NLP
1. Step-unrolled denoising autoencoders for text generation (Savinov et al., 2021)
    <br> * Paper: https://arxiv.org/pdf/2112.06749.pdf

2. Structured Denoising Diffusion Models in Discrete State-Space (Austin et al., NeurIPS 2021)
    <br> * Paper: https://proceedings.neurips.cc/paper/2021/hash/958c530554f78bcd8e97125b70e6973d-Abstract.html

3. Argmax flows and multinomial diffusion: Towards non-autoregressive language models (Hoogeboom et al, 2021)
    <br> * Paper: https://arxiv.org/pdf/2102.05379v3.pdf

4. Diffusion-LM Improves Controllable Text Generation (Li et al, 2022)
    <br> * Paper: https://arxiv.org/pdf/2205.14217.pdf
    <br> * Code: https://github.com/XiangLi1999/Diffusion-LM

5. Analog Bits: Generating Discrete Data using Diffusion Models with Self-Conditioning (Chen et al., 2022)
    <br> * Paper: https://arxiv.org/pdf/2208.04202.pdf

6. DiffuSeq: Sequence to Sequence Text Generation with Diffusion Models (Gong et al., 2022)
    <br> * Paper: https://arxiv.org/pdf/2210.08933.pdf

7. CLIP-Diffusion-LM: Apply Diffusion Model on Image Captioning (Shitong Xu, 2022)
    <br> * Paper: https://arxiv.org/pdf/2210.04559.pdf
    <br> * Code: https://github.com/xu-shitong/diffusion-image-captioning

8. Categorical SDEs with Simplex Diffusion (Richemond et al., 2022)
    <br> * Paper: https://arxiv.org/pdf/2210.14784.pdf

9. Latent Diffusion Energy-Based Model for Interpretable Text Modeling (Yu et al., ICML 2022)
    <br> * Paper: https://proceedings.mlr.press/v162/yu22h.html
### Diffusion in other fields
1. Deep Unsupervised Learning using Non equilibrium Thermodynamics (Sohl-Dickstein et al., ICML 2015)
    <br> * Paper: https://openreview.net/forum?id=rkbVIoZdWH

2. Generative modeling by estimating gradients of the data distribution (Yang Song and Stefano Ermon, NeurIPS 2019)
    <br> * Paper: https://proceedings.neurips.cc/paper/2019/hash/3001ef257407d5a371a96dcd947c7d93-Abstract.html

3. Denoising Diffusion Implicit Models (Song et al., ICLR 2019)
    <br> * Paper: https://openreview.net/forum?id=St1giarCHLP

4. Score-Based Generative Modeling through Stochastic Differential Equations (Song et al., ICLR 2021)
    <br> * Paper: https://openreview.net/forum?id=PxTIG12RRHS

5. Adversarial score matching and improved sampling for image generation (Jolicoeur-Martineau et al., ICLR 2021)
    <br> * Paper: https://openreview.net/forum?id=eLfqMl3z3lq

6. Improved denoising diffusion probabilistic models (Nichol and Dhariwal, ICML 2021)
    <br> * Paper: https://proceedings.mlr.press/v139/nichol21a.html

7. Variational diffusion models (Kingma et al., NeurIPS 2021)
    <br> * Paper: https://proceedings.neurips.cc/paper/2021/hash/b578f2a52a0229873fefc2a4b06377fa-Abstract.html

8. Maximum likelihood training of score-based diffusion models (Song et al., NeurIPS 2021)
    <br> * Paper: https://proceedings.neurips.cc/paper/2021/hash/0a9fdbb17feb6ccb7ec405cfb85222c4-Abstract.html

9. A variational perspective on diffusion-based generative models and score matching (Huang et al., NeurIPS 2021)
    <br> * Paper: https://proceedings.neurips.cc/paper/2021/hash/c11abfd29e4d9b4d4b566b01114d8486-Abstract.html

10. Score-based generative modeling in latent space (Vahdat et al., NeurIPS 2021)
    <br> * Paper: https://proceedings.neurips.cc/paper/2021/hash/5dca4c6b9e244d24a30b4c45601d9720-Abstract.html

11. Learning gradient fields for molecular conformation generation (Shi et al., ICML 2021)
    <br> * Paper: http://proceedings.mlr.press/v139/shi21b.html

12. 3d shape generation and completion through point-voxel diffusion (Zhou et al., ICCV 2021)
    <br> * Paper: https://openaccess.thecvf.com/content/ICCV2021/html/Zhou_3D_Shape_Generation_and_Completion_Through_Point-Voxel_Diffusion_ICCV_2021_paper.html

13. Diffusion probabilistic models for 3d point cloud generation (Luo and Hu, CVPR 2021)
    <br> * Paper: https://openaccess.thecvf.com/content/CVPR2021/html/Luo_Diffusion_Probabilistic_Models_for_3D_Point_Cloud_Generation_CVPR_2021_paper.html

14. CSDI: Conditional score-based diffusion models for probabilistic time series imputation (Tashiro et al., NeurIPS 2021)
    <br> * Paper: https://proceedings.neurips.cc/paper/2021/hash/cfe8504bda37b575c70ee1a8276f3486-Abstract.html

15. Autoregressive denoising diffusion models for multivariate probabilistic time series forecasting (Rasul et al., ICML 2021)
    <br> * Paper: http://proceedings.mlr.press/v139/rasul21a.html

16. WaveGrad: Estimating Gradients for Waveform Generation (Chen et al., ICLR 2021)
    <br> * Paper: https://openreview.net/forum?id=NsMLjcFaO8O

17. DiffWave: A Versatile Diffusion Model for Audio Synthesis (Kong et al., ICLR 2021)
    <br> * Paper: https://openreview.net/forum?id=a-xFK8Ymz5J

18. GLIDE: Towards Photorealistic Image Generation and Editing with Text-Guided Diffusion Models (Nichol et al., 2021)
    <br> * Paper: https://arxiv.org/abs/2112.10741

19. Grad-TTS: A Diffusion Probabilistic Model for Text-to-Speech (Popov et al., ICML 2021)
    <br> * Paper: https://proceedings.mlr.press/v139/popov21a.html

20. Adversarial purification with score-based generative models (Yoon et al., ICML 2021)
    <br> * Paper: http://proceedings.mlr.press/v139/yoon21a.html

21. Learning gradient fields for molecular conformation generation (Shi et al., ICML 2021)
    <br> * Paper: http://proceedings.mlr.press/v139/shi21b.html

22. Predicting molecular conformation via dynamic graph score matching (Luo et al., NeurIPS 2021)
    <br> * Paper: https://proceedings.neurips.cc/paper/2021/hash/a45a1d12ee0fb7f1f872ab91da18f899-Abstract.html

23. A variational perspective on diffusion-based generative models and score matching (Huang et al., NeurIPS 2021)
    <br> * Paper: https://proceedings.neurips.cc/paper/2021/hash/c11abfd29e4d9b4d4b566b01114d8486-Abstract.html

24. Diffusion Models Beat GANs on Image Synthesis (Dhariwal and Nichol, NeurIPS 2021)
    <br> * Paper: https://proceedings.neurips.cc/paper/2021/hash/49ad23d1ec9fa4bd8d77d02681df5cfa-Abstract.html

25. Diffusion Normalizing Flow (Zhang and Chen, NeurIPS 2021)
    <br> * Paper: https://proceedings.neurips.cc/paper/2021/hash/876f1f9954de0aa402d91bb988d12cd4-Abstract.html

26. Learning Energy-Based Models by Diffusion Recovery Likelihood (Gao et al., ICLR 2021)
    <br> * Paper: https://openreview.net/forum?id=v_1Soh8QUNc

27. Come-closer-diffuse-faster: Accelerating conditional diffusion models for inverse problems through stochastic contraction (Chung et al., CVPR 2022)
    <br> * Paper: https://openaccess.thecvf.com/content/CVPR2022/html/Chung_Come-Closer-Diffuse-Faster_Accelerating_Conditional_Diffusion_Models_for_Inverse_Problems_Through_Stochastic_CVPR_2022_paper.html

28. Score-Based Generative Modeling with Critically-Damped Langevin Diffusion (Dockhorn et al., ICLR 2022)
    <br> * Paper: https://openreview.net/forum?id=CzceR82CYc

29. Elucidating the Design Space of Diffusion-Based Generative Models (Karras et al., NeurIPS 2022)
    <br> * Paper: https://arxiv.org/abs/2206.00364

30. DPM-Solver: A Fast ODE Solver for Diffusion Probabilistic Model Sampling in Around 10 Step (Lu et al., NeurIPS 2022)
    <br> * Paper: https://arxiv.org/abs/2206.00927

31. Pseudo Numerical Methods for Diffusion Models on Manifolds (Liu et al., ICLR 2022)
    <br> * Paper: https://openreview.net/forum?id=PlKWVd2yBkY
    <br> * Code: https://github.com/luping-liu/PNDM

32. GENIE: Higher-Order Denoising Diffusion Solvers (Dockhorn et al., NeurIPS 2022)
    <br> * Paper: https://arxiv.org/abs/2210.05475

33. Learning fast samplers for diffusion models by differentiating through sample quality (Watson et al., ICLR 2022)
    <br> * Paper: https://openreview.net/forum?id=VFBjuF8HEp

34. Progressive Distillation for Fast Sampling of Diffusion Models (Salimans and Ho, ICLR 2022)
    <br> * Paper: https://openreview.net/forum?id=TIdIXIpzhoI

35. Analytic-DPM: an Analytic Estimate of the Optimal Reverse Variance in Diffusion Probabilistic Models (Bao et al., ICLR 2022)
    <br> * Paper: https://openreview.net/forum?id=0xiJLKH-ufZ

36. Maximum Likelihood Training for Score-based Diffusion ODEs by High Order Denoising Score Matching (Lu et al., ICML 2022)
    <br> * Paper:https://proceedings.mlr.press/v162/lu22f.html

37. Hierarchical text-conditional image generation with clip latents (Ramesh et al., 2022)
    <br> * Paper: https://arxiv.org/abs/2204.06125

38. High-resolution image synthesis with latent diffusion models (Rombach et al., CVPR 2022)
    <br> * Paper: https://openaccess.thecvf.com/content/CVPR2022/html/Rombach_High-Resolution_Image_Synthesis_With_Latent_Diffusion_Models_CVPR_2022_paper.html
    <br> * Code: https://github.com/CompVis/stable-diffusion

39. GeoDiff: A Geometric Diffusion Model for Molecular Conformation Generation (Xu et al., ICLR 2022)
    <br> * Paper: https://openreview.net/forum?id=PzcvxEMzvQC

40. Score-based Generative Modeling of Graphs via the System of Stochastic Differential Equations (Jo et al., ICML 2022)
    <br> * Paper: https://proceedings.mlr.press/v162/jo22a.html

41. Vector quantized diffusion model for text-to-image synthesis (Gu et al., CVPR 2022)
    <br> * Paper: https://openaccess.thecvf.com/content/CVPR2022/html/Gu_Vector_Quantized_Diffusion_Model_for_Text-to-Image_Synthesis_CVPR_2022_paper.html

42. SRDiff: Single Image Super-Resolution with Diffusion Probabilistic Models (Li et al., ACM 2022)
    <br> * Paper: https://www.sciencedirect.com/science/article/abs/pii/S0925231222000522

43. Repaint: Inpainting using denoising diffusion probabilistic models (Lugmayr et al., CVPR 2022)
    <br> * Paper: https://openaccess.thecvf.com/content/CVPR2022/html/Lugmayr_RePaint_Inpainting_Using_Denoising_Diffusion_Probabilistic_Models_CVPR_2022_paper.html

44. Palette: Image-to-image diffusion models (Saharia et al., ACM 2022)
    <br> * Paper: https://openreview.net/forum?id=FPGs276lUeq

45. Generative Visual Prompt: Unifying Distributional Control of Pre-Trained Generative Models (Wu et al., NeurIPS 2022)
    <br> * Paper: https://arxiv.org/abs/2209.06970

46. Cascaded Diffusion Models for High Fidelity Image Generation (Ho et al., JMLR 2022)
    <br> * Paper: https://www.jmlr.org/papers/v23/21-0635.html

47. Solving Inverse Problems in Medical Imaging with Score-Based Generative Models (Song et al., ICLR 2022)
    <br> * Paper: https://openreview.net/forum?id=vaRCHVj0uGI

48. Sdedit: Guided image synthesis and editing with stochastic differential equations (Meng et al., ICLR 2022)
    <br> * Paper: https://arxiv.org/abs/2108.01073
    <br> * Code: https://github.com/ermongroup/SDEdit

49. Label-Efficient Semantic Segmentation with Diffusion Models (Baranchuk et al., ICLR 2022)
    <br> * Paper: https://openreview.net/forum?id=SlxSY2UZQT

50. Diffusion models as plug-and-play priors (Graikos et al., NeurIPS 2022)
    <br> * Paper: https://arxiv.org/abs/2206.09012

51. A Conditional Point Diffusion-Refinement Paradigm for 3D Point Cloud Completion (Lyu et al., ICLR 2022)
    <br> * Paper: https://openreview.net/forum?id=wqD6TfbYkrn

52. LION: Latent Point Diffusion Models for 3D Shape Generation (Zeng et al., NeurIPS 2022)
    <br> * Paper: https://arxiv.org/abs/2210.06978

53. Neural Markov Controlled SDE: Stochastic Optimization for Continuous-Time Data (Park et al., ICLR 2022)
    <br> * Paper: https://openreview.net/forum?id=7DI6op61AY

54. Photorealistic Text-to-Image Diffusion Models with Deep Language Understanding (Saharia et al., 2022)
    <br> * Paper: https://arxiv.org/abs/2205.11487

55. Blended diffusion for text-driven editing of natural images (Avrahami et al., CVPR 2022)
    <br> * Paper: https://openaccess.thecvf.com/content/CVPR2022/html/Avrahami_Blended_Diffusion_for_Text-Driven_Editing_of_Natural_Images_CVPR_2022_paper.html

56. DiffusionCLIP: Text-Guided Diffusion Models for Robust Image Manipulation (Kim et al., CVPR 2022)
    <br> * Paper: https://openaccess.thecvf.com/content/CVPR2022/html/Kim_DiffusionCLIP_Text-Guided_Diffusion_Models_for_Robust_Image_Manipulation_CVPR_2022_paper.html

57. Zero-Shot Voice Conditioning for Denoising Diffusion TTS Models (Levkovitch et al., Interspeech 2022)
    <br> * Paper: https://arxiv.org/abs/2206.02246

58. ProDiff: Progressive Fast Diffusion Model For High-Quality Text-to-Speech (Huang et al., ACM 2022)
    <br> * Paper: https://arxiv.org/abs/2207.06389

59. Diffusion Models for Adversarial Purification (Nie et al. ICML 2022)
    <br> * Paper: https://arxiv.org/abs/2205.07460

60. Equivariant Diffusion for Molecule Generation in 3D (Hoogeboom et al., ICML 2022)
    <br> * Paper: https://proceedings.mlr.press/v162/hoogeboom22a.html

61. GeoDiff: A Geometric Diffusion Model for Molecular Conformation Generation (Xu et al., ICLR 2022) 
    <br> * Paper: https://openreview.net/forum?id=PzcvxEMzvQC

62. Crystal Diffusion Variational Autoencoder for Periodic Material Generation (Xie et al., ICLR 2022)
    <br> * Paper: https://arxiv.org/abs/2110.06197
    <br> * Code: https://github.com/txie-93/cdvae

63. Solving Inverse Problems in Medical Imaging with Score-Based Generative Models (Song et al., ICLR 2022)
    <br> * Paper: https://openreview.net/forum?id=vaRCHVj0uGI

64. Towards performant and reliable undersampled MR reconstruction via diffusion model sampling (Peng et al., MICCAI 2022)
    <br> * Paper: https://arxiv.org/pdf/2203.04292.pdf

65. Come-closer-diffuse-faster: Accelerating conditional diffusion models for inverse problems through stochastic contraction (Chung et al., CVPR 2022)
    <br> * Paper: https://openaccess.thecvf.com/content/CVPR2022/papers/Chung_Come-Closer-Diffuse-Faster_Accelerating_Conditional_Diffusion_Models_for_Inverse_Problems_Through_Stochastic_CVPR_2022_paper.pdf

66. Tackling the generative learning trilemma with denoising diffusion gans (Xiao et al., ICLR 2022)
    <br> * Paper: https://openreview.net/forum?id=JprM0p-q0Co

67. Autoregressive Diffusion Models (Hoogeboom et al., ICLR 2022)
    <br> * Paper: https://openreview.net/forum?id=Lm8T39vLDTE

68. Latent Diffusion Energy-Based Model for Interpretable Text Modeling (Yu et al., ICML 2022)
    <br> * Paper: https://proceedings.mlr.press/v162/yu22h.html

### Controllable Text Generation

1. Hafez: an Interactive Poetry Generation System (Ghazvininejad et al., ACL 2017)
    <br> * Paper: https://aclanthology.org/P17-4008.pdf

2. Toward Controlled Generation of Text (Hu et al., ICML 2017)
    <br> * Paper: http://proceedings.mlr.press/v70/hu17e.html

3. Controlling Linguistic Style Aspects in Neural Language Generation (Ficler et al., ACL Workshop 2017)
    <br> * Paper: https://aclanthology.org/W17-4912/

4. Lexically constrained decoding for sequence generation using grid beam search (Hokamp et al., ACL 2017)
    <br> * Paper: https://aclanthology.org/P17-1141.pdf

5. Affect-LM: A Neural Language Model for Customizable Affective Text Generation (Ghosh et al., ACL 2017)
    <br> * Paper: https://aclanthology.org/P17-1059/

6. Guided Open Vocabulary Image Captioning with Constrained Beam Search (Anderson et al., EMNLP 2017)
    <br> * Paper: https://aclanthology.org/D17-1098/?ref=https://githubhelp.com

7. Fast Lexically Constrained Decoding with Dynamic Beam Allocation for Neural Machine Translation (Post and Vilar, NAACL 2018)
    <br> * Paper: https://aclanthology.org/N18-1119/

8. Towards Controllable Story Generation (Peng et al., ACL Workshop 2018)
    <br> * Paper: https://aclanthology.org/W18-1505.pdf

9. Controllable Abstractive Summarization (Fan et al., ACL Workshop 2018)
    <br> * Paper: https://aclanthology.org/W18-2706/

10. Content preserving text generation with attribute controls (Logeswaren et al., NeurIPS 2018)
    <br> * Paper: https://proceedings.neurips.cc/paper/2018/hash/7cf64379eb6f29a4d25c4b6a2df713e4-Abstract.html

11. Polite Dialogue Generation Without Parallel Data (Niu and Bansal, TACL 2018)
    <br> * Paper: https://aclanthology.org/Q18-1027/

12. Latent Constraints: Learning to Generate Conditionally from Unconditional Generative Models (Engel et al., ICLR 2018)
    <br> * Paper: https://openreview.net/forum?id=Sy8XvGb0-

13. Plug and play language models: A simple approach to controlled text generation (Dathathri et al., ICLR 2019)
    <br> * Paper: https://arxiv.org/abs/1912.02164

14. CTRL: A Conditional Transformer Language Model for Controllable Generation (Keskar et al., 2019)
    <br> * Paper: https://arxiv.org/abs/1909.05858

15. What makes a good conversation? How controllable attributes affect human judgment (See et al., NAACL 2019)
    <br> * Paper: http://aclanthology.lst.uni-saarland.de/N19-1170.pdf

16. CGMH: Constrained Sentence Generation by Metropolis-Hastings Sampling (Miao et al., AAAI 2019)
    <br> * Paper: https://arxiv.org/pdf/1811.10996.pdf

17. Topic-Guided Variational Autoencoders for Text Generation (Wang et al., NAACL 2019)
    <br> * Paper: https://aclanthology.org/N19-1015/

18. GeDi: Generative Discriminator Guided Sequence Generation (Krause et al., 2020)
    <br> * Paper: https://arxiv.org/abs/2009.06367

19. Gradient-guided unsupervised lexically constrained text generation (Sha, EMNLP 2020)
    <br> * Paper: https://aclanthology.org/2020.emnlp-main.701.pdf

20. Plug-and-Play Conversational Models (Madotto et al., EMNLP 2020)
    <br> * Paper: https://aclanthology.org/2020.findings-emnlp.219/

21. Exploring Controllable Text Generation Techniques (Prabhumoye et al., COLING 2020)
    <br> * Paper: https://aclanthology.org/2020.coling-main.1/

22. Towards Controllable Biases in Language Generation (Sheng et al., EMNLP 2020)
    <br> * Paper: https://aclanthology.org/2020.findings-emnlp.291/

23. A distributional approach to controlled text generation (Khalifa et al., ICLR 2020)
    <br> * Paper: https://arxiv.org/pdf/2012.11635.pdf

24. Controllable Text Generation with Focused Variation (Shu et al., ACL 2020)
    <br> * Paper: https://aclanthology.org/2020.findings-emnlp.339/

25. Rigid Formats Controlled Text Generation (Li et al., ACL 2020)
    <br> * Paper: https://aclanthology.org/2020.acl-main.68/

26. POINTER: Constrained Progressive Text Generation via Insertion-based Generative Pre-training (Zhang et al., EMNLP 2020)
    <br> * Paper: https://arxiv.org/abs/2005.00558

27. PowerTransformer: Unsupervised controllable revision for biased language correction (Ma et al., EMNLP 2020)
    <br> * Paper: https://aclanthology.org/2020.emnlp-main.602/

28. MixingBoard: a Knowledgeable Stylized Integrated Text Generation Platform (Gao et al., ACL 2020)
    <br> * Paper: https://aclanthology.org/2020.acl-demos.26/

29. Pre-train and Plug-in: Flexible Conditional Text Generation with Variational Auto-Encoders (Duan et al., ACL 2020)
    <br> * Paper: https://arxiv.org/pdf/1911.03882.pdf

30. Plug and Play Autoencoders for Conditional Text Generation (Mai et al., EMNLP 2020)
    <br> * Paper: https://arxiv.org/pdf/2010.02983.pdf

31. DEXPERTS: Decoding-Time Controlled Text Generation with Experts and Anti-Experts (Liu et al., ACL 2021)
    <br> * Paper: https://aclanthology.org/2021.acl-long.522.pdf

32. FUDGE: Controlled Text Generation With Future Discriminators (Yang and Klein, NAACL 2021)
    <br> * Paper: https://aclanthology.org/2021.naacl-main.276/

33. Controlled Text Generation as Continuous Optimization with Multiple Constraints (Kumar et al., NeurIPS 2021)
    <br> * Paper: https://openreview.net/forum?id=kTy7bbm-4I4

34. Neurologic decoding:(un) supervised neural text generation with predicate logic constraints (Lu et al., NAACL 2021)
    <br> * Paper: https://aclanthology.org/2021.naacl-main.339.pdf

35. A Causal Lens for Controllable Text Generation (Hu and Li, NeurIPS 2021)
    <br> * Paper: https://proceedings.neurips.cc/paper/2021/hash/d0f5edad9ac19abed9e235c0fe0aa59f-Abstract.html

36. A Controllable Model of Grounded Response Generation (Wu et al., AAAI 2021)
    <br> * Paper: https://ojs.aaai.org/index.php/AAAI/article/view/17658

37. Cocon: A self-supervised approach for controlled text generation (Chan et al., ICLR 2021)
    <br> * Paper: https://openreview.net/forum?id=VD_ozqvBy4W

38. Generate your counterfactuals: Towards controlled counterfactual generation for text (Madaan et al., AAAI 2021)
    <br> * Paper: https://ojs.aaai.org/index.php/AAAI/article/view/17594

39. Mix and Match: Learning-free Controllable Text Generationusing Energy Language Models (Mireshghallah et al., ACL 2022)
    <br> * Paper: https://aclanthology.org/2022.acl-long.31/


### Style Transfer

1. Style Transfer from Non-Parallel Text by Cross-Alignment (Shen et al., NeurIPS 2017)
    <br> * Paper: https://arxiv.org/pdf/1705.09655.pdf

2. Sequence to Better Sequence: Continuous Revision of Combinatorial Structures (Mueller et al., ICML 2017)
    <br> * Paper: http://proceedings.mlr.press/v70/mueller17a/mueller17a.pdf

3. Delete, Retrieve, Generate: A Simple Approach to Sentiment and Style Transfer (Li et al., NAACL 2018)
    <br> * Paper: https://openreview.net/forum?id=B1bTGQ-ubS

4. Style Transfer in Text: Exploration and Evaluation (Fu et al., AAAI 2018)
    <br> * Paper: https://ojs.aaai.org/index.php/AAAI/article/view/11330

5. Style Transfer Through Back-Translation (Prabhumoye et al., ACL 2018)
    <br> * Paper: https://aclanthology.org/P18-1080/?ref=https://githubhelp.com

6. Unsupervised Text Style Transfer using Language Models as Discriminators (Yang et al., NeurIPS 2018)
    <br> * Paper: https://proceedings.neurips.cc/paper/2018/hash/398475c83b47075e8897a083e97eb9f0-Abstract.html

7. Adversarially Regularized Autoencoders (Zhao et al., ICML 2018)
    <br> * Paper: https://arxiv.org/pdf/1706.04223.pdf

8. Unpaired Sentiment-to-Sentiment Translation: A Cycled Reinforcement Learning Approach (Xu et al., ACL 2018)
    <br> * Paper: https://arxiv.org/pdf/1805.05181.pdf

9. Adversarial Text Generation via Feature-Mover's Distance (Chen et al., NeurIPS 2018)
    <br> * Paper: https://arxiv.org/pdf/1809.06297.pdf

10. Dear Sir or Madam, May I introduce the YAFC Dataset: Corpus, Benchmarks and Metrics for Formality Style Transfer (Rao and Tetreault, NAACL 2018)
    <br> * Paper: https://arxiv.org/pdf/1803.06535.pdf

11. Fighting Offensive Language on Social Media with Unsupervised Text Style Transfer (Santos et al., ACL 2018)
    <br> * Paper: https://arxiv.org/pdf/1805.07685.pdf

12. Style Transformer: Unpaired Text Style Transfer without Disentangled Latent Representation (Dai et al., ACL 2019)
    <br> * Paper: https://arxiv.org/pdf/1905.05621.pdf

13. Transforming delete, retrieve, generate approach for controlled text style transfer (Sudhakar et al., EMNLP 2019)
    <br> * Paper: https://aclanthology.org/D19-1322/

14. Disentangled representation learning for non-parallel text style transfer (John et al., ACL 2019)
    <br> * Paper: https://aclanthology.org/P19-1041/?ref=https://githubhelp.com

15. Multiple-attribute text rewriting (Lample et al., ICLR 2019)
    <br> * Paper: https://openreview.net/forum?id=H1g2NhC5KQ

16. A dual reinforcement learning framework for unsupervised text style transfer (Luo et al., IJCAI 2019)
    <br> * Paper: https://openreview.net/forum?id=oJvtywo_YP0

17. Politeness Transfer: A Tag and Generate Approach (Madaan et al., ACL 2020)
    <br> * Paper: https://arxiv.org/abs/2004.14257

18. Contextual text style transfer (Cheng et al., EMNLP 2020)
    <br> * Paper: https://aclanthology.org/2020.findings-emnlp.263/

19. Reformulating unsupervised style transfer as paraphrase generation (Krishna et al., EMNLP 2020)
    <br> * Paper: https://aclanthology.org/2020.emnlp-main.55/

20. A probabilistic formulation of unsupervised text style transfer (He et al., ICLR 2020)
    <br> * Paper: https://openreview.net/forum?id=HJlA0C4tPS

21. Revision in Continuous Space: Unsupervised Text Style Transfer without Adversarial Learning (Liu et al., AAAI 2020)
    <br> * Paper: https://arxiv.org/pdf/1905.12304.pdf

22. On Variational Learning of Controllable Representations for Text without Supervision (Xu et al., ICML 2020)
    <br> * Paper: https://arxiv.org/pdf/1905.11975.pdf

23. Exploring Contextual Word-level Style Relevance for Unsupervised Style Transfer (Zhou et al., ACL 2020)
    <br> * Paper: https://arxiv.org/pdf/2005.02049.pdf

24. Text Style Transfer via Learning Style Instance Supported Latent Space (Yi et al., IJCAI 2020)
    <br> * Paper: https://www.ijcai.org/Proceedings/2020/0526.pdf

25. Cycle-Consistent Adversarial Autoencoders for Unsupervised Text Style Transfer (Huang et al., COLING 2020)
    <br> * Paper: https://arxiv.org/pdf/2010.00735.pdf

26. How Positive Are You: Text Style Transfer using Adaptive Style Embedding (Kim and Sohn, COLING 2020)
    <br> * Paper: https://aclanthology.org/2020.coling-main.191.pdf

27. Unsupervised Text Style Transfer with Padded Masked Language Models (Malmi et al., EMNLP 2020)
    <br> * Paper: https://arxiv.org/pdf/2010.01054.pdf

28. Unsupervised Text Generation by Learning from Search (Li et al., NeurIPS 2020)
    <br> * Paper: https://papers.nips.cc/paper/2020/file/7a677bb4477ae2dd371add568dd19e23-Paper.pdf

29. DGST: a Dual-Generator Network for Text Style Transfer (Li et al., EMNLP 2020)
    <br> * Paper: https://arxiv.org/pdf/2010.14557.pdf

30. Formality Style Transfer with Shared Latent Space (Wang et al., COLING 2020)
    <br> * Paper: https://aclanthology.org/2020.coling-main.203.pdf

31. Parallel Data Augmentation for Formality Style Transfer (Zhang et al., ACL 2020)
    <br> * Paper: https://arxiv.org/pdf/2005.07522.pdf

32. Expertise Style Transfer: A New Task Towards Better Communication between Experts and Laymen (Cao et al., ACL 2020)
    <br> * Paper: https://arxiv.org/pdf/2005.00701.pdf

33. Adapting Language Models for Non-Parallel Author-Stylized Rewriting (Syed et al., AAAI 2020)
    <br> * Paper: https://arxiv.org/pdf/1909.09962.pdf

34. Hooks in the Headline: Learning to Generate Headlines with Controlled Styles (Jin et al., ACL 2020)
    <br> * Paper: https://arxiv.org/pdf/2004.01980.pdf

35. Exploring Contextual Word-level Style Relevance for Unsupervised Style Transfer (Zhou et al., ACL 2020)
    <br> * Paper: https://arxiv.org/pdf/2005.02049.pdf

36. Style versus Content: A distinction without a (learnable) difference? (Jafaritazehjani et al., COLING 2020)
    <br> * Paper: https://aclanthology.org/2020.coling-main.197.pdf

37. Towards A Friendly Online Community: An Unsupervised Style Transfer Framework for Profanity Redaction (Tran et al., COLING 2020)
    <br> * Paper: https://arxiv.org/pdf/2011.00403.pdf

38. Style Pooling: Automatic Text Style Obfuscation for Improved Classification Fairness (Mireshghallah et al., EMNLP 2021)
    <br> * Paper: https://aclanthology.org/2021.emnlp-main.152/

39. On Learning Text Style Transfer with Direct Rewards (Liu et al., NAACL 2021)
    <br> * Paper: https://arxiv.org/pdf/2010.12771.pdf

40. Multi-Style Transfer with Discriminative Feedback on Disjoint Corpus (Goyal et al., NAACL 2021)
    <br> * Paper: https://arxiv.org/pdf/2010.11578.pdf

41. A Hierarchical VAE for Calibrating Attributes while Generating Text using Normalizing Flow (Samanta et al., ACL 2021)
    <br> * Paper: https://aclanthology.org/2021.acl-long.187.pdf

42. Enhancing Content Preservation in Text Style Transfer Using Reverse Attention and Conditional Layer Normalization (Lee et al., ACL 2021)
    <br> * Paper: https://arxiv.org/pdf/2108.00449.pdf

43. Counterfactuals to Control Latent Disentangled Text Representations for Style Transfer (Nangi et al., ACL 2021)
    <br> * Paper: https://aclanthology.org/2021.acl-short.7.pdf

44. TextSETTR: Few-Shot Text Style Extraction and Tunable Targeted Restyling (Riley et al., ACL 2021)
    <br> * Paper: https://aclanthology.org/2021.acl-long.293.pdf

45. LEWIS: Levenshtein Editing for Unsupervised Text Style Transfer (Reid and Zhong, ACL Findings 2021)
    <br> * Paper: https://arxiv.org/pdf/2105.08206.pdf

46. NAST: A Non-Autoregressive Generator with Word Alignment for Unsupervised Text Style Transfer (Huang et al., ACL Findings 2021)
    <br> * Paper: https://arxiv.org/pdf/2106.02210.pdf

47. Olá, Bonjour, Salve! XFORMAL: A Benchmark for Multilingual Formality Style Transfer (Briakou et al., NAACL 2021)
    <br> * Paper: https://aclanthology.org/2021.naacl-main.256.pdf

48. Improving Formality Style Transfer with Context-Aware Rule Injection (Yao et al., ACL 2021)
    <br> * Paper: https://arxiv.org/pdf/2106.00210.pdf

49. Thank you BART! Rewarding Pre-Trained Models Improves Formality Style Transfer (Lai et al., ACL 2021)
    <br> * Paper: https://aclanthology.org/2021.acl-short.62.pdf

50. Controllable Text Simplification with Explicit Paraphrasing (Maddela et al., NAACL 2021)
    <br> * Paper: https://arxiv.org/pdf/2010.11004.pdf

51. Paragraph-level Simplification of Medical Texts (Devaraj et al., NAACL 2021)
    <br> * Paper: https://aclanthology.org/2021.naacl-main.395.pdf

52. Towards Modeling the Style of Translators in Neural Machine Translation (Wang et al., NAACL 2021)
    <br> * Paper: https://aclanthology.org/2021.naacl-main.94.pdf

53. Inference Time Style Control for Summarization (Cao et al., NAACL 2021)
    <br> * Paper: https://arxiv.org/pdf/2104.01724.pdf

54. Exploring Non-Autoregressive Text Style Transfer (Ma and Li, EMNLP 2021)
    <br> * Paper: https://aclanthology.org/2021.emnlp-main.730.pdf

55. Generic resources are what you need: Style transfer tasks without task-specific parallel training data (Lai et al., EMNLP 2021)
    <br> * Paper: https://arxiv.org/pdf/2109.04543.pdf

56. Transductive Learning for Unsupervised Text Style Transfer (Xiao et al., EMNLP 2021)
    <br> * Paper: https://arxiv.org/pdf/2109.07812.pdf

57. Collaborative Learning of Bidirectional Decoders for Unsupervised Text Style Transfer (Ma et al., EMNLP 2021)
    <br> * Paper: https://aclanthology.org/2021.emnlp-main.729.pdf

58. STYLEPTB: A Compositional Benchmark for Fine-grained Controllable Text Style Transfer (Lyu et al., NAACL 2021)
    <br> * Paper: https://arxiv.org/pdf/2104.05196.pdf

59. Rethinking Sentiment Style Transfer (Yu et al., EMNLP 2021)
    <br> * Paper: https://aclanthology.org/2021.findings-emnlp.135.pdf

60. Evaluating the Evaluation Metrics for Style Transfer: A Case Study in Multilingual Formality Transfer (Briakou et al., EMNLP 2021)
    <br> * Paper: https://aclanthology.org/2021.emnlp-main.100.pdf

61. Does It Capture STEL? A Modular, Similarity-based Linguistic Style Evaluation Framework (Wegmann et al., EMNLP 2021)
    <br> * Paper: https://aclanthology.org/2021.emnlp-main.569.pdf

62. MISS: An Assistant for Multi-Style Simultaneous Translation (Li et al., EMNLP 2021)
    <br> * Paper: https://aclanthology.org/2021.emnlp-demo.1.pdf

63. A Recipe for Arbitrary Text Style Transfer with Large Language Models (Reif et al., ACL 2022)
    <br> * Paper: https://aclanthology.org/2022.acl-short.94/

64. So Different Yet So Alike! Constrained Unsupervised Text Style Transfer (Kashyap et al., ACL 2022)
    <br> * Paper: https://aclanthology.org/2022.acl-long.32.pdf

65. Semi-Supervised Formality Style Transfer with Consistency Training (Liu et al., ACL 2022)
    <br> * Paper: https://aclanthology.org/2022.acl-long.321.pdf




## Dataset


## Acknowledgement
We thank [Yang et al.](https://arxiv.org/pdf/2209.00796.pdf) and [Jin et al.](https://arxiv.org/pdf/2011.00416.pdf) for their comprehensive survey on recent development of diffusion models and style transfer. Their work can be found in the following github repositories:

[Diffusion Models: A Comprehensive Survey of Methods and Applications](https://github.com/YangLing0818/Diffusion-Models-Papers-Survey-Taxonomy)

[Deep Learning for Text Style Transfer: A Survey](https://github.com/zhijing-jin/Text_Style_Transfer_Survey)