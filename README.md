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
































































## Dataset
