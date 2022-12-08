# Text-Style-Generation

目标：对不同的目标用户可以生成不同的文本，可以是可控制的，简化的，风格迁移的等等（如果有其他的方式也欢迎补充）<br>
主要重点收集2020年后的顶会Paper，例如NLP: EACL, EMNLP, ACL, COLING, NAACL. ML&AI: IJCAL, ICML, AAAI, NeurIPS, ICLR (其他有影响力的Paper也可以补充，但最好是17年后的）<br>


## Controllable Text Generation

1. Hafez: an Interactive Poetry Generation System (Ghazvininejad et al., ACL 2017)
    <br> * Paper: https://aclanthology.org/P17-4008.pdf 
    <br> * Approach: RNN+FSA
2. Toward Controlled Generation of Text (Hu et al., ICML 2017)
    <br> * Paper: http://proceedings.mlr.press/v70/hu17e.html
    <br> * Code: https://github.com/asyml/texar
    <br> * Approach: VAE+holistic feature discriminator (LSTM)
3. Controlling Linguistic Style Aspects in Neural Language Generation (Ficler et al., ACL Workshop 2017)
    <br> * Paper: https://aclanthology.org/W17-4912/
    <br> * Approach: Conditional LSTM model
4. Lexically constrained decoding for sequence generation using grid beam search (Hokamp et al., ACL 2017)
    <br> * Paper: https://aclanthology.org/P17-1141.pdf
    <br> * Code: https://github.com/chrishokamp/constrained_decoding
    <br> * Approach: Grid beam search as decoding that can be paired with any autoregressively generative model
5. Affect-LM: A Neural Language Model for Customizable Affective Text Generation (Ghosh et al., ACL 2017)
    <br> * Paper: https://aclanthology.org/P17-1059/
    <br> * Approach: LSTM with additional energy term as condition
6. Guided Open Vocabulary Image Captioning with Constrained Beam Search (Anderson et al., EMNLP 2017)
    <br> * Paper: https://aclanthology.org/D17-1098/?ref=https://githubhelp.com
    <br> * Code: https://github.com/nocaps-org/updown-baseline
    <br> * Approach: Constrained beam search as decoding that can be paired with any autoregressively generative model
7. Fast Lexically Constrained Decoding with Dynamic Beam Allocation for Neural Machine Translation (Post and Vilar, NAACL 2018)
    <br> * Paper: https://aclanthology.org/N18-1119/
    <br> * Approach: Dynamic beam allocation as decoding that can be paired with any autoregressively generative model
8. Towards Controllable Story Generation (Peng et al., ACL Workshop 2018)
    <br> * Paper: https://aclanthology.org/W18-1505.pdf 
    <br> * Approach: Conditional LSTM / seq2seq LSTM model from storyline to story
9. Controllable Abstractive Summarization (Fan et al., ACL Workshop 2018)
    <br> * Paper: https://aclanthology.org/W18-2706/
    <br> * Approach: Convolutional seq2seq model with attention, specific prefix to control certain attributes
10. Content preserving text generation with attribute controls (Logeswaren et al., NeurIPS 2018)
    <br> * Paper: https://proceedings.neurips.cc/paper/2018/hash/7cf64379eb6f29a4d25c4b6a2df713e4-Abstract.html
    <br> * Code: https://github.com/hwijeen/CPTG
    <br> * Approach: RNN-based encoder-decoder model trained with auto-encoding, back-translation and adversarial loss
11. Polite Dialogue Generation Without Parallel Data (Niu and Bansal, TACL 2018)
    <br> * Paper: https://aclanthology.org/Q18-1027/
    <br> * Code: https://github.com/WolfNiu/polite-dialogue-generation
    <br> * Approach: RNN-based fusion/label fine-tuning/RL models
12. Latent Constraints: Learning to Generate Conditionally from Unconditional Generative Models (Engel et al., ICLR 2018)
    <br> * Paper: https://openreview.net/forum?id=Sy8XvGb0-
    <br> * Approach: Gradient-based optimization/amortized generator (LSTM)
13. Plug and play language models: A simple approach to controlled text generation (Dathathri et al., ICLR 2019)
    <br> * Paper: https://arxiv.org/abs/1912.02164
    <br> * Code: https://github.com/uber-research/PPLM
    <br> * Approach: Iterative gradient-based optimization of history matrix with transformer generator
14. CTRL: A Conditional Transformer Language Model for Controllable Generation (Keskar et al., 2019)
    <br> * Paper: https://arxiv.org/abs/1909.05858
    <br> * Code: https://github.com/PaddlePaddle/PaddleNLP/tree/develop/paddlenlp/transformers/ctrl
    <br> * Approach: A large scale pre-trained conditional transformer conditioned on selected control codes
15. What makes a good conversation? How controllable attributes affect human judgment (See et al., NAACL 2019)
    <br> * Paper: http://aclanthology.lst.uni-saarland.de/N19-1170.pdf
    <br> * Code: https://github.com/HMJiangGatech/dialogue_ope_data
    <br> * Approach: dialogue-level Bi-LSTM based conditional training/ word-level weighted decoding
16. CGMH: Constrained Sentence Generation by Metropolis-Hastings Sampling (Miao et al., AAAI 2019)
    <br> * Paper: https://arxiv.org/pdf/1811.10996.pdf
    <br> * Code: https://github.com/NingMiao/CGMH
    <br> * Approach: RNN based, a specially designed MCMC approach for decoding
17. Topic-Guided Variational Autoencoders for Text Generation (Wang et al., NAACL 2019)
    <br> * Paper: https://aclanthology.org/N19-1015/
    <br> * Approach: VAE with a neural topic model (RNN based)
18. GeDi: Generative Discriminator Guided Sequence Generation (Krause et al., 2020)
    <br> * Paper: https://arxiv.org/abs/2009.06367
    <br> * Code: https://github.com/salesforce/GeDi
    <br> * Approach: use a small class conditional LM to assist a large generative model (GPT-2) to make conditional next token prediction
19. Gradient-guided unsupervised lexically constrained text generation (Sha, EMNLP 2020)
    <br> * Paper: https://aclanthology.org/2020.emnlp-main.701.pdf
    <br> * Approach: Edit the output of a generator (LSTM) following the guidance of the gradient of a loss function (lexical constraint, semantic similarity, fluency) w.r.t word representations
20. Plug-and-Play Conversational Models (Madotto et al., EMNLP 2020)
    <br> * Paper: https://aclanthology.org/2020.findings-emnlp.219/
    <br> * Code: https://github.com/andreamad8/PPCM
    <br> * Approach: First generate a pseudo dataset with certain attributes using a plug-and-play transformer, then train a adapter for each attribute for low inference latency
21. Exploring Controllable Text Generation Techniques (Prabhumoye et al., COLING 2020)
    <br> * Paper: https://aclanthology.org/2020.coling-main.1/
    <br> * Approach: A literature overview
22. Towards Controllable Biases in Language Generation (Sheng et al., EMNLP 2020)
    <br> * Paper: https://aclanthology.org/2020.findings-emnlp.291/
    <br> * Code: https://github.com/ewsheng/nlg-bias
    <br> * Approach: [Prompt]Design objectives to search triggers that can induce and mitigate biases in genetated texts
23. A distributional approach to controlled text generation (Khalifa et al., ICLR 2020)
    <br> * Paper: https://arxiv.org/pdf/2012.11635.pdf
    <br> * Code: https://github.com/naver/gdc
    <br> * Approach: First train an energy-based model that satisfies hard/distributional conditions, then train a controlled autoregressive lm through an adaptive distributional variant of Policy Gradient
24. Controllable Text Generation with Focused Variation (Shu et al., ACL 2020)
    <br> * Paper: https://aclanthology.org/2020.findings-emnlp.339/
    <br> * Approach: A VAE model with content/style encoders. Text encoder and decoder are LSTM
25. Rigid Formats Controlled Text Generation (Li et al., ACL 2020)
    <br> * Paper: https://aclanthology.org/2020.acl-main.68/
    <br> * Code: https://github.com/lipiji/SongNet
    <br> * Approach: Transformer based decoder, format-related embeddings introduced, a global attention mechanism is applied during training to capture future information
26. POINTER: Constrained Progressive Text Generation via Insertion-based Generative Pre-training (Zhang et al., EMNLP 2020)
    <br> * Paper: https://arxiv.org/abs/2005.00558
    <br> * Code: https://github.com/dreasysnail/POINTER
    <br> * Approach: An insertion-based transformer pre-trained on large-scale corpus and fine-tuned on downstream hard lexical constrained text generation tasks
27. PowerTransformer: Unsupervised controllable revision for biased language correction (Ma et al., EMNLP 2020)
    <br> * Paper: https://aclanthology.org/2020.emnlp-main.602/
    <br> * Approach: GPT model trained jointly on verb MLM+paraphrasing tasks, rescale the logits of next tokens with certain attributes before making predictions 
28. MixingBoard: a Knowledgeable Stylized Integrated Text Generation Platform (Gao et al., ACL 2020)
    <br> * Paper: https://aclanthology.org/2020.acl-demos.26/
    <br> * Code: https://github.com/microsoft/MixingBoard
    <br> * Approach: A platform with various generation sub-models (stylized, constrained, conditioned, knowledge based generation etc.) and integrate them using token probability interpolation and latent interpolation. Stylized generation is implemented as first generate then edit/retrieve.
29. Pre-train and Plug-in: Flexible Conditional Text Generation with Variational Auto-Encoders (Duan et al., ACL 2020)
    <br> * Paper: https://arxiv.org/pdf/1911.03882.pdf
    <br> * Code: https://github.com/WHUIR/PPVAE
    <br> * Approach: A pre-trained unconditional VAE model (GRU encoder, transformer decoder) and a plug-in conditional VAE model (MLP encoder decoder) mapping global latent space to conditional latent space. By training a new plug-in VAE on a small amount of data it is possible to add new conditions.
30. Plug and Play Autoencoders for Conditional Text Generation (Mai et al., EMNLP 2020)
    <br> * Paper: https://arxiv.org/pdf/2010.02983.pdf
    <br> * Code: https://github.com/florianmai/emb2emb
    <br> * Approach: Train a plug-in mapping between latent space of a pre-trained VAE model and a conditional subspace. The predicted embedding is fed to decoder, and a gradient-based optimization method is used to guide generation. (LSTM autoencoder)
31. DEXPERTS: Decoding-Time Controlled Text Generation with Experts and Anti-Experts (Liu et al., ACL 2021)
    <br> * Paper: https://aclanthology.org/2021.acl-long.522.pdf
    <br> * Code: https://github.com/alisawuffles/DExperts
    <br> * Approach: LMs (gpt-2) trained on corpora of desired and undesired attributes (experts and anti-experts) generate logits of next token prediction separately, which are combined with logits from a frozen language model (gpt-2). The weighted sum of logits are normalized to decide final output. By applying autoencoder as generative model it is also possible to perform stylistic rewriting.
32. FUDGE: Controlled Text Generation With Future Discriminators (Yang and Klein, NAACL 2021)
    <br> * Paper: https://aclanthology.org/2021.naacl-main.276/
    <br> * Code: https://github.com/yangkevin2/naacl-2021-fudge-controlled-generation
    <br> * Approach: Train a light-weight classifier (LSTM) that predict a certain attribute given incomplete prefix, it is then paired with the logits from an autoregressive generative model (GPT-2) to get conditional probability at each step (Bayes rule)
33. Controlled Text Generation as Continuous Optimization with Multiple Constraints (Kumar et al., NeurIPS 2021)
    <br> * Paper: https://openreview.net/forum?id=kTy7bbm-4I4
    <br> * Code: https://github.com/sachin19/mucoco
    <br> * An output sentence of certain lenghth of represented as a matrix, where each vector represents a probability distribution over vocabulary to enable continuous optimization. The matrix is updated to minimize the joint loss of LM and multiple attribute classifiers (gpt-2) as a gradient-based optimization. (Style transfer and style-controlled generation)
34. Neurologic decoding:(un) supervised neural text generation with predicate logic constraints (Lu et al., NAACL 2021)
    <br> * Paper: https://aclanthology.org/2021.naacl-main.339.pdf
    <br> * Approach: A lexical constrained decoding scheme, considering log-likelihood and constraints fulfilled at each decoding step
35. A Causal Lens for Controllable Text Generation (Hu and Li, NeurIPS 2021)
    <br> * Paper: https://proceedings.neurips.cc/paper/2021/hash/d0f5edad9ac19abed9e235c0fe0aa59f-Abstract.html
    <br> * Approach: VAE with structural causal model, incorporating style transfer and controlled generation in the same framework (gpt-2)
36. A Controllable Model of Grounded Response Generation (Wu et al., AAAI 2021)
    <br> * Paper: https://ojs.aaai.org/index.php/AAAI/article/view/17658
    <br> * Code: https://github.com/ellenmellon/CGRG
    <br> * Approach: With user-provided or automatically extracted control phrases (idf-based or bert), and relevant grounding knowledge from datasets, a GPT-2 model generates response conditioned on the concatenation of context and control phrases and knowledge. Only corresponding phrases and knowledge are allowed to to attend each other.
37. Cocon: A self-supervised approach for controlled text generation (Chan et al., ICLR 2021)
    <br> * Paper: https://openreview.net/forum?id=VD_ozqvBy4W
    <br> * Code: https://github.com/alvinchangw/COCON_ICLR2021
    <br> * Approach: A content conditional system. A single layer transformer block is inserted into GPT-2, which allows intermediate representation of prompt/generated token to attend to content on which generation is conditioned. Model can be trained in a self-supervised manner with various losses.
38. Generate your counterfactuals: Towards controlled counterfactual generation for text (Madaan et al., AAAI 2021)
    <br> * Paper: https://ojs.aaai.org/index.php/AAAI/article/view/17594
    <br> * Approach: A gradient-based PPLM model (leveraging gpt-2) that optimizes the history matrix during inference. Two perturbations are learned, one for content preservation and one for attribute control. 
39. Mix and Match: Learning-free Controllable Text Generationusing Energy Language Models (Mireshghallah et al., ACL 2022)
    <br> * Paper: https://aclanthology.org/2022.acl-long.31/
    <br> * Code: https://github.com/mireshghallah/mixmatch
    <br> * Approach: An energy-based model. Pre-trained components are linearly combined without further fine-tuning. A Metropolis-Hastings MCMC is used to sample from MLM (controlled text generation and text revision).



## Text Simplification

## Text Style Transfer

1. Style Transfer from Non-Parallel Text by Cross-Alignment (Shen et al., NeurIPS 2017)
    <br> * Paper: https://arxiv.org/pdf/1705.09655.pdf
    <br> * Code: https://github.com/shentianxiao/language-style-transfer
    <br> * Approach: Assume a common content distribution determines generative process of two non-parallel datasets of different attributes, two auto-encoders with aligned posterior or aligned samples is trained via discriminators (GRU-based). The auto-encoders can be then used as style transfer functions. 
2. Sequence to Better Sequence: Continuous Revision of Combinatorial Structures (Mueller et al., ICML 2017)
    <br> * Paper: http://proceedings.mlr.press/v70/mueller17a/mueller17a.pdf
    <br> * Approach: RNN-based VAE, with a ffnn predicting an output scalar given latent variable. A source sentence is first encoded, the latent representation is then refined according to the gradient of ffnn to generate its revised version.
3. Delete, Retrieve, Generate: A Simple Approach to Sentiment and Style Transfer (Li et al., NAACL 2018)
    <br> * Paper: https://openreview.net/forum?id=B1bTGQ-ubS
    <br> * Code: https://github.com/lijuncen/Sentiment-and-Style-Transfer
    <br> * Approach: First identify attribute markers from datasets, then delete source attribute markers and retrieve attribute sentences/markers that are similar to the content of source sentence (both in a non-neural manner). To transfer, RNN generates a new sentence conditioned on the content of the source and retrieved sentence
4. Style Transfer in Text: Exploration and Evaluation (Fu et al., AAAI 2018)
    <br> * Paper: https://ojs.aaai.org/index.php/AAAI/article/view/11330
    <br> * Code: https://github.com/fuzhenxin/text_style_transfer
    <br> * Approach: A encoder-multiple decoder system or encoder-decoder-style embedding system (GRU-based). The encoder is intended to learn only content and is trained via adversarial networks
5. Style Transfer Through Back-Translation (Prabhumoye et al., ACL 2018)
    <br> * Paper: https://aclanthology.org/P18-1080/?ref=https://githubhelp.com
    <br> * Code: https://github.com/shrimai/Style-Transfer-Through-Back-Translation
    <br> * Approach: A latent representation that preserves only content is obtained by a back-translation system (en-fr, fr-en) trained on style-agnostic corpus. An LSTM with attention model generates output conditioned on latent code, which is trained to minimize generation loss and classification loss provided by a pre-trained CNN label-classifier.
6. Unsupervised Text Style Transfer using Language Models as Discriminators (Yang et al., NeurIPS 2018)
    <br> * Paper: https://proceedings.neurips.cc/paper/2018/hash/398475c83b47075e8897a083e97eb9f0-Abstract.html
    <br> * Code: https://github.com/asyml/texar/tree/master/examples/text_style_transfer
    <br> * Approach: Use a language model on target corpus as discriminator, train the encoder, generator (VAE) and language model (discriminator) by maximizing real sentences in the corpus and minimizing transferred sentences from other datasets. 
7. Adversarially Regularized Autoencoders (Zhao et al., ICML 2018)
    <br> * Paper: https://arxiv.org/pdf/1706.04223.pdf
    <br> * Code: https://github.com/jakezhaojb/ARAE
    <br> * Approach: A Wasserstein autoencoder model (RNN) with prior learned by a GAN generator and Wasserstein distance learned by GAN discriminator (MLP). An MLP classifier can provide guidance during training.
8. Unpaired Sentiment-to-Sentiment Translation: A Cycled Reinforcement Learning Approach (Xu et al., ACL 2018)
    <br> * Paper: https://arxiv.org/pdf/1805.05181.pdf
    <br> * Code: https://github.com/lancopku/unpaired-sentiment-translation
    <br> * Approach: An LSTM with attention model is first trained to predict the sentiment of a sentence, and its attention weight is used to detect neutral and polar words in an input sentence. A neutralization model learns to predict neutral/polar words. A emotionalization model (Seq2seq) encodes neutral words of a sentence and sentiment is added to the decoder. Both models are first pre-trained and then trained end2end with RL approach.
9. Adversarial Text Generation via Feature-Mover's Distance (Chen et al., NeurIPS 2018)
    <br> * Paper: https://arxiv.org/pdf/1809.06297.pdf
    <br> * Code: https://github.com/vijini/FM-GAN
    <br> * Approach: A conditional VAE to extract style-agnostic content and add style, trained with GAN with Feature-Mover's distance as discriminator. (LSTM+CNN) 
10. Dear Sir or Madam, May I introduce the YAFC Dataset: Corpus, Benchmarks and Metrics for Formality Style Transfer (Rao and Tetreault, NAACL 2018)
    <br> * Paper: https://arxiv.org/pdf/1803.06535.pdf
    <br> * Approach: A supervised dataset with formal/informal sentence pairs. Baseline models are proprosed which see supervised style transfer as a machine translation task.
11. Fighting Offensive Language on Social Media with Unsupervised Text Style Transfer (Santos et al., ACL 2018)
    <br> * Paper: https://arxiv.org/pdf/1805.07685.pdf
    <br> * Approach: A GRU with attention encoder decoder system is trained with auto-encoding and back-translation tasks, with a classifier ensuring the transferred sentence have the desired attribute (adversarial training)
12. Style Transformer: Unpaired Text Style Transfer without Disentangled Latent Representation (Dai et al., ACL 2019)
    <br> * Paper: https://arxiv.org/pdf/1905.05621.pdf
    <br> * Code: https://github.com/fastnlp/style-transformer
    <br> * Approach: A transformer with style embedding is trained with auto-encoding and back-translation, and a transformer encoder classifier is used to ensure the transferred sentence have the given style. Both models are trained iteratively. 
13. Transforming delete, retrieve, generate approach for controlled text style transfer (Sudhakar et al., EMNLP 2019)
    <br> * Paper: https://aclanthology.org/D19-1322/
    <br> * Code: https://github.com/agaralabs/transformer-drg-style-transfer
    <br> * Approach: A Bert-based style classifier detects stylistic components in a sentence according to attention scores, these components will be removed to get the style-agnostic content, and a non-neural retrieve model selects sentences from target style datasets whose content is similar to the input's content. A GPT model learns to generate transferred sentence given the input content and target style or retrieved target attributes. Since there is no parallel data, the generative models learns from a denoising reconstruction task. Attributes can be set by user during inference.
14. Disentangled representation learning for non-parallel text style transfer (John et al., ACL 2019)
    <br> * Paper: https://aclanthology.org/P19-1041/?ref=https://githubhelp.com
    <br> * Code: https://github.com/vineetjohn/linguistic-style-transfer
    <br> * Approach: An RNN-based VAE model whose latent space is expected to be a concatenation of style encoding and content encoding. This is trained by 1. maximizing the probability of style/content encoding predicting style/content and 2. minimizing the probability of style/content encoding predicting content/style (four discriminators). 
15. Multiple-attribute text rewriting (Lample et al., ICLR 2019)
    <br> * Paper: https://openreview.net/forum?id=H1g2NhC5KQ
    <br> * Code: https://github.com/martiansideofthemoon/style-transfer-paraphrase
    <br> * Approach: LSTM encoder decoder model trained with denoising auto-encoding and back translation task. Attribute-specific start of sentence token is used for decoding in different styles.
16. A dual reinforcement learning framework for unsupervised text style transfer (Luo et al., IJCAI 2019)
    <br> * Paper: https://openreview.net/forum?id=oJvtywo_YP0
    <br> * Code: https://github.com/luofuli/DualLanST
    <br> * Approach: Two models that transfer styles in two directions are trained together with first back-translated pseudo parallel data then with content-preserving and style changing metrics in an RL approach.
17. Politeness Transfer: A Tag and Generate Approach (Madaan et al., ACL 2020)
    <br> * Paper: https://arxiv.org/abs/2004.14257
    <br> * Code: https://github.com/tag-and-generate/Politeness-Transfer-A-Tag-and-Generate-Approach
    <br> * Approach: First collect phrases relevant to certain styles from datasets, which prepare style-agnostic and stylistic components as dataset for model training.Then a tagger learns to replace stylistic components from input with [tag]/add [tag] to neutral inputs, and a generator learns to fill in the [tag]. Both models are implemented with a transformer.
18. Contextual text style transfer (Cheng et al., EMNLP 2020)
    <br> * Paper: https://aclanthology.org/2020.findings-emnlp.263/
    <br> * Approach: A dual encoder single decoder system that leverage context in style transfer. The sentence encoder decoder is trained unsupervised with auto-encoding, back-translation and a pre-trained style classifier teaches the model to add style. The whole system is also trained on supervised data, where the decoder is conditioned on the context encoding and input sentence encoding and target style, aiming to minimize difference from parallel data and maximize fluency. 
19. Reformulating unsupervised style transfer as paraphrase generation (Krishna et al., EMNLP 2020)
    <br> * Paper: https://aclanthology.org/2020.emnlp-main.55/
    <br> * Code: https://github.com/martiansideofthemoon/style-transfer-paraphrase
    <br> * Approach: A pre-trained GPT-2 model learns to normalize input sentences (strip styles) to form normalized-stylistic training pairs and another GPT-2 model learns to add style from the pseudo parallel data. During inference inputs are first normalized then style is inserted. 
20. A probabilistic formulation of unsupervised text style transfer (He et al., ICLR 2020)
    <br> * Paper: https://openreview.net/forum?id=HJlA0C4tPS
    <br> * Code: https://github.com/cindyxinyiwang/deep-latent-sequence-model
    <br> * Approach:
21. Revision in Continuous Space: Unsupervised Text Style Transfer without Adversarial Learning (Liu et al., AAAI 2020)
    <br> * Paper: https://arxiv.org/pdf/1905.12304.pdf
    <br> * Code: https://github.com/dayihengliu/Fine-Grained-Style-Transfer
    <br> * Approach: An RNN-based VAE is trained jointly with a content (bag-of-word) predictor and attribute predictor (based on latent representation). During inference the revision is done on latent space and is guided by gradients of predictors.
22. On Variational Learning of Controllable Representations for Text without Supervision (Xu et al., ICML 2020)
    <br> * Paper: https://arxiv.org/pdf/1905.11975.pdf
    <br> * Code: https://github.com/BorealisAI/CP-VAE
    <br> * Approach: Learn to map posterior of VAE (LSTM+MLP) to a probability simplex which has no vacancy. Latent representation is a concatenation of content representation and style representation, which can be manipulated during inference.P
23. Exploring Contextual Word-level Style Relevance for Unsupervised Style Transfer (Zhou et al., ACL 2020)
    <br> * Paper: https://arxiv.org/pdf/2005.02049.pdf
    <br> * Code: https://github.com/PaddlePaddle/Research
    <br> * Approach: An encoder decoder model (GRU with attention) learns to reconstruct input and detect each word's relevance to the style (learn from a pre-trained style classifier). An MLP learns to add the revision to hidden states at each step, which is trained jointly with the encoder decoder model against style classification and content preserving losses.
24. Text Style Transfer via Learning Style Instance Supported Latent Space (Yi et al., IJCAI 2020)
    <br> * Paper: https://www.ijcai.org/Proceedings/2020/0526.pdf
    <br> * Approach: A generative flow based model learns latent distribution of each style, and an lstm with attention encoder decoder is trained on auto-encoding and back-translation with adversarial training of a style classifier. The style representation is concatenated to word embedding in decoder at each step. 
25. Cycle-Consistent Adversarial Autoencoders for Unsupervised Text Style Transfer (Huang et al., COLING 2020)
    <br> * Paper: https://arxiv.org/pdf/2010.00735.pdf
    <br> * Approach: Two autoencoders (LSTM) are trained for two style corpora. Two transformation functions are learned to map latent space of one autoencoder to the other by adversarial training and cycle-consistent constraints.
26. How Positive Are You: Text Style Transfer using Adaptive Style Embedding (Kim and Sohn, COLING 2020)
    <br> * Paper: https://aclanthology.org/2020.coling-main.191.pdf
    <br> * Code: https://github.com/kinggodhj/how-positive-are-you-text-style-transfer-using-adaptive-style-embedding
    <br> * Approach: A transformer autoencoder learns to reconstruct inputs, and another model learns style embedding. The style is added to latent representation of autoencoder before it is fed into decoder. Both models are trained jointly.
27. Unsupervised Text Style Transfer with Padded Masked Language Models (Malmi et al., EMNLP 2020)
    <br> * Paper: https://arxiv.org/pdf/2010.01054.pdf
    <br> * Approach: Train two padded MLMs (in the experiment one conditional MLM) on source and target domain, find the span that two MLMs differ the most in terms of probability from an input, mask these spans and ask the target-side model to predict tokens. 
28. Unsupervised Text Generation by Learning from Search (Li et al., NeurIPS 2020)
    <br> * Paper: https://papers.nips.cc/paper/2020/file/7a677bb4477ae2dd371add568dd19e23-Paper.pdf
    <br> * Approach:
29. DGST: a Dual-Generator Network for Text Style Transfer (Li et al., EMNLP 2020)
    <br> * Paper: https://arxiv.org/pdf/2010.14557.pdf
    <br> * Approach:
30. Formality Style Transfer with Shared Latent Space (Wang et al., COLING 2020)
    <br> * Paper: https://aclanthology.org/2020.coling-main.203.pdf
    <br> * Code: https://github.com/jimth001/formality_style_transfer_with_shared_latent_space
    <br> * Approach:
31. Parallel Data Augmentation for Formality Style Transfer (Zhang et al., ACL 2020)
    <br> * Paper: https://arxiv.org/pdf/2005.07522.pdf
    <br> * Code: https://github.com/lancopku/Augmented_Data_for_FST
    <br> * Approach:
32. Expertise Style Transfer: A New Task Towards Better Communication between Experts and Laymen (Cao et al., ACL 2020)
    <br> * Paper: https://arxiv.org/pdf/2005.00701.pdf
    <br> * Approach:
33. Adapting Language Models for Non-Parallel Author-Stylized Rewriting (Syed et al., AAAI 2020)
    <br> * Paper: https://arxiv.org/pdf/1909.09962.pdf
34. Hooks in the Headline: Learning to Generate Headlines with Controlled Styles (Jin et al., ACL 2020)
    <br> * Paper: https://arxiv.org/pdf/2004.01980.pdf
    <br> * Code: https://github.com/jind11/TitleStylist
    <br> * Approach:
35. Exploring Contextual Word-level Style Relevance for Unsupervised Style Transfer (Zhou et al., ACL 2020)
    <br> * Paper: https://arxiv.org/pdf/2005.02049.pdf
    <br> * Code: https://github.com/PaddlePaddle/Research
    <br> * Approach:
36. Style versus Content: A distinction without a (learnable) difference? (Jafaritazehjani et al., COLING 2020)
    <br> * Paper: https://aclanthology.org/2020.coling-main.197.pdf
    <br> * Approach:
37. Towards A Friendly Online Community: An Unsupervised Style Transfer Framework for Profanity Redaction (Tran et al., COLING 2020)
    <br> * Paper: https://arxiv.org/pdf/2011.00403.pdf 
    <br> * Approach:
38. Style Pooling: Automatic Text Style Obfuscation for Improved Classification Fairness (Mireshghallah et al., EMNLP 2021)
    <br> * Paper: https://aclanthology.org/2021.emnlp-main.152/
    <br> * Code: https://github.com/mireshghallah/style-pooling
    <br> * Approach:
39. On Learning Text Style Transfer with Direct Rewards (Liu et al., NAACL 2021)
    <br> * Paper: https://arxiv.org/pdf/2010.12771.pdf
    <br> * Code: https://github.com/yixinL7/Direct-Style-Transfer
    <br> * Approach:
40. Multi-Style Transfer with Discriminative Feedback on Disjoint Corpus (Goyal et al., NAACL 2021)
    <br> * Paper: https://arxiv.org/pdf/2010.11578.pdf
    <br> * Approach:
41. A Hierarchical VAE for Calibrating Attributes while Generating Text using Normalizing Flow (Samanta et al., ACL 2021)
    <br> * Paper: https://aclanthology.org/2021.acl-long.187.pdf
    <br> * Approach:
42. Enhancing Content Preservation in Text Style Transfer Using Reverse Attention and Conditional Layer Normalization (Lee et al., ACL 2021)
    <br> * Paper: https://arxiv.org/pdf/2108.00449.pdf
    <br> * Code: https://github.com/MovingKyu/RACoLN
    <br> * Approach:
43. Counterfactuals to Control Latent Disentangled Text Representations for Style Transfer (Nangi et al., ACL 2021)
    <br> * Paper: https://aclanthology.org/2021.acl-short.7.pdf
    <br> * Approach:
44. TextSETTR: Few-Shot Text Style Extraction and Tunable Targeted Restyling (Riley et al., ACL 2021)
    <br> * Paper: https://aclanthology.org/2021.acl-long.293.pdf
    <br> * Code: https://github.com/mingkaid/rl-prompt
    <br> * Approach:
45. LEWIS: Levenshtein Editing for Unsupervised Text Style Transfer (Reid and Zhong, ACL Findings 2021)
    <br> * Paper: https://arxiv.org/pdf/2105.08206.pdf
    <br> * Code: https://github.com/machelreid/lewis
    <br> * Approach:
46. NAST: A Non-Autoregressive Generator with Word Alignment for Unsupervised Text Style Transfer (Huang et al., ACL Findings 2021)
    <br> * Paper: https://arxiv.org/pdf/2106.02210.pdf
    <br> * Code: https://github.com/thu-coai/NAST
    <br> * Approach:
47. Olá, Bonjour, Salve! XFORMAL: A Benchmark for Multilingual Formality Style Transfer (Briakou et al., NAACL 2021)
    <br> * Paper: https://aclanthology.org/2021.naacl-main.256.pdf
    <br> * Approach:
48. Improving Formality Style Transfer with Context-Aware Rule Injection (Yao et al., ACL 2021)
    <br> * Paper: https://arxiv.org/pdf/2106.00210.pdf
    <br> * Approach:
49. Thank you BART! Rewarding Pre-Trained Models Improves Formality Style Transfer (Lai et al., ACL 2021)
    <br> * Paper: https://aclanthology.org/2021.acl-short.62.pdf
    <br> * Code: https://github.com/laihuiyuan/Pre-trained-formality-transfer
    <br> * Approach:
50. Controllable Text Simplification with Explicit Paraphrasing (Maddela et al., NAACL 2021)
    <br> * Paper: https://arxiv.org/pdf/2010.11004.pdf
    <br> * Code: https://github.com/facebookresearch/access
    <br> * Approach:
51. Paragraph-level Simplification of Medical Texts (Devaraj et al., NAACL 2021)
    <br> * Paper: https://aclanthology.org/2021.naacl-main.395.pdf
    <br> * Code: https://github.com/AshOlogn/Paragraph-level-Simplification-of-Medical-Texts
    <br> * Approach:
52. Towards Modeling the Style of Translators in Neural Machine Translation (Wang et al., NAACL 2021)
    <br> * Paper: https://aclanthology.org/2021.naacl-main.94.pdf 
    <br> * Approach:
53. Inference Time Style Control for Summarization (Cao et al., NAACL 2021)
    <br> * Paper: https://arxiv.org/pdf/2104.01724.pdf
    <br> * Approach:
54. Exploring Non-Autoregressive Text Style Transfer (Ma and Li, EMNLP 2021)
    <br> * Paper: https://aclanthology.org/2021.emnlp-main.730.pdf
    <br> * Code: https://github.com/sunlight-ym/nar_style_transfer
    <br> * Approach:
55. Generic resources are what you need: Style transfer tasks without task-specific parallel training data (Lai et al., EMNLP 2021)
    <br> * Paper: https://arxiv.org/pdf/2109.04543.pdf
    <br> * Code: https://github.com/laihuiyuan/generic-resources-for-tst
    <br> * Approach:
56. Transductive Learning for Unsupervised Text Style Transfer (Xiao et al., EMNLP 2021)
    <br> * Paper: https://arxiv.org/pdf/2109.07812.pdf
    <br> * Code: https://github.com/xiaofei05/tsst
    <br> * Approach:
57. Collaborative Learning of Bidirectional Decoders for Unsupervised Text Style Transfer (Ma et al., EMNLP 2021)
    <br> * Paper: https://aclanthology.org/2021.emnlp-main.729.pdf
    <br> * Code: https://github.com/sunlight-ym/cbd_style_transfer
    <br> * Approach:
58. STYLEPTB: A Compositional Benchmark for Fine-grained Controllable Text Style Transfer (Lyu et al., NAACL 2021)
    <br> * Paper: https://arxiv.org/pdf/2104.05196.pdf
    <br> * Code: https://github.com/lvyiwei1/StylePTB
    <br> * Approach:
59. Rethinking Sentiment Style Transfer (Yu et al., EMNLP 2021)
    <br> * Paper: https://aclanthology.org/2021.findings-emnlp.135.pdf 
    <br> * Approach:
60. Evaluating the Evaluation Metrics for Style Transfer: A Case Study in Multilingual Formality Transfer (Briakou et al., EMNLP 2021)
    <br> * Paper: https://aclanthology.org/2021.emnlp-main.100.pdf
    <br> * Code: https://github.com/elbria/xformal-fost-meta
    <br> * Approach:
61. Does It Capture STEL? A Modular, Similarity-based Linguistic Style Evaluation Framework (Wegmann et al., EMNLP 2021)
    <br> * Paper: https://aclanthology.org/2021.emnlp-main.569.pdf
    <br> * Code: https://github.com/nlpsoc/stel
    <br> * Approach:
62. MISS: An Assistant for Multi-Style Simultaneous Translation (Li et al., EMNLP 2021)
    <br> * Paper: https://aclanthology.org/2021.emnlp-demo.1.pdf 
    <br> * Approach:
63. A Recipe for Arbitrary Text Style Transfer with Large Language Models (Reif et al., ACL 2022)
    <br> * Paper: https://aclanthology.org/2022.acl-short.94/
    <br> * Approach:
64. So Different Yet So Alike! Constrained Unsupervised Text Style Transfer (Kashyap et al., ACL 2022)
    <br> * Paper: https://aclanthology.org/2022.acl-long.32.pdf
    <br> * Code: https://github.com/abhinavkashyap/dct
    <br> * Approach:
65. Semi-Supervised Formality Style Transfer with Consistency Training (Liu et al., ACL 2022)
    <br> * Paper: https://aclanthology.org/2022.acl-long.321.pdf
    <br> * Code: https://github.com/aolius/semi-fst
    <br> * Approach:
## Stylistic Text Generation

## Diffusion Related Model

### Diffusion in NLP
1. Step-unrolled denoising autoencoders for text generation (Savinov et al., ICLR 2022)
    <br> * Paper: https://arxiv.org/pdf/2112.06749.pdf
    <br> * Code: https://github.com/vvvm23/sundae
2. Structured Denoising Diffusion Models in Discrete State-Space (Austin et al., NeurIPS 2021)
    <br> * Paper: https://proceedings.neurips.cc/paper/2021/hash/958c530554f78bcd8e97125b70e6973d-Abstract.html
    <br> * Code: https://github.com/samb-t/unleashing-transformers    
3. Argmax flows and multinomial diffusion: Towards non-autoregressive language models (Hoogeboom et al, CoRR 2021)
    <br> * Paper: https://arxiv.org/pdf/2102.05379v3.pdf
4. Diffusion-LM Improves Controllable Text Generation (Li et al, 2022)
    <br> * Paper: https://arxiv.org/pdf/2205.14217.pdf
    <br> * Code: https://github.com/XiangLi1999/Diffusion-LM

5. Analog Bits: Generating Discrete Data using Diffusion Models with Self-Conditioning (Chen et al., 2022)
    <br> * Paper: https://arxiv.org/pdf/2208.04202.pdf
    <br> * Code: https://github.com/lucidrains/imagen-pytorch
6. DiffuSeq: Sequence to Sequence Text Generation with Diffusion Models (Gong et al., 2022)
    <br> * Paper: https://arxiv.org/pdf/2210.08933.pdf
    <br> * Code: https://github.com/Shark-NLP/DiffuSeq
7. CLIP-Diffusion-LM: Apply Diffusion Model on Image Captioning (Shitong Xu, 2022)
    <br> * Paper: https://arxiv.org/pdf/2210.04559.pdf
    <br> * Code: https://github.com/xu-shitong/diffusion-image-captioning
8. Categorical SDEs with Simplex Diffusion (Richemond et al., 2022)
    <br> * Paper: https://arxiv.org/pdf/2210.14784.pdf
9. Latent Diffusion Energy-Based Model for Interpretable Text Modeling (Yu et al., ICML 2022)
    <br> * Paper: https://proceedings.mlr.press/v162/yu22h.html
    <br> * Code: https://github.com/yupeiyu98/ldebm
### Diffusion in other fields
1. Deep Unsupervised Learning using Non equilibrium Thermodynamics (Sohl-Dickstein et al., ICML 2015)
    <br> * Paper: https://openreview.net/forum?id=rkbVIoZdWH

2. Generative modeling by estimating gradients of the data distribution (Yang Song and Stefano Ermon, NeurIPS 2019)
    <br> * Paper: https://proceedings.neurips.cc/paper/2019/hash/3001ef257407d5a371a96dcd947c7d93-Abstract.html
    <br> * Code: https://github.com/yang-song/score_sde
3. Denoising Diffusion Implicit Models (Song et al., ICLR 2019)
    <br> * Paper: https://openreview.net/forum?id=St1giarCHLP
    <br> * Code: https://github.com/labmlai/annotated_deep_learning_paper_implementations
4. Score-Based Generative Modeling through Stochastic Differential Equations (Song et al., ICLR 2021)
    <br> * Paper: https://openreview.net/forum?id=PxTIG12RRHS
    <br> * Code: https://github.com/yang-song/score_sde
5. Adversarial score matching and improved sampling for image generation (Jolicoeur-Martineau et al., ICLR 2021)
    <br> * Paper: https://openreview.net/forum?id=eLfqMl3z3lq
    <br> * Code: https://github.com/AlexiaJM/AdversarialConsistentScoreMatching
6. Improved denoising diffusion probabilistic models (Nichol and Dhariwal, ICML 2021)
    <br> * Paper: https://proceedings.mlr.press/v139/nichol21a.html
    <br> * Code: https://github.com/openai/improved-diffusion
7. Variational diffusion models (Kingma et al., NeurIPS 2021)
    <br> * Paper: https://proceedings.neurips.cc/paper/2021/hash/b578f2a52a0229873fefc2a4b06377fa-Abstract.html
    <br> * Code: https://github.com/google-research/vdm
8. Maximum likelihood training of score-based diffusion models (Song et al., NeurIPS 2021)
    <br> * Paper: https://proceedings.neurips.cc/paper/2021/hash/0a9fdbb17feb6ccb7ec405cfb85222c4-Abstract.html
    <br> * Code: https://github.com/yang-song/score_flow
9. A variational perspective on diffusion-based generative models and score matching (Huang et al., NeurIPS 2021)
    <br> * Paper: https://proceedings.neurips.cc/paper/2021/hash/c11abfd29e4d9b4d4b566b01114d8486-Abstract.html
    <br> * Code: https://github.com/CW-Huang/sdeflow-light
10. Score-based generative modeling in latent space (Vahdat et al., NeurIPS 2021)
    <br> * Paper: https://proceedings.neurips.cc/paper/2021/hash/5dca4c6b9e244d24a30b4c45601d9720-Abstract.html
    <br> * Code: https://github.com/NVlabs/LSGM
11. Learning gradient fields for molecular conformation generation (Shi et al., ICML 2021)
    <br> * Paper: http://proceedings.mlr.press/v139/shi21b.html
    <br> * Code: https://github.com/DeepGraphLearning/ConfGF
12. 3d shape generation and completion through point-voxel diffusion (Zhou et al., ICCV 2021)
    <br> * Paper: https://openaccess.thecvf.com/content/ICCV2021/html/Zhou_3D_Shape_Generation_and_Completion_Through_Point-Voxel_Diffusion_ICCV_2021_paper.html
    <br> * Code: https://github.com/alexzhou907/pvd
13. Diffusion probabilistic models for 3d point cloud generation (Luo and Hu, CVPR 2021)
    <br> * Paper: https://openaccess.thecvf.com/content/CVPR2021/html/Luo_Diffusion_Probabilistic_Models_for_3D_Point_Cloud_Generation_CVPR_2021_paper.html
    <br> * Code: https://github.com/luost26/diffusion-point-cloud
14. CSDI: Conditional score-based diffusion models for probabilistic time series imputation (Tashiro et al., NeurIPS 2021)
    <br> * Paper: https://proceedings.neurips.cc/paper/2021/hash/cfe8504bda37b575c70ee1a8276f3486-Abstract.html
    <br> * Code: https://github.com/ermongroup/csdi
15. Autoregressive denoising diffusion models for multivariate probabilistic time series forecasting (Rasul et al., ICML 2021)
    <br> * Paper: http://proceedings.mlr.press/v139/rasul21a.html
    <br> * Code: https://github.com/zalandoresearch/pytorch-ts
16. WaveGrad: Estimating Gradients for Waveform Generation (Chen et al., ICLR 2021)
    <br> * Paper: https://openreview.net/forum?id=NsMLjcFaO8O
    <br> * Code: https://github.com/coqui-ai/TTS
17. DiffWave: A Versatile Diffusion Model for Audio Synthesis (Kong et al., ICLR 2021)
    <br> * Paper: https://openreview.net/forum?id=a-xFK8Ymz5J
    <br> * Code: https://github.com/lmnt-com/diffwave
18. GLIDE: Towards Photorealistic Image Generation and Editing with Text-Guided Diffusion Models (Nichol et al., 2021)
    <br> * Paper: https://arxiv.org/abs/2112.10741
    <br> * Code: https://github.com/openai/glide-text2im
19. Grad-TTS: A Diffusion Probabilistic Model for Text-to-Speech (Popov et al., ICML 2021)
    <br> * Paper: https://proceedings.mlr.press/v139/popov21a.html
    <br> * Code: https://github.com/huawei-noah/Speech-Backbones
20. Adversarial purification with score-based generative models (Yoon et al., ICML 2021)
    <br> * Paper: http://proceedings.mlr.press/v139/yoon21a.html
    <br> * Code: https://github.com/jmyoon1/adp
21. Learning gradient fields for molecular conformation generation (Shi et al., ICML 2021)
    <br> * Paper: http://proceedings.mlr.press/v139/shi21b.html
    <br> * Code: https://github.com/DeepGraphLearning/ConfGF
22. Predicting molecular conformation via dynamic graph score matching (Luo et al., NeurIPS 2021)
    <br> * Paper: https://proceedings.neurips.cc/paper/2021/hash/a45a1d12ee0fb7f1f872ab91da18f899-Abstract.html
23. A variational perspective on diffusion-based generative models and score matching (Huang et al., NeurIPS 2021)
    <br> * Paper: https://proceedings.neurips.cc/paper/2021/hash/c11abfd29e4d9b4d4b566b01114d8486-Abstract.html
    <br> * Code: https://github.com/CW-Huang/sdeflow-light
24. Diffusion Models Beat GANs on Image Synthesis (Dhariwal and Nichol, NeurIPS 2021)
    <br> * Paper: https://proceedings.neurips.cc/paper/2021/hash/49ad23d1ec9fa4bd8d77d02681df5cfa-Abstract.html
    <br> * Code: https://github.com/openai/guided-diffusion
25. Diffusion Normalizing Flow (Zhang and Chen, NeurIPS 2021)
    <br> * Paper: https://proceedings.neurips.cc/paper/2021/hash/876f1f9954de0aa402d91bb988d12cd4-Abstract.html
    <br> * Code: https://github.com/qsh-zh/DiffFlow
26. Learning Energy-Based Models by Diffusion Recovery Likelihood (Gao et al., ICLR 2021)
    <br> * Paper: https://openreview.net/forum?id=v_1Soh8QUNc
    <br> * Code: https://github.com/qsh-zh/DiffFlow
27. Come-closer-diffuse-faster: Accelerating conditional diffusion models for inverse problems through stochastic contraction (Chung et al., CVPR 2022)
    <br> * Paper: https://openaccess.thecvf.com/content/CVPR2022/html/Chung_Come-Closer-Diffuse-Faster_Accelerating_Conditional_Diffusion_Models_for_Inverse_Problems_Through_Stochastic_CVPR_2022_paper.html
28. Score-Based Generative Modeling with Critically-Damped Langevin Diffusion (Dockhorn et al., ICLR 2022)
    <br> * Paper: https://openreview.net/forum?id=CzceR82CYc
    <br> * Code: https://github.com/nv-tlabs/CLD-SGM
29. Elucidating the Design Space of Diffusion-Based Generative Models (Karras et al., NeurIPS 2022)
    <br> * Paper: https://arxiv.org/abs/2206.00364
    <br> * Code: https://github.com/nvlabs/edm
30. DPM-Solver: A Fast ODE Solver for Diffusion Probabilistic Model Sampling in Around 10 Step (Lu et al., NeurIPS 2022)
    <br> * Paper: https://arxiv.org/abs/2206.00927
    <br> * Code: https://github.com/luchengthu/dpm-solver
31. Pseudo Numerical Methods for Diffusion Models on Manifolds (Liu et al., ICLR 2022)
    <br> * Paper: https://openreview.net/forum?id=PlKWVd2yBkY
    <br> * Code: https://github.com/luping-liu/PNDM

32. GENIE: Higher-Order Denoising Diffusion Solvers (Dockhorn et al., NeurIPS 2022)
    <br> * Paper: https://arxiv.org/abs/2210.05475
    <br> * Code: https://github.com/nv-tlabs/GENIE
33. Learning fast samplers for diffusion models by differentiating through sample quality (Watson et al., ICLR 2022)
    <br> * Paper: https://openreview.net/forum?id=VFBjuF8HEp
34. Progressive Distillation for Fast Sampling of Diffusion Models (Salimans and Ho, ICLR 2022)
    <br> * Paper: https://openreview.net/forum?id=TIdIXIpzhoI
    <br> * Code: https://github.com/google-research/google-research/tree/master/diffusion_distillation
35. Analytic-DPM: an Analytic Estimate of the Optimal Reverse Variance in Diffusion Probabilistic Models (Bao et al., ICLR 2022)
    <br> * Paper: https://openreview.net/forum?id=0xiJLKH-ufZ
    <br> * Code: https://github.com/baofff/Analytic-DPM
36. Maximum Likelihood Training for Score-based Diffusion ODEs by High Order Denoising Score Matching (Lu et al., ICML 2022)
    <br> * Paper:https://proceedings.mlr.press/v162/lu22f.html
    <br> * Code: https://github.com/luchengthu/mle_score_ode
37. Hierarchical text-conditional image generation with clip latents (Ramesh et al., 2022)
    <br> * Paper: https://arxiv.org/abs/2204.06125
    <br> * Code: https://github.com/lucidrains/DALLE2-pytorch
38. High-resolution image synthesis with latent diffusion models (Rombach et al., CVPR 2022)
    <br> * Paper: https://openaccess.thecvf.com/content/CVPR2022/html/Rombach_High-Resolution_Image_Synthesis_With_Latent_Diffusion_Models_CVPR_2022_paper.html
    <br> * Code: https://github.com/CompVis/stable-diffusion

39. GeoDiff: A Geometric Diffusion Model for Molecular Conformation Generation (Xu et al., ICLR 2022)
    <br> * Paper: https://openreview.net/forum?id=PzcvxEMzvQC
    <br> * Code: https://github.com/minkaixu/geodiff
40. Score-based Generative Modeling of Graphs via the System of Stochastic Differential Equations (Jo et al., ICML 2022)
    <br> * Paper: https://proceedings.mlr.press/v162/jo22a.html
    <br> * Code: https://github.com/harryjo97/gdss
41. Vector quantized diffusion model for text-to-image synthesis (Gu et al., CVPR 2022)
    <br> * Paper: https://openaccess.thecvf.com/content/CVPR2022/html/Gu_Vector_Quantized_Diffusion_Model_for_Text-to-Image_Synthesis_CVPR_2022_paper.html
    <br> * Code: https://github.com/cientgu/vq-diffusion
42. SRDiff: Single Image Super-Resolution with Diffusion Probabilistic Models (Li et al., ACM 2022)
    <br> * Paper: https://www.sciencedirect.com/science/article/abs/pii/S0925231222000522
    <br> * Code: https://github.com/LeiaLi/SRDiff
43. Repaint: Inpainting using denoising diffusion probabilistic models (Lugmayr et al., CVPR 2022)
    <br> * Paper: https://openaccess.thecvf.com/content/CVPR2022/html/Lugmayr_RePaint_Inpainting_Using_Denoising_Diffusion_Probabilistic_Models_CVPR_2022_paper.html
    <br> * Code: https://github.com/andreas128/RePaint
44. Palette: Image-to-image diffusion models (Saharia et al., ACM 2022)
    <br> * Paper: https://openreview.net/forum?id=FPGs276lUeq
    <br> * Code: https://github.com/Janspiry/Palette-Image-to-Image-Diffusion-Models
45. Generative Visual Prompt: Unifying Distributional Control of Pre-Trained Generative Models (Wu et al., NeurIPS 2022)
    <br> * Paper: https://arxiv.org/abs/2209.06970
    <br> * Code: https://github.com/chenwu98/generative-visual-prompt
46. Cascaded Diffusion Models for High Fidelity Image Generation (Ho et al., JMLR 2022)
    <br> * Paper: https://www.jmlr.org/papers/v23/21-0635.html
47. Solving Inverse Problems in Medical Imaging with Score-Based Generative Models (Song et al., ICLR 2022)
    <br> * Paper: https://openreview.net/forum?id=vaRCHVj0uGI
    <br> * Code: https://github.com/yang-song/score_inverse_problems
48. Sdedit: Guided image synthesis and editing with stochastic differential equations (Meng et al., ICLR 2022)
    <br> * Paper: https://arxiv.org/abs/2108.01073
    <br> * Code: https://github.com/ermongroup/SDEdit

49. Label-Efficient Semantic Segmentation with Diffusion Models (Baranchuk et al., ICLR 2022)
    <br> * Paper: https://openreview.net/forum?id=SlxSY2UZQT
    <br> * Code: https://github.com/yandex-research/ddpm-segmentation
50. Diffusion models as plug-and-play priors (Graikos et al., NeurIPS 2022)
    <br> * Paper: https://arxiv.org/abs/2206.09012
    <br> * Code: https://github.com/alexgraikos/diffusion_priors
51. A Conditional Point Diffusion-Refinement Paradigm for 3D Point Cloud Completion (Lyu et al., ICLR 2022)
    <br> * Paper: https://openreview.net/forum?id=wqD6TfbYkrn
    <br> * Code: https://github.com/zhaoyanglyu/point_diffusion_refinement
52. LION: Latent Point Diffusion Models for 3D Shape Generation (Zeng et al., NeurIPS 2022)
    <br> * Paper: https://arxiv.org/abs/2210.06978
    <br> * Code: https://github.com/nv-tlabs/LION
53. Neural Markov Controlled SDE: Stochastic Optimization for Continuous-Time Data (Park et al., ICLR 2022)
    <br> * Paper: https://openreview.net/forum?id=7DI6op61AY
54. Photorealistic Text-to-Image Diffusion Models with Deep Language Understanding (Saharia et al., 2022)
    <br> * Paper: https://arxiv.org/abs/2205.11487
    <br> * Code: https://github.com/lucidrains/imagen-pytorch
55. Blended diffusion for text-driven editing of natural images (Avrahami et al., CVPR 2022)
    <br> * Paper: https://openaccess.thecvf.com/content/CVPR2022/html/Avrahami_Blended_Diffusion_for_Text-Driven_Editing_of_Natural_Images_CVPR_2022_paper.html
    <br> * Code: https://github.com/omriav/blended-diffusion
56. DiffusionCLIP: Text-Guided Diffusion Models for Robust Image Manipulation (Kim et al., CVPR 2022)
    <br> * Paper: https://openaccess.thecvf.com/content/CVPR2022/html/Kim_DiffusionCLIP_Text-Guided_Diffusion_Models_for_Robust_Image_Manipulation_CVPR_2022_paper.html
    <br> * Code: 
57. Zero-Shot Voice Conditioning for Denoising Diffusion TTS Models (Levkovitch et al., Interspeech 2022)
    <br> * Paper: https://arxiv.org/abs/2206.02246
    <br> * Code: https://github.com/gwang-kim/diffusionclip
58. ProDiff: Progressive Fast Diffusion Model For High-Quality Text-to-Speech (Huang et al., ACM 2022)
    <br> * Paper: https://arxiv.org/abs/2207.06389
    <br> * Code: https://github.com/Rongjiehuang/ProDiff
59. Diffusion Models for Adversarial Purification (Nie et al. ICML 2022)
    <br> * Paper: https://arxiv.org/abs/2205.07460
    <br> * Code: https://github.com/NVlabs/DiffPure
60. Equivariant Diffusion for Molecule Generation in 3D (Hoogeboom et al., ICML 2022)
    <br> * Paper: https://proceedings.mlr.press/v162/hoogeboom22a.html
    <br> * Code: https://github.com/ehoogeboom/e3_diffusion_for_molecules
61. GeoDiff: A Geometric Diffusion Model for Molecular Conformation Generation (Xu et al., ICLR 2022) 
    <br> * Paper: https://openreview.net/forum?id=PzcvxEMzvQC
    <br> * Code: https://github.com/minkaixu/geodiff
62. Crystal Diffusion Variational Autoencoder for Periodic Material Generation (Xie et al., ICLR 2022)
    <br> * Paper: https://arxiv.org/abs/2110.06197
    <br> * Code: https://github.com/txie-93/cdvae

63. Solving Inverse Problems in Medical Imaging with Score-Based Generative Models (Song et al., ICLR 2022)
    <br> * Paper: https://openreview.net/forum?id=vaRCHVj0uGI
    <br> * Code: https://github.com/yang-song/score_inverse_problems
64. Towards performant and reliable undersampled MR reconstruction via diffusion model sampling (Peng et al., MICCAI 2022)
    <br> * Paper: https://arxiv.org/pdf/2203.04292.pdf
    <br> * Code: https://github.com/cpeng93/diffuserecon
65. Come-closer-diffuse-faster: Accelerating conditional diffusion models for inverse problems through stochastic contraction (Chung et al., CVPR 2022)
    <br> * Paper: https://openaccess.thecvf.com/content/CVPR2022/papers/Chung_Come-Closer-Diffuse-Faster_Accelerating_Conditional_Diffusion_Models_for_Inverse_Problems_Through_Stochastic_CVPR_2022_paper.pdf
66. Tackling the generative learning trilemma with denoising diffusion gans (Xiao et al., ICLR 2022)
    <br> * Paper: https://openreview.net/forum?id=JprM0p-q0Co
    <br> * Code: https://github.com/NVlabs/denoising-diffusion-gan
67. Autoregressive Diffusion Models (Hoogeboom et al., ICLR 2022)
    <br> * Paper: https://openreview.net/forum?id=Lm8T39vLDTE
    <br> * Code: https://github.com/google-research/google-research/tree/master/autoregressive_diffusion
68. Latent Diffusion Energy-Based Model for Interpretable Text Modeling (Yu et al., ICML 2022)
    <br> * Paper: https://proceedings.mlr.press/v162/yu22h.html
    <br> * Code: https://github.com/yupeiyu98/ldebm


## Dataset


## Acknowledgement
We thank [Yang et al.](https://arxiv.org/pdf/2209.00796.pdf) and [Jin et al.](https://arxiv.org/pdf/2011.00416.pdf) for their comprehensive survey on recent development of diffusion models and style transfer. Their work can be found in the following github repositories:

[Diffusion Models: A Comprehensive Survey of Methods and Applications](https://github.com/YangLing0818/Diffusion-Models-Papers-Survey-Taxonomy)

[Deep Learning for Text Style Transfer: A Survey](https://github.com/zhijing-jin/Text_Style_Transfer_Survey)