Normalization in End-to-End TTS for Low-Resource Morphologically Complex Languages

Short description:
In theory, end-to-end TTS models like Tacotron aim to eliminate the need for manual text normalization, but in practice, especially for Low-Resource Morphologically Complex languages (like e.g. Serbian Nosek, T., Suzić, S., Sečujski, M., Stanojev, V., Pekar, D., & Delić, V. End-to-end speech synthesis for the Serbian language based on Tacotron. In International Conference on Speech and Computer, pp. 219-229, Cham: Springer Nature Switzerland), normalization is still needed, particularly for tricky cases numbers, abbreviations, and symbols.
In experimental part, we‘ll train TTS models with different levels of normalization and evaluate it (objective and subjective tests).

---

### **MSc Thesis Plan**

**Thesis Title:** The Role of Explicit Text Normalization in End-to-End Text-to-Speech Synthesis for Low-Resource, Morphologically-Rich Languages: A Case Study on [e.g., Serbian, Lithuanian, Finnish]

**Abstract (Draft):**
End-to-end (E2E) text-to-speech (TTS) models, such as Tacotron 2, theoretically learn to map raw text to speech, minimizing the need for complex linguistic preprocessing. However, this assumption often fails in practice, particularly for low-resource and morphologically-rich languages. These languages present unique challenges due to complex declension and conjugation patterns which are difficult for models to learn from limited data, especially when dealing with non-standard words (NSWs) like numbers, abbreviations, and symbols. This thesis investigates the impact of varying levels of explicit text normalization on the performance of an E2E TTS system for [Target Language]. We develop a multi-level normalization pipeline, from minimal cleaning to full linguistic verbalization that respects the language's morphology. By training separate Tacotron 2 models on each level of normalized text, we systematically evaluate their performance through objective metrics (Mel-Cepstral Distortion, Word Error Rate) and subjective listening tests (Mean Opinion Score). Our findings demonstrate that while E2E models can handle some simple cases, explicit, morphologically-aware normalization is crucial for achieving intelligible and natural-sounding speech, particularly for numerical expressions. This work not only quantifies the necessity of text normalization but also provides a reusable open-source normalization module for [Target Language].

---

### **Thesis Structure**

#### **Chapter 1: Introduction**
1.  **Motivation:**
    * The promise of E2E TTS: simplifying the traditional pipeline.
    * The reality: the "brittleness" of E2E models with out-of-domain text.
    * The importance of speech technologies for digital inclusivity and accessibility for low-resource languages.
    * The specific challenge: Morphologically-rich languages where a single number or symbol can have many spoken forms depending on grammatical context (e.g., case, gender).
2.  **Problem Statement:**
    * Standard E2E models trained on limited, un-normalized data for morphologically complex languages fail to correctly pronounce non-standard words, leading to significant degradation in intelligibility and naturalness.
3.  **Research Questions (RQs):**
    * **RQ1:** To what extent does the level of explicit text normalization affect the objective and subjective quality of a Tacotron 2-based TTS system for a low-resource, morphologically-rich language?
    * **RQ2:** Is there a point of diminishing returns, where the effort of advanced linguistic normalization no longer yields significant improvements in synthesized speech quality?
    * **RQ3:** Which categories of non-standard words (cardinal numbers, ordinals, abbreviations, currency symbols, etc.) present the most significant challenge for models trained with minimal normalization?
4.  **Hypothesis:**
    * A TTS model trained on a fully, morphologically-aware normalized dataset will achieve statistically significant improvements in both intelligibility (lower WER) and naturalness (higher MOS) compared to models trained on raw or minimally processed text. The most substantial gains will be observed in sentences containing complex numerical expressions.
5.  **Contributions:**
    * A systematic, empirical evaluation of the impact of text normalization on E2E TTS for [Target Language].
    * The development and release of an open-source text normalization module for [Target Language].
    * A publicly available set of trained models and generated audio samples demonstrating the results.
6.  **Thesis Outline:**
    * Briefly describe the structure of the remaining chapters.

#### **Chapter 2: Background and Literature Review**
1.  **Evolution of Speech Synthesis:**
    * From Concatenative (Unit Selection) and Statistical Parametric (HMMs) to Neural/E2E models.
2.  **End-to-End TTS Architectures:**
    * **Core Model:** Detailed explanation of the **Tacotron 2** architecture (Encoder-Decoder with Attention).
    * **Vocoders:** Explanation of why a neural vocoder is needed (e.g., **WaveGlow, HiFi-GAN**) to convert the mel-spectrogram into an audio waveform.
3.  **Text Normalization (TN):**
    * Definition: The process of converting non-standard words (NSWs) into their full spoken form.
    * The classic TN pipeline: Text Analysis -> NSW Detection -> Verbalization.
    * Rule-based vs. statistical/neural approaches to TN.
4.  **Linguistic Challenges of Morphologically-Rich Languages:**
    * Define morphological complexity (inflection, declension, conjugation).
    * Provide concrete examples from [Target Language]. For example, in Serbian, the number `2` can be `dva`, `dve`, `dvojica`, `dvoje`, `drugi`, `druga`, `drugo`, etc., depending on gender, animacy, and case. This is the core challenge.
5.  **Related Work:**
    * Review the cited paper: Nosek et al. on Serbian TTS.
    * Survey existing TTS systems for other morphologically complex languages (e.g., Finnish, Hungarian, Turkish, Polish).
    * Investigate studies that have specifically analyzed the impact of text preprocessing on E2E models in any language.

#### **Chapter 3: Methodology and Experimental Setup**
1.  **Dataset:**
    * **Source:** Describe the speech corpus used for [Target Language]. (e.g., National library recordings, publicly available audiobooks, custom-recorded data).
    * **Size:** Total hours of audio, number of utterances.
    * **Preprocessing:** Audio cleaning (resampling to 22050 Hz, trimming silence, normalizing volume) and transcript preparation.
2.  **The Normalization Pipeline: Defining the Levels**
    * This is the core independent variable of your experiment.
    * **Level 0 (Baseline):** Raw text. Only essential cleaning is performed (e.g., removing XML tags, sentence splitting).
    * **Level 1 (Basic):** Level 0 + Lowercasing, punctuation removal/standardization.
    * **Level 2 (Graphemic Expansion):** Level 1 + Expansion of abbreviations and symbols into their constituent words, but without grammatical context. Numbers are spelled out digit by digit.
        * e.g., `125€` -> `jedan dva pet evra` (one two five euros)
        * e.g., `dr.` -> `doktor`
    * **Level 3 (Full Linguistic/Morphological Normalization):** Level 2 + Full verbalization with correct morphological inflection. This requires a rule-based or statistical grammar engine.
        * e.g., `125€` -> `sto dvadeset pet evra` (one hundred twenty-five euros)
        * e.g., `Bio sam na 2. spratu.` (I was on the 2nd floor) -> `bio sam na drugom spratu`. The system must correctly infer the locative case `drugom`.
3.  **TTS Model and Training:**
    * **Model:** Tacotron 2 + HiFi-GAN (state-of-the-art vocoder).
    * **Framework:** Specify the toolkit (e.g., Coqui-TTS, ESPnet, NVIDIA's Tacotron 2 implementation).
    * **Training Procedure:** Detail the hyperparameters (batch size, learning rate, number of epochs), hardware used (e.g., NVIDIA V100 GPU), and training/validation/test splits. You will train one model for each normalization level.
4.  **Evaluation Protocol:**
    * **Test Set:** A held-out set of sentences, specifically designed to include a variety of challenging NSWs (dates, ordinals, currency, abbreviations, long numbers).
    * **Objective Evaluation:**
        * **Mel-Cepstral Distortion (MCD):** Measures the acoustic difference between synthesized and ground-truth spectrograms. Requires time alignment (Dynamic Time Warping).
        * **Word Error Rate (WER) / Character Error Rate (CER):** Use a pre-trained Automatic Speech Recognition (ASR) model for [Target Language] to transcribe the synthesized audio. Compare the ASR output to the fully normalized (Level 3) text. This is an excellent proxy for intelligibility.
    * **Subjective Evaluation:**
        * **Mean Opinion Score (MOS):** The gold standard. Recruit at least 10-15 native speakers. Have them listen to samples from each model (randomized and blinded) and rate the naturalness on a 5-point scale (1-Bad, 5-Excellent).
        * **ABX Preference Test:** Present listeners with pairs of samples from two different models (e.g., Level 1 vs. Level 3) saying the same sentence and ask them which one they prefer.

#### **Chapter 4: Results**
1.  **Objective Results:**
    * Present MCD and WER/CER results in a clear table, comparing Model 0, Model 1, Model 2, and Model 3.
    * Use bar charts to visualize the performance differences.
2.  **Subjective Results:**
    * Present MOS scores with means and 95% confidence intervals in a table and a bar chart.
    * Present the results of the ABX preference tests (e.g., "Model 3 was preferred over Model 1 in 85% of cases").
3.  **Qualitative Analysis & Case Studies:**
    * This is crucial. Provide specific examples.
    * Choose a few challenging sentences from your test set.
    * For each sentence, present the audio generated by each of the four models.
    * Analyze the errors made by the lower-level models. For instance, show how Model 0 tries to pronounce `€` or how Model 2 spells out a year (`1-9-9-5`) instead of verbalizing it (`nineteen ninety-five`).

#### **Chapter 5: Discussion**
1.  **Interpretation of Results:**
    * Analyze *why* the results occurred. Connect the performance increase directly to the normalization level.
    * Discuss the answers to your research questions based on the data. Was the hypothesis confirmed?
    * Discuss the trade-offs. How much effort was Level 3 normalization, and was the MOS improvement from Level 2 to Level 3 worth it?
2.  **Error Analysis:**
    * Categorize the types of errors that persisted even in the best model. Does it still struggle with very rare abbreviations or complex grammatical constructions?
3.  **Limitations of the Study:**
    * Acknowledge any limitations. (e.g., size of the dataset, limited number of participants in the MOS study, only one TTS architecture was tested).

#### **Chapter 6: Conclusion and Future Work**
1.  **Summary of Findings:**
    * Concisely summarize the problem, methodology, and key results.
2.  **Conclusion:**
    * State the main takeaway: For low-resource, morphologically-rich languages, explicit, linguistically-informed text normalization is not an optional preprocessing step but a fundamental requirement for building a high-quality E2E TTS system.
3.  **Future Work:**
    * Investigate hybrid approaches: Can a model be trained to learn simple normalization while rules handle complex cases?
    * Apply the created normalization module to other speech technologies like ASR.
    * Explore more advanced TTS architectures (e.g., VITS) to see if they are more robust to un-normalized text.
    * Develop a neural text normalization module for the target language and compare it to the rule-based one.

---

### **Project Plan**

**1. Research & Planning**
* Literature Review, Finalize RQs
* Data Sourcing & Cleaning
**2. Development**
* Build Normalization Pipeline (Levels 0-3)
* Setup TTS Model & Training Environment
**3. Experimentation**
* Train all 4 TTS Models
* Generate Audio Samples for Evaluation
**4. Evaluation**
* Run Objective Tests (MCD, WER)
* Design & Conduct Subjective MOS/ABX Tests
**5. Writing & Defense**
* Write Chapters 1-3
* Write Chapters 4-5 (Results & Discussion)
* Write Chapter 6, Finalize Thesis
* Prepare Presentation & Defend Thesis

### **Required Resources:**
* **Hardware:** A computer with a powerful GPU (e.g., NVIDIA RTX 3080/4080 or access to a university HPC cluster / cloud computing platform like GCP or AWS).
* **Software:** Python, PyTorch, a TTS framework (e.g., Coqui-TTS), audio processing libraries (librosa, soundfile).
* **Data:** A suitable audio corpus for the chosen target language (at least 10-20 hours of high-quality, single-speaker audio).
* **Human Resources:** Native-speaking volunteers for the subjective evaluation.
