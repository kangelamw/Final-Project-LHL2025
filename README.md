**Final-Project-LHL2025:**

# **News Angles: The Bias Decoder**

**README Contents:**

- [Project Details](#project-details)
- [Process Overview](#process-overview)
- [Who benefits?](#who-benefits)
- [Project Resources](#resources)
  - [Pre-Trained Models](#models)
  - [Datasets](#datasets)
- [Final Model](#final-model-fine-tuned-roberta-base-classifier--pegasus-summarizer)
- [Results](#results)
  - [Performance Metrics](#performance-metrics)
  - [Hyperparameters](#hyperparameters)
- [References](#references)
  - [Readings](#readings)
  - [Videos](#videos)
  - [Citations](#citations)

<br>

**Repo File Structure:**

    FINAL-PROJECT-LHL2025
    ├── data
    │   ├── AllSides // Downloaded
    │   ├── eval
    │   ├── Labeled_AllSides.csv // Not included
    ├── images
    ├── LICENSE
    ├── model
    ├── notebooks
    │   ├── 0_data_cleaning_eda.ipynb
    │   ├── 1_model_training.ipynb
    │   ├── 2_model_eval.ipynb
    │   ├── 3_model_deployment.ipynb
    │   ├── functions.py
    │   ├── random.ipynb
    ├── Project_Journal.md
    ├── README.md

<br>

## **Project Details**

An AI tool for analyzing news articles to reveal their political slant. Using machine learning, the system provides **probabilistic scores** across the political spectrum: liberal, center, and conservative.

By breaking down an article's underlying ideological leanings, the tool helps readers critically understand potential narrative biases and unconscious influences in their news consumption.

<br>

## **Process Overview**  

- **Data Preparation:** Preprocess the dataset
- **Bias Classification:** Fine-tune **RoBERTa-base** to predict political bias with a softmax layer, outputting probability scores for liberal, center, and conservative bias.
- **Summarization:** Use **PEGASUS** to generate summaries of the articles for better readability.
- **Visualization:** Display bias probabilities in a spider chart or numerical format.
- **Deployment:** Package into an API and simple frontend for user interaction.

<br>

## **Who Benefits?**  

- **General Public**
  > Equips readers with a critical lens to recognize hidden ideological influences and navigate media manipulation by revealing the subtle ideological currents that can unconsciously shape perception.

- **Fact-Checking Organizations**
  > Transforms bias assessment from subjective guesswork to data-driven analysis, providing a quantitative approach to understanding media political leanings.

- **Journalists & Media Analysts**
  > Shows potential biases in news reporting to promote balance and objectivity in journalism.

<br>

## **Project Resources**  

#### **Models:**  

  1. [RoBERTa-base](https://huggingface.co/FacebookAI/xlm-roberta-base) (base model for classifier)
  2. [google/pegasus-cnn_dailymail](https://huggingface.co/google/pegasus-cnn_dailymail) (pre-trained, for news summarization)

#### **Datasets:**

  - [valurank/PoliticalBias_AllSides_Txt](https://huggingface.co/datasets/valurank/PoliticalBias_AllSides_Txt) (labeled political bias dataset for classifier training)

<br>

## **Model: kangelamw/news-angles-bias-decoder**

[Link to the Model on Huggingface]()

![Model Screenshot]()

<br>

## **Results**  

### **Performance Metrics**

- **Accuracy** (how often the model predicts the correct category)
- **Cross-Entropy Loss** (measures prediction confidence)
- **F1-Score** (balances precision and recall for each bias category)
- **KL-Divergence** (compares predicted probability distributions to true labels)

### **Hyperparameters:**  
**Training Args:**

## **Deployment Options**  
- **API Backend:** Flask/FastAPI on **Azure Functions** or **Google Cloud Run**  
- **Frontend:** Simple UI on **GitHub Pages** for bias visualization  
- **Google Colab Notebook:** Open-source version for reproducibility  

<br>

## **References** 

### Readings
- ****
  - Link
- ****
  - Link

### Videos
- ****
  - Link
- ****
  - Link

### Citations

- google/pegasus-cnn_dailymail
  ```
  @misc{zhang2019pegasus,
      title={PEGASUS: Pre-training with Extracted Gap-sentences for Abstractive Summarization},
      author={Jingqing Zhang and Yao Zhao and Mohammad Saleh and Peter J. Liu},
      year={2019},
      eprint={1912.08777},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
  }
  ```

- FacebookAI/xlm-roberta-base

  ```
  @article{DBLP:journals/corr/abs-1911-02116,
    author    = {Alexis Conneau and
                Kartikay Khandelwal and
                Naman Goyal and
                Vishrav Chaudhary and
                Guillaume Wenzek and
                Francisco Guzm{\'{a}}n and
                Edouard Grave and
                Myle Ott and
                Luke Zettlemoyer and
                Veselin Stoyanov},
    title     = {Unsupervised Cross-lingual Representation Learning at Scale},
    journal   = {CoRR},
    volume    = {abs/1911.02116},
    year      = {2019},
    url       = {http://arxiv.org/abs/1911.02116},
    eprinttype = {arXiv},
    eprint    = {1911.02116},
    timestamp = {Mon, 11 Nov 2019 18:38:09 +0100},
    biburl    = {https://dblp.org/rec/journals/corr/abs-1911-02116.bib},
    bibsource = {dblp computer science bibliography, https://dblp.org}
  }
  ```