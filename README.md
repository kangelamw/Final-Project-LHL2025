**Final-Project-LHL2025:**

# **News Angles: The Bias Decoder**

**README Contents:**

- [Project Details](#project-details)
- [Process Overview](#process-overview)
- [Who benefits?](#who-benefits)
- [Project Resources](#project-resources)
- [Model](#model-name-kangelamwnews-angles-bias-decoder)
- [Results](#results)
  - [Performance Metrics](#performance-metrics)
  - [Hyperparameters](#hyperparameters)
- [References](#references)
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
- **Visualization:** Display bias probabilities in a spider chart or numerical format.
- **Deployment:** Package into an API and simple frontend for user interaction.

<br>

## **Who Benefits?**  

- **General Public**
  > Equips readers with a critical lens to recognize hidden ideological influences and navigate media manipulation by revealing the subtle ideological currents that can unconsciously shape perception.

- **Journalists & Media Analysts**
  > Shows potential biases in news reporting to promote balance and objectivity in journalism.

<br>

## **Project Resources**  

#### **Model:**  [RoBERTa-base](https://huggingface.co/FacebookAI/roberta-base) &mdash; base model for classifier

#### **Dataset:** [valurank/PoliticalBias_AllSides_Txt](https://huggingface.co/datasets/valurank/PoliticalBias_AllSides_Txt) &mdash; labeled political bias dataset for classifier training

<br>

## **Model name: kangelamw/news-angles-bias-decoder**

[Link to the Model on Huggingface](https://huggingface.co/kangelamw/RoBERTa-political-bias-classifier-softmax)

![Model Screenshot](/images/model_ss.png)

> You can try it [here!](https://political-bias-classifier.streamlit.app/) It might take a bit to 'wake up' &rarr; I'm on a free tier with Render :>

<br>

## **Results**  

### **Performance Metrics**

- **Accuracy:** how often the model predicts the correct category
- **Cross-Entropy Loss:** measures prediction confidence
- **F1-Score:** balances precision and recall for each bias category
- **KL-Divergence:** compares predicted probability distributions to true labels

<center>

| Metric                  |   Value |
|:------------------------|--------:|
| **eval_accuracy**           |  0.9204 |
| **eval_f1**                 |  0.9206 |
| **eval_cross_entropy**     |  0.2789 |
| **eval_kl_divergence**      |  0.2789 |
| epoch                   |  4.9875 |

</center>

The model started at 82.11% **accuracy** and was able to get up to 92.03% by epoch 3. The **f1** is essentially the same at 82.18% and 92.05%

![Training Metrics per Epoch v1](/images/v1_training_metrics.png)

![Evaluation Metrics per Epoch v1](/images/v1_eval_metrics.png)


### **Hyperparameters:**  

1. Learning Rate: Controls how much the model adjusts weights per step.
    - Lowering LR stabilized training and smoothed out the gradient norm fluctuations. However, final accuracy dropped slightly from 92.03% to 90.02%.
    - Likely means v1’s 2e-5 was close to optimal, and reducing LR slowed learning without significant gains.

2. Learning Rate Scheduler: Defines how the LR decays over time
    - Cosine smoothed training but didn’t outperform linear in final accuracy. Gradient spikes were still present, but they were not extreme.
    - Linear decay worked slightly better for this task.

3. Epochs: Determines how long the model trains/how many times it sees the data.
    - v1’s best model was between epochs 3 and 4.
    - v2’s accuracy plateaued at epoch 4.
    - Running longer (5 epochs) didn’t add much value.

#### Training Args:
  ```python
    TrainingArguments(
      num_train_epochs=4,
      learning_rate=2e-5,
      lr_scheduler_type="linear",
    )
  ```

## **Deployment: Streamlit + Flask + Render**
- **Streamlit:** The UI is built with Streamlit, providing an interactive platform for users to input news articles and visualize bias predictions. The project is hosted as a Streamlit app, making it easily accessible through a web browser.
- **Flask:** A Flask API was created to serve the bias predictions. This API acts as the backend, receiving article text from the Streamlit frontend and returning the processed bias scores.
- **Render:** Render is used to deploy the Flask API as a web service. Streamlit consumes this API to fetch the bias predictions for the articles submitted by users.

<br>

## **References** Deploy a Flask App on Render

Deploy a Flask App on Render
- https://render.com/docs/deploy-flask

### Citations

- FacebookAI/roberta-base
    ```
  @article{DBLP:journals/corr/abs-1907-11692,
    author    = {Yinhan Liu and
                Myle Ott and
                Naman Goyal and
                Jingfei Du and
                Mandar Joshi and
                Danqi Chen and
                Omer Levy and
                Mike Lewis and
                Luke Zettlemoyer and
                Veselin Stoyanov},
    title     = {RoBERTa: {A} Robustly Optimized {BERT} Pretraining Approach},
    journal   = {CoRR},
    volume    = {abs/1907.11692},
    year      = {2019},
    url       = {http://arxiv.org/abs/1907.11692},
    archivePrefix = {arXiv},
    eprint    = {1907.11692},
    timestamp = {Thu, 01 Aug 2019 08:59:33 +0200},
    biburl    = {https://dblp.org/rec/journals/corr/abs-1907-11692.bib},
    bibsource = {dblp computer science bibliography, https://dblp.org}
  }
    ```