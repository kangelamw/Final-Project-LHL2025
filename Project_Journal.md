# Project Journal

> This is where I keep a log of ideas, concepts, and decisions related to this project.

<br>

## Classifier Training thoughts

### Training 01: Model's v1 Performance

<center>

| Metric                  |   Value |
|:------------------------|--------:|
| eval_accuracy           |  0.9204 |
| eval_f1                 |  0.9206 |
| eval_cross_entropy      |  0.2789 |
| eval_kl_divergence      |  0.2789 |

</center>

The model started at 82.11% **accuracy** and was able to get up to 92.03% by epoch 3. The **f1** is essentially the same at 82.18% and 92.05%

![Training Metrics per Epoch v1](/images/v1_training_metrics.png)

![Evaluation Metrics per Epoch v1](/images/v1_eval_metrics.png)

The charts looked decent, but i'm not happy with how the gradient norm sort of exploded during training.

You can also see that loss, KL divergence and cross entropy converged at around tbe 3rd epoch where accuracy and f1 started to plateau.

I restarted training and changed the learning rate (from 2e-5 to 1e-5), lr scheduler type (from linear to cosine) and epochs (5 to 4).

For the epochs, the best model (i think) is between 3 and 4. I'm trying to get it from there.

<br>

### More training runs...

I didn't record it, just eyeballed the metrics that came up and adjusting the training args accordingly until I'm happy or it breaks the 92% of v1. It's already pretty good.

I'm also monitoring the GPU (`nvidia-smi -l 6`) and the model with tensorboard.

It takes about half an hr per training run with my GPU, so I'm not too worried about compute power. The model is only 125M

I have enough time. I'm doing this in the project's Week1 Day4. I'm thinking for ways I could expand this project...

---

### Last Model Trained

<center>

| Metric                  |   Value |
|:------------------------|--------:|
| eval_accuracy           |  0.902  |
| eval_f1                 |  0.9025 |
| eval_cross_entropy      |  0.2775 |
| eval_kl_divergence      |  0.2775 |
| epoch                   |  4      |

</center>

The model started at 66.89% **accuracy** and was able to get up to 90.02% by epoch 4. The **f1** is essentially the same at 66% and 90.24%

![Training Metrics per Epoch v2](/images/v2_training_metrics.png)

![Evaluation Metrics per Epoch v2](/images/v2_eval_metrics.png)

I'll stick with v1. This run looked more unstable than the first.