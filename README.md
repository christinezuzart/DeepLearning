# Classification of the Medical Condition based on Drug Review using Transfer Learning

# Challenge Description
This problem is based on multi-class classification wherein medical domain dataset is provided.

Each sample set is the description of patient health condition or problem he/she is facing.

Given the description of patient's problem the model should understand it and predict about the disease or medical condition.

Inorder to complete this challenge, Deep Learning models with transfer learning needs to be used.

Evaluation Critera: For the performance we will consider accuracy as Predicted Label / True Label . Minimum accuracy for this problem is 85%.

# Current trends
Text Classification is a classic problem in the field of Machine Learning.

There was a huge hype of transfer learning in NLP in recent times.

It started with ULMFit by Jeremy Howard et. al. from fast.ai, ELMo, GLoMo, Open AI Transformer, Google's BERT, OpenAI’s GPT-2.

These pre-trained models enhance the language modeling capabilities and eases out the problem of text classification.

# Problem Statement Understanding
Given the medical reviews of the patients sypmtoms and the drugs the patient is administered the model should predict the disease or the medical condition.

I tried to ponder on the application of this in real world and felt that it may act as a virtual assist to a new bee medical practitioner who is seeing a patient suffering from a medical problem or it may help a pharmacist.

Proposed Solution & Rationale of using models and tuning parameters
This challenge insisted on using pre-trained model.

I looked for the list of pre-trained models available. Came across this list put down beautifully by analyticsvidyha. https://www.analyticsvidhya.com/blog/2019/03/pretrained-models-get-started-nlp/

ULMFit, Google's BERT, OpenAI’s GPT-2 are the ones I plan to experiment before freezing on the best solution.

I started with ULMFit from fast.ai .

Being new to NLP I started with fast.ai's course on NLP. The transfer learning lecture delivered by Jeremy Howard on the imdb dataset put things right on what are the essential building blocks to do so. This is what I referred to.

https://github.com/fastai/fastai/blob/master/examples/ULMFit.ipynb

I tried to study similiar examples using ULMFit to gather more understanding on how fine tuning is done. Stance Classification of Tweets using Transfer Learning was one example https://github.com/prrao87/tweet-stance-prediction/blob/master/ulmfit.ipynb .

My work is still in progess in terms of tuning parameters.I have used the default parameters specified above.

# Data and Preprocessing
Input Dataset:

Number of training examples = 161297

Number of testing examples = 53766

Training and Testing Dataset exploration gave good insights.

There were examples where condition was blank.There were such examples present in both datasets.

There were some examples where the condition was not appropriate as in the real world scenario.Its of the form where it is mentioned as "</span> users found this comment helpful."

There were some conditions present in the Test dataset that were not present in the Training dataset.

Some conditions had a few number of examples.

The reviews had html text.

# Network architecture
The network architecture has the belo building blocks:-

General-Domain Language Model Pretraining:

A LM is pretrained on a large general-domain corpus. In ULMFit it refers to the WikiText-103 dataset. This model is able to predict the next word in a sequence. The model learns the general features of the language, e.g. the sentence structure of the English language is verb or a noun.

Target Task Language Model Fine-Tuning:

The Language model in this stage is fine tuned for our target dataset i.e Drug Reviews. The target dataset is of a different distribution than the WikiText-103 dataset. As I understand , there may be specific terms or features in this Drug Review dataset that comes from Medical domain where the language model can learn.

Target Task Classifier: Our goal is to provide text classification for the reviews in our dataset. The pretrained LM is expanded by two linear blocks so that the final output is a probability distribution over the medical conditions.
Performance measures
For the performance we will consider accuracy as Predicted Label / True Label .

The current model predicts 75.85% accuracy.

# Open source/Research references
https://www.fast.ai/2019/07/08/fastai-nlp/

https://www.analyticsvidhya.com/blog/2019/03/pretrained-models-get-started-nlp/

https://github.com/prrao87/tweet-stance-prediction/blob/master/ulmfit.ipynb

https://github.com/fastai/fastai/blob/master/examples/ULMFit.ipynb

https://arxiv.org/abs/1801.06146

# Tools & Framework Used
Used the Jupyter notebook environment provided by Google Colab. https://colab.research.google.com/

This requires no setup.

# Future work
Experimentation on Google's BERT, OpenAI’s GPT-2 pre-trained models is desired.

These would give better insights in terms of performance of the model on the given dataset
