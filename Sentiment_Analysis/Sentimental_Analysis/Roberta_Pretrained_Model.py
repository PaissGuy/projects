import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification
from scipy.special import softmax
from tqdm import tqdm


plt.style.use('ggplot')

class Roberta:

    def __init__(self, data_path='../input/amazon-fine-food-reviews/Reviews.csv'):
        self.df = pd.read_csv(data_path)
        self.df = self.df.head(500)
        MODEL = f"cardiffnlp/twitter-roberta-base-sentiment"
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL)
        self.model = AutoModelForSequenceClassification.from_pretrained(MODEL)


    def Word_Score(self, example):
        encoded_text = self.tokenizer(example, return_tensors='pt')
        output = self.model(**encoded_text)
        scores = output[0][0].detach().numpy()
        scores = softmax(scores)
        scores_dict = {
            'roberta_neg': scores[0],
            'roberta_neu': scores[1],
            'roberta_pos': scores[2]
        }
        return scores_dict

    def Res_Gen(self):
        res = {}
        for i, row in tqdm(self.df.iterrows(), total=len(self.df)):
            try:
                text = row['Text']
                myid = row['Id']
                roberta_result = self.Word_Score(text)
                res[myid] = roberta_result
            except RuntimeError:
                print(f'Broke for id {myid}')

        self.results_df = pd.DataFrame(res).T
        self.results_df = self.results_df.reset_index().rename(columns={'index': 'Id'})
        self.results_df = self.results_df.merge(self.df, how='left')


        fig, axs = plt.subplots(1, 3, figsize=(12, 3))
        sns.barplot(data=self.results_df, x='Score', y='roberta_pos', ax=axs[0])
        sns.barplot(data=self.results_df, x='Score', y='roberta_neu', ax=axs[1])
        sns.barplot(data=self.results_df, x='Score', y='roberta_neg', ax=axs[2])
        axs[0].set_title('Positive')
        axs[1].set_title('Neutral')
        axs[2].set_title('Negative')
        plt.show()

    def run(self):
        self.Res_Gen()



x = Roberta()
x.run()


