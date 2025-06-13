import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import nltk #Natural Language Toolkit
from nltk.sentiment import SentimentIntensityAnalyzer
from tqdm import tqdm


plt.style.use('ggplot')


class VADER:
    """
    This class uses old school sentiment analysis via VADER (Valence aware dictionary and Sentiment Reasoner)
    """

    def __init__(self, data_path = '../input/amazon-fine-food-reviews/Reviews.csv'):
        self.df = pd.read_csv(data_path)
        self.df = self.df.head(500)
        self.vaders = pd.DataFrame()

    def Exploratory_Data_Analysis(self, field_name='Score', title = 'Count of reviews by stars', xlabel = 'Review Stars'):
        ax = self.df[field_name].value_counts().sort_index() \
            .plot(kind='bar',
                  title = title,
                  figsize = (10,5))
        ax.set_xlabel(xlabel)
        plt.show()

    def Tokenize_Tag(self, field_name = 'Text', index = 50):
        example = self.df[field_name][index]
        self.tokens = nltk.word_tokenize(example)
        self.tags = nltk.pos_tag(self.tokens)

    def Word_Score(self):
        '''
        Run polarity score over our dataset.
        :return:
        '''
        sia = SentimentIntensityAnalyzer()
        res = {}
        for i, row in tqdm(self.df.iterrows(), total=len(self.df)):
            text = row['Text']
            myid = row['Id']
            res[myid] = sia.polarity_scores(text)
        self.vaders = pd.DataFrame(res).T
        self.vaders = self.vaders.reset_index().rename(columns={'index': 'Id'})
        self.vaders = self.vaders.merge(self.df, how='left')

        ax = sns.barplot(data=self.vaders, x='Score', y='compound')
        ax.set_title('Compound Score by Amazon Star Review')
        fig, axs = plt.subplots(1, 3, figsize=(12, 3))
        sns.barplot(data = self.vaders, x = 'Score', y = 'pos', ax = axs[0])
        sns.barplot(data = self.vaders, x = 'Score', y = 'neu', ax = axs[1])
        sns.barplot(data = self.vaders, x = 'Score', y = 'neg', ax = axs[2])
        axs[0].set_title('Positive')
        axs[1].set_title('Neutral')
        axs[2].set_title('Negative')
        plt.show()

    def run(self):
        self.Exploratory_Data_Analysis()
        self.Word_Score()




x = VADER()
x.run()


