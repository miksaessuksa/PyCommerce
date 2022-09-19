import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import io
#import plotly.figure_factory as ff
import plotly.express as px
import plotly.graph_objects as go
import re
import datetime as dt
import base64
from PIL import Image

#import scikitplot as skplt
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import precision_recall_curve
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import f1_score, precision_score,recall_score

from sklearn.model_selection import train_test_split
#from sklearn.linear_model import LogisticRegression    
from sklearn.metrics import plot_confusion_matrix, plot_roc_curve
from catboost import CatBoostClassifier
#from sklearn.svm import SVC
from scipy.sparse import hstack
import joblib
from sklearn.feature_selection import SelectKBest

import networkx as nx
from pyvis.network import Network
from community.community_louvain import best_partition
import streamlit.components.v1 as components
import collections
from wordcloud import WordCloud
from sklearn.feature_extraction.text import CountVectorizer


#Creation of a dataframe with with the data from the file "reviews_trust.csv":
df=pd.read_csv('reviews_trust.csv', index_col=0)

#Setting option to show max rows and max columns
pd.set_option("display.max_columns",None)
pd.set_option("display.max_rows", None)


#Sidebar creation: 
st.sidebar.markdown('# PyCommerce')
rad = st.sidebar.radio('MENU', ("Project presentation", "Explorative Data Analysis", "Data processing", "Modeling","Evaluation", "Conclusion & Perspectives", 'Project team'))
    

def add_bg_from_local(image_file):
    with open(image_file, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read())
    st.markdown(
    f"""
    <style>
    [data-testid="stSidebar"] > div:first-child {{
        background-image: url(data:image/{"jpg"};base64,{encoded_string.decode()});
        background-size: cover
    }}
    </style>
    """,
    unsafe_allow_html=True
    )
add_bg_from_local('pycommerce_bg.jpg')


#################PROJECT PRESENTATION
if rad == "Project presentation":
    st.header("PyCommerce Project")
    st.subheader('Context')
    st.markdown("Nowadays, with the prevalence of costumer reviews and review platforms, it’s little surprising that they can play a crucial role in understanding the parameters that affects the performance of a business in the market. In order to stand out from competitors, it is crucial for e-commerce companies to understand clients' pain points.")
    st.markdown(" Data sciences methods such as **sentiment analysis**, allow businesses to extract values from customers' feedback. Identifying areas of improvements, strenghts, and product trends represent a strong advantage to plan out deadlines and optimize decision making.")
    st.markdown("Sentiment analysis is made possible by the use of **Natural Language Processing methods (NLP)** or which consists in understandingand manipulating natural language by the machines.")
    st.markdown ("Throughout this project, we will analyze clients'feedback for the Trusted Shop website originating from two sources: **Trusted Shop**: verified comments, meaning they result from a client order.**Trustpilot**: comments from internet users.")
    st.markdown("After analyzing customers' comments, our objective will be to categorize new product feedback using a **binary classification**")
    st.subheader('Steps')
    st.markdown("In order to reach our goal, we will divide our project in four steps, based on a data siences approach: ")
    st.markdown(">- **Exploratory Data Analysis (EDA)**: a mandatory step to comprehend and understand our datase (descriptive data, correlation between variables, data cleaning, visualization graphs...) ")
    st.markdown(">- **Data preprocessing**: NLP methods require to transform raw data (comments) in usable and workable data (tokenization, stemming, regular expression...).")
    st.markdown(">- **Modeling**: training machine learning and deep learning models and interpreting results.")
    st.markdown(">- **Predictions**: test driving the model chosen upon selection criteria (interpretability & performances).")



###############EDA
elif rad == "Explorative Data Analysis":
    st.header('Explorative Data Analysis')
    st.markdown('### Data description and preparation')
    st.markdown('> #### 1. Dataset discovery')


#Visualize the first lines of the dataframe:
    nb_rows=st.number_input("Choose the number of rows you want to display:", 0, 19000, 5)
    st.write(df.head(nb_rows))

#Visualize information and statistics about the dataframe:
    buffer = io.StringIO()
    df.info(buf=buffer)
    info = buffer.getvalue()
    
    st.markdown ("The dataset includes 19,863 entries for 11 variables (2 are numerical, while the others are categorical) : \n"
                 ">- **Commentaire**: feedback left by the customer  \n"
                 ">- **star**: rating (1 to 5)  \n"
                 ">- **date**: date of customer's feedback  \n"
                 ">- **client**: lots of missing values (  %)  \n"
                 ">- **reponse**: answer to customers'feedback, lots of missing values (%)  \n"
                 ">- **source**: the reviews have been collected from TrustedShop and TrustPilot  \n"
                 ">- **company**: the reviews refer to 2 e-commerce platforms (VeePee and ShowRoom)  \n"
                 ">- **ville**: lots of missing values  \n"
                 ">- **date_commande**: order date, lots of missing values  \n"
                 ">- **ecart**: day interval between customer's order and feedback, lots of missing values  \n"
                 ">- **maj**: delivery date  \n")
    if st.button('Click if you want to read general information regarding the dataset'):
        st.text(info)                

#CLEANING DUPLICATES AND NA
    st.markdown('> #### 2. Cleaning of Duplicates & NaN')
    
    #Check for duplicates:
    nb_duplicates=df.duplicated().sum()
    st.markdown("The dataset includes 427 duplicates, that will be removed so as to only keep unique entries.  \n"
                "It also includes 80 446 missing values. We will remove the columns with a large amount of missing values (more than 45 %:client,reponse,ville,maj, date_commande, ecart) that cannot be used for the analysis, and we will drop the few remaining rows with missing values.  \n"
                "We also noticed few reviews in other languages while exploring the dataset and are going to check for the number of non-french reviews by using the language detection library langdetect (99% over precision for 49 languages, developed by Shuyo Nakatani).")
    status_1 = st.radio("Select the variable you want to display: ", ('None','Duplicates', 'Missing values'))
    if status_1=='None':
        st.markdown('')
    elif status_1=='Duplicates':
        st.write(df[df.duplicated()])
    else :
        st.write(df.isna().sum())
    #Check for missing values:
    #Delete variables with a large proportion of missing values:
    df1=df.copy()
    df1=df1.drop(['ville','client','maj','date_commande','ecart','reponse'],axis=1)
            
    #Delete the few rows with missing values:
    df1=df1.dropna()
    df1.reset_index(drop=True, inplace=True)
     
    st.markdown("> #### 1.3. Identification of existing Languages")
    #Load the reviews including the corresponding language detected:
    df1=pd.read_csv('df_lang4.csv',index_col=0)
    st.markdown("The dataset includes 32 different languages with Italian, Portuguese, Spanish and English among the top ten of foreign languages detected, with some languages being misidentified.  \n\n"
                "Considering that most available NLP libraries and models  have been been trained on English, which makes practical use of such models in French quite limited, we will select only **French reviews** for further analysis. \n")
    lang=df1['langue'].value_counts().rename_axis('Langue').reset_index(name='counts')
    if st.button('Click if you want to see the count of all languages used in the dataset'):
        st.write(lang)
#selection of top ten languages
    top_lang=lang.head(10).replace(['fr','it','pt','lv','es','en','ro','ca','sk','de'],['French','Italian','Portuguese','Latvian','Spanish','English','Romanian','Catalan','Slovak','German'])
    
#Plotting Teemap of the top 10 languages
    fig = px.treemap(top_lang, path=['Langue'], values='counts',color='Langue', color_discrete_map={"French": "#4628dd", "Italian": "#004cf2","Portuguese": "#0065ff", "Spanish": "#0079ff","Latvian":"#008bff","English":"#009bff", "Catalan":"#00aaff","Romanian":"#00b7f7",  "Slovak":"#00c4ee", "German":"#00d1e6"})
    st.markdown('##### Top 10 languages detected in the reviews')
    st.plotly_chart(fig)
#We keep only French identified reviews:
    df1_fr=df1.copy()
    df1_fr=df1_fr[df1_fr['langue']=='fr']
    df1_fr.reset_index(drop=True, inplace=True)
    df1_fr= df1_fr.drop('langue', axis=1)

#DATA VISUALIZATION    
    st.markdown("### Data Visualization")
   #Ratings distribution
    st.markdown("Let's first have a look at the ratings distribution that will be our target variable:")
#Pie chart
    reviews_count = df1_fr.groupby('star').count()['Commentaire'].reset_index().sort_values(by='Commentaire',ascending=False)
    fig = px.pie(df1_fr, values=reviews_count.Commentaire, names=reviews_count.star)
    fig.update_traces(textfont_size=14,textinfo='label+percent')
    st.markdown('##### Star Rating distribution')
    st.plotly_chart(fig)
    st.markdown("The dataset is imbalanced with 1 and 5 star-rated reviews accounting for nearly 65% of the reviews")
   
    #Check for reviews length 
    def words_counter (text):
        r = re.compile(r'\w+')
        nb_words = r.findall(text)
        return len(nb_words) 
    df1_fr['nb_words'] = df1_fr.Commentaire.apply(lambda x: words_counter(x))

#Characters count:
    df1_fr['char_count'] = df1_fr.Commentaire.apply(lambda x: len([ele for ele in x if ele.isalpha()]))

#Sentences count:
    df1_fr['sentences_count'] = df1_fr.Commentaire.apply(lambda x: len(str(x).split(".")))

#Average words and sentences length:
    df1_fr['avg_word_length'] = (df1_fr.char_count / df1_fr.nb_words).round(1)
    df1_fr['avg_sentence_length'] = (df1_fr.nb_words/ df1_fr.sentences_count).round(1)   
#Function to get the descriptive statistics of a column:
    def show_dist(df, col_group,col):
        st.write(df.groupby(col_group)[col].describe(), "\n")
        
    st.markdown("Let's see if we can identify some interesting features that correlate with the reviews rating. ")
    
    # Change date format
    df1_fr['date'] = pd.to_datetime(df1_fr['date'], format='%Y-%m-%d', utc=True)
    df1_fr['year'] = df1_fr.date.dt.year.astype(int)
    df1_fr['month'] = df1_fr['date'].dt.month
    df1_fr['weekday'] = df1_fr['date'].dt.weekday
    df1_fr['quarter'] = df1_fr.month.replace([1,2,3,4,5,6,7,8,9,10,11,12],[1,1,1,2,2,2,3,3,3,4,4,4])
    #Remove variable "date"
    df1_fr=df1_fr.drop('date', axis=1)
   
    
    status2 = st.radio("Select the variable you want to display: ", ('None','Text length', 'Source','Company', 'Source and Company','Date', ))
#Words count:
    if (status2=='None'):
        st.markdown('')
    elif (status2=='Text length'):
     #Descriptive stats of the variable "nb_words", and distribution:
        st.markdown('##### Word count')
        st.markdown('**Descriptive statistics for word count:**\n')
        show_dist(df1_fr, 'star','nb_words')
    #WORDS COUNT distribution
        sns.set() 
        x, y = "nb_words", "star"
        fig, ax = plt.subplots(1, 3,figsize=(16,5))
        # Set figures size (width, height)
        #fig.suptitle('Word count', fontsize=16)
        for i in df1_fr[y].unique():
            # Plot the first figure: 
            sns.distplot(df1_fr[df1_fr[y]==i][x], hist=True, kde=False, bins=10, hist_kws={"alpha":0.6}, axlabel="Word count",ax=ax[0]) 
            # Plot the second figure:
            sns.distplot(df1_fr[df1_fr[y]==i][x], hist=False, kde=True, kde_kws={"shade":True}, axlabel="Word count",  ax=ax[1])
            # Plot the third figure: 
            my_pal = ['#4628dd', '#4628dd', '#4628dd', '#27DCE0', '#27DCE0']
            sns.boxplot(x='star', y='nb_words', palette=my_pal, 
                data=df1_fr) 
                    
        #Set legend and titles
        ax[0].set_title('Histogram')
        ax[0].set_ylabel('Frequency')
        ax[0].grid(True)
        ax[0].legend(df1_fr[y].unique())
        ax[1].set_title('Density')
        ax[1].grid(True)
        ax[2].set_title('Barplot')
        ax[2].grid(True)
        ax[2].set_xlabel( "Star rating" )
        st.markdown('**Word count distribution per Star rating:**\n')
        st.pyplot(fig)
        st.write("Word count to Star rating Correlation:",df1_fr.nb_words.corr(df1_fr.star).round(2))
   
   
#SENTENCES COUNT distribution
        st.markdown('##### Sentence count')
        st.write('**Descriptive stats for sentence count:**')
        show_dist(df1_fr, 'star','sentences_count')
        sns.set() 
        x, y = "sentences_count", "star"
        fig, ax = plt.subplots(1, 3,figsize=(16,5))
        # Set figures size (width, height)
        #fig.suptitle('Sentence count', fontsize=16)
        for i in df1_fr[y].unique():
            # Plot the first figure: 
            sns.distplot(df1_fr[df1_fr[y]==i][x], hist=True, kde=False, 
                 bins=10, hist_kws={"alpha":0.6}, 
                 axlabel="Sentence count",ax=ax[0])
    # Plot the second figure:
            sns.distplot(df1_fr[df1_fr[y]==i][x], hist=False, kde=True, 
                 kde_kws={"shade":True}, axlabel="Sentence count",   
                 ax=ax[1])
# Plot the third figure: 
        my_pal = ['#4628dd', '#4628dd', '#4628dd', '#27DCE0', '#27DCE0']
        sns.boxplot(x='star', y='sentences_count', palette=my_pal,data=df1_fr)  
       
#Set legend and titles
        ax[0].set_title('Histogram')
        ax[0].set_ylabel('Frequency')
        ax[0].grid(True)
        ax[0].legend(df1_fr[y].unique())
        ax[1].set_title('Density')
        ax[1].grid(True)
        ax[2].set_title('Barplot')
        ax[2].grid(True)
        ax[2].set_xlabel( "Star rating" )
        st.markdown('**Sentence count distribution per Star rating:**\n')
        st.pyplot(fig)
        st.write("Sentence count to Rating Correlation:",df1_fr.sentences_count.corr(df1_fr.star).round(2))
        st.markdown("The majority of reviews are less than 50 words and less than 4 sentences. Lowest rated reviews correlate with a higher number of words and sentences, and also present more outliers.")
           
    elif (status2=='Company'):
#Ratings distribution of the ratings per company:
    
        df_ShowRoom = df1_fr[df1_fr['company']=='ShowRoom']
        df_ShowRoom=df_ShowRoom['star'].value_counts().rename_axis('star').reset_index(name='count').sort_values(by='star')
        df_ShowRoom['company']='ShowRoom'
        df_VeePee = df1_fr[df1_fr['company']=='VeePee']
        df_VeePee=df_VeePee['star'].value_counts().rename_axis('star').reset_index(name='count').sort_values(by='star')
        df_VeePee['company']='Veepee'
        df_company = pd.concat([df_ShowRoom, df_VeePee], axis=0)


        fig=px.funnel(df_company, x='star', y='count', color='company',color_discrete_map={"ShowRoom": "#4628DD", "Veepee": "#27DCE0"})
        st.markdown('##### Ratings distribution per Company')
        st.plotly_chart(fig)

    elif (status2=='Source'):
#Ratings distribution per source:
        st.markdown ("Interestingly, we notice that the rating trends for TrustedShop and TrustPilot are inverted, with TrustPilot displaying mostly low-rated reviews, while TrustedShop records mainly 5 stars rated reviews (Fig. 6), which could suggest some review gating strategy, where happy customers are encouraged to leave reviews.")

        df_TrustPilot = df1_fr[df1_fr['source']=='TrustPilot']
        df_TrustPilot =df_TrustPilot ['star'].value_counts().rename_axis('star').reset_index(name='count').sort_values(by='star')
        df_TrustPilot ['source']='TrustPilot'
        df_TrustedShop = df1_fr[df1_fr['source']=='TrustedShop']
        df_TrustedShop=df_TrustedShop['star'].value_counts().rename_axis('star').reset_index(name='count').sort_values(by='star')
        df_TrustedShop['source']='TrustedShop'
        df_source = pd.concat([df_TrustPilot, df_TrustedShop], axis=0)
    
        fig=px.funnel(df_source, x='star', y='count', color='source', color_discrete_map={"TrustPilot": "#4628DD", "TrustedShop": "#27DCE0"},title =  "**Ratings distribution per source**")
        st.markdown('##### Ratings distribution per Source')
        st.plotly_chart(fig)

    elif (status2=='Source and Company'):
#Ratings per company and per source:
    #ShowRoom
        df_ShowRoom_TP = df1_fr[(df1_fr['company']=='ShowRoom')&(df1_fr['source']=='TrustPilot')]
        df_ShowRoom_TP=df_ShowRoom_TP['star'].value_counts().rename_axis('star').reset_index(name='count').sort_values(by='star')
        df_ShowRoom_TP['source']='TrustPilot'
        df_ShowRoom_TS = df1_fr[(df1_fr['company']=='ShowRoom')&(df1_fr['source']=='TrustedShop')]
        df_ShowRoom_TS=df_ShowRoom_TS['star'].value_counts().rename_axis('star').reset_index(name='count').sort_values(by='star')
        df_ShowRoom_TS['source']='TrustedShop'
        df_company = pd.concat([df_ShowRoom_TP, df_ShowRoom_TS], axis=0)
        fig = px.funnel(df_company, x='star', y='count', color='source',
                color_discrete_map={"TrustPilot": "#4628DD", 
                                    "TrustedShop": "#27DCE0"},         
                )
        fig.update_layout(
    
        height=600, 
        width=800
        )
        st.markdown('##### Showroom Ratings distribution per source')
        st.plotly_chart(fig)

    #VeePee
        df_VeePee_TP = df1_fr[(df1_fr['company']=='VeePee')&(df1_fr['source']=='TrustPilot')]
        df_VeePee_TP=df_VeePee_TP['star'].value_counts().rename_axis('star').reset_index(name='count').sort_values(by='star')
        df_VeePee_TP['source']='TrustPilot'
        df_VeePee_TS = df1_fr[(df1_fr['company']=='VeePee')&(df1_fr['source']=='TrustedShop')]
        df_VeePee_TS=df_VeePee_TS['star'].value_counts().rename_axis('star').reset_index(name='count').sort_values(by='star')
        df_VeePee_TS['source']='TrustedShop'
        df_company = pd.concat([df_VeePee_TP, df_VeePee_TS], axis=0)
        fig = px.funnel(df_company, x='star', y='count', color='source',
                color_discrete_map={"TrustPilot": "#4628DD"}, 
                )
        fig.update_layout(
        
        height=600, 
        width=800
        )
        st.markdown('##### VeePee Ratings distribution per source')
        st.plotly_chart(fig)
    
        st.markdown("The majority of reviews have been collected via TrustedShop and refer to ShowRoom, with only TrustPilot Reviews for VeePee.  \n")


    
    #Evolution of Ratings over Time
    else:
        st.markdown("Ratings have been collected for over 5 years")
        st.write('date min: 2015-10-02') #2015-10-02
        st.write ('date max:2021-06-20') #'2021-06-20'


        def reviews_per_year(year):
            reviews_count_year = df1_fr[df1_fr['year']==year].groupby('star').count()['Commentaire'].reset_index()
            return reviews_count_year
#.sort_values(by='star',ascending=True
        reviews_2015=reviews_per_year(2015)
        reviews_2016=reviews_per_year(2016)
        reviews_2017=reviews_per_year(2017)
        reviews_2018=reviews_per_year(2018)
        reviews_2019=reviews_per_year(2019)
        reviews_2020=reviews_per_year(2020)
        reviews_2021=reviews_per_year(2021)

        plotdata=pd.DataFrame({'2015':pd.Series(reviews_2015['Commentaire']),
          '2016':pd.Series(reviews_2016['Commentaire']),
          '2017':pd.Series(reviews_2017['Commentaire']),
          '2018':pd.Series(reviews_2018['Commentaire']),
          '2019':pd.Series(reviews_2019['Commentaire']),
          '2020':pd.Series(reviews_2020['Commentaire']),
          '2021':pd.Series(reviews_2021['Commentaire'])},
        )
  
        index=['1','2','3','4','5']
        colors = colors=px.colors.sequential.Viridis
        #colors = ['#93b5ff', '#9fbcff','#abc4ff', '#c1d3fe', '#ccdbfd', '#d7e3fc', '#edf2fa']
        # plotly
        fig = px.bar(plotdata, 
             x = index,
             y = [c for c in plotdata.columns],
             
             color_discrete_sequence = colors,
             
             )
        fig.update_xaxes(title="Star")
        fig.update_yaxes(title="Count")
        fig.update_layout(legend_title_text='Year') 
        st.markdown('##### Ratings count per year')
        st.plotly_chart(fig)
    
        def reviews_per_month(month):
            reviews_count_month = df1_fr[df1_fr['month']==month].groupby('star').count()['Commentaire'].reset_index()
            return reviews_count_month
        #.sort_values(by='star',ascending=True
        reviews_Jan=reviews_per_month(1)
        reviews_Feb=reviews_per_month(2)
        reviews_March=reviews_per_month(3)
        reviews_Apr=reviews_per_month(4)
        reviews_May=reviews_per_month(5)
        reviews_Jun=reviews_per_month(6)
        reviews_Jul=reviews_per_month(7)
        reviews_Aug=reviews_per_month(8)
        reviews_Sep=reviews_per_month(9)
        reviews_Oct=reviews_per_month(10)
        reviews_Nov=reviews_per_month(11)
        reviews_Dec=reviews_per_month(12)

        plotdata_month=pd.DataFrame({'January':pd.Series(reviews_Jan['Commentaire']),
          'February':pd.Series(reviews_Feb['Commentaire']),
          'March':pd.Series(reviews_March['Commentaire']),
          'April':pd.Series(reviews_Apr['Commentaire']),
          'May':pd.Series(reviews_May['Commentaire']),
          'June':pd.Series(reviews_Jun['Commentaire']),
          'July':pd.Series(reviews_Jul['Commentaire']),
          'August':pd.Series(reviews_Aug['Commentaire']),           
          'September':pd.Series(reviews_Sep['Commentaire']),           
          'October':pd.Series(reviews_Oct['Commentaire']),           
           'November':pd.Series(reviews_Nov['Commentaire']),
            'December':pd.Series(reviews_Dec['Commentaire']),
               },
        )

        index=['1','2','3','4','5']

#colors = px.colors.qualitative.Safe
        colors=px.colors.sequential.Viridis

# plotly
        fig = px.bar(plotdata_month, 
             x = index,
             y = [c for c in plotdata_month.columns],
             color_discrete_sequence = colors,
              
             )
        fig.update_xaxes(title="Star")
        fig.update_yaxes(title="Count")
        fig.update_layout(legend_title_text='Month') 
        st.markdown('##### Ratings count per month')
        st.plotly_chart(fig)
    
        def reviews_per_quarter(quarter):
            reviews_count_quarter = df1_fr[df1_fr['quarter']==quarter].groupby('star').count()['Commentaire'].reset_index()
            return reviews_count_quarter
    #.sort_values(by='star',ascending=True
        reviews_Q1=reviews_per_quarter(1)
        reviews_Q2=reviews_per_quarter(2)
        reviews_Q3=reviews_per_quarter(3)
        reviews_Q4=reviews_per_quarter(4)
    
        plotdata_quarter=pd.DataFrame({'Q1':pd.Series(reviews_Q1['Commentaire']),
              'Q2':pd.Series(reviews_Q2['Commentaire']),
              'Q3':pd.Series(reviews_Q3['Commentaire']),
              'Q4':pd.Series(reviews_Q4['Commentaire']),
                                      },
            )
        
        index=['1','2','3','4','5']
    
    #colors = px.colors.qualitative.Prism  
    
        colors=px.colors.sequential.Viridis
    
    # plotly
        fig = px.bar(plotdata_quarter, 
                 x = index,
                 y = [c for c in plotdata_quarter.columns],
                 
                 color_discrete_sequence = colors,
            
                 )
        fig.update_xaxes(title="Star")
        fig.update_yaxes(title="Count")
        fig.update_layout(legend_title_text='Quarter') 
        st.markdown('##### Ratings count per quarter')
        st.plotly_chart(fig,use_container_width=True)
        
        st.markdown("The count of 1 star reviews is quite homogenous all along the year while the rest of the reviews have been mainly recorded during the summer : June, July, August (Fig. ).  \n"  
                    "Interestingly, we notice a drastic increase in feedback recorded during 2020 (together with an increase in high-rated reviews compared to the previous years), with a decrease in 2021.  \n\n")  
    st.markdown( "Considering the difficulty to identify specific vocabulary to distinguish between ratings varying by only 1 star, we simplified the rating system by pooling the reviews having a negative tone (1, 2 and 3 star ratings replaced by 0) and reviews having a positive tone (4 and 5 stars ratings replaced by 1). The prediction problem is now reduced to a binary classification and the new rating distribution is now more balanced with 44.8% of negative reviews versus 55.2% of positive reviews")
    
    #Replace 4 star rating per positive and negative ratings (1 and 0, respectively)
    df1_fr['rating'] = df1_fr.star.replace([1,2,3,4,5],[0,0,0,1,1])
    #New Ratings distribution:
    colors = ['#27DCE0', '#4628DD']
    reviews_count = df1_fr.groupby('rating').count()['Commentaire'].reset_index().sort_values(by='Commentaire',ascending=False)
    fig = px.pie(df1_fr, values=reviews_count.Commentaire, names=reviews_count.rating)
    fig.update_traces(textfont_size=14,marker=dict(colors=colors),textinfo='label+percent')
    fig.update_layout(  
    font_color="black",
    autosize=False,
    width=400,
    height=400
   )
    st.markdown('##### New Rating Distribution')
    st.plotly_chart(fig)

#SENTIMENT ANALYSIS    
    st.markdown("### Sentiment analysis")
    st.markdown("Let's go further into the comments and identify the most redundant key words to have a first flavour of the general sentiment. For this purpose we will use mainly **NLTK** and **Spacy** libraries. \n")
    st.markdown("With help of these librairies, we went through several manipualtion as tokenization, lemmatization, POS tagging, etc...")  
    if st.button('Click here if you want to discover the full cleaning process we went through'):
        st.image('text_processing.jpg')
    
    st.markdown('The first result of this sentiment analysisis is given by the word cloud below.')    
    
    df1_fr = pd.read_csv('df_text_V6neg_joined.csv',index_col=0)
   
    #Grouping texts per sentiment. Positive, negative and neutral
    def MyWordCloud(data, title,background):
        """Function that takes 3 arguments: data (text input), title for the WordCloud, and color of background
        for the Wordcloud, and returns a WordCloud"""
        #Define parameters of the WordCloud
        wordcloud = WordCloud(                          
            background_color=background,
            #stopwords = stop_words,
            max_words=100,
            max_font_size=40,
            scale=3,
            random_state=1).generate(str(data))
        #Plot the WordCloud:
        fig = plt.figure(1, figsize=(20, 20))
        plt.axis('off')
        if title:
            fig.suptitle(title,fontsize=30)
            fig.subplots_adjust(top=2.25)
        plt.imshow(wordcloud)
        plt.show()
        st.pyplot(fig)
    
    st.markdown("---") 
    st.markdown("***Word Cloud Visualization for each category of comments***")
    slctbox_status = st.selectbox("Select which category of comments to display the corresponding word cloud",['Positive', 'Negative', 'Neutral'])
    if (slctbox_status == 'Negative') :
        df1_fr_neg = df1_fr[df1_fr.rating == 0]
        MyWordCloud(df1_fr_neg.No_stopwords_joined,title="Positive reviews Wordcloud\n\n", background='black')
    
    elif (slctbox_status == 'Positive'):
        df1_fr_pos = df1_fr[df1_fr.rating == 1]
        MyWordCloud(df1_fr_pos.No_stopwords_joined,title="Negative reviews Wordcloud\n\n",background='white')
    
    elif (slctbox_status == 'Neutral'):
        df1_fr_neutral = df1_fr[df1_fr.star == 3]
        MyWordCloud(df1_fr_neutral.No_stopwords_joined,title="Neutral reviews Wordcloud\n\n",background='lightgray')
    
    st.markdown("---")     
    st.markdown("""As expected, the WordCloud displaying most frequent words related to neutral reviews (star=3) combined words with 
                both positive and negative connotations, which renders difficult to identify neutral specific words.""")
    st.markdown("""To further the analysis, we took advantage of tools provided by **Sklearn** and **Collections** not only to make easier the identification of 
                the most frequent words per sentiment, but also to aim at determining words that are specific to each sentiment. The 
                use of Part-of-speech (POS) tagging allowed us to refine our analysis to single out specifically nouns, adjectives, verbs and adverbs
                associated with a sentiment.""")
    st.markdown("""You can see below the results of this analysis.""")
    
    df1_fr=pd.read_csv('df_rawtext_V6neg_postagging.csv',index_col=0)
    df1_fr=df1_fr.fillna(' ') 
    df1_fr.head()
    
    def words_counter (col):
        """function that takes a column with texts and creates a dataframe listing unique tokens and their frequency"""
        cv = CountVectorizer() 
        vec = cv.fit_transform(col)
        word_freq = dict(zip(cv.get_feature_names(), np.asarray(vec.sum(axis=0)).ravel())) #flatten
        word_counter = collections.Counter(word_freq)
        word_counter_df = pd.DataFrame(word_counter.most_common(), columns = ['word', 'freq'])
        return (word_counter_df)
    
    def show_tree_map(df, nb_words, score_label, pos_tag=None):
        if score_label == "Negative" :
            score = 0 
        else:
            score = 1
        
        # Getting back columns corresponding to Tag selected
        if len(pos_tag) > 1:
            list_postag = [str(x).lower()+"_list_joined" for x in pos_tag]
            print(list_postag)
            df['target_column'] = df[list_postag].apply(' '.join, axis=1)
            word_counter_df = words_counter(col=df[df.rating == score]['target_column'])        
        elif len(pos_tag) == 1:
            column_name = str(pos_tag[0]).lower()+"_list_joined"
            print(column_name)
            df['target_column'] = df[column_name]
            word_counter_df = words_counter(col=df[df.rating == score]['target_column'])        
        else:
            df['target_column'] = df['No_stopwords_joined']
            word_counter_df = words_counter(col=df[df.rating == score]['target_column'])
            
        # Chart of common words
        fig1 = px.treemap(word_counter_df.iloc[:nb_words,], 
                          path=['word'], 
                          values='freq', 
                          title= '{nb} most commonly used words in {txt} reviews'.format(nb = str(nb_words), txt = score_label), 
                          color='freq',
                          color_continuous_scale='GnBu')
        st.plotly_chart(fig1)
    
        # Chart of unique words
        word_counter_df_opposite = words_counter(col = df[df.rating == abs(score-1)]['target_column'])
        common_no_stopwords = set(word_counter_df['word']).intersection(set(word_counter_df_opposite['word']))
        common_list = list(common_no_stopwords)
    
        unique_words = word_counter_df[word_counter_df['word'].isin(common_list) == False][:nb_words]
        fig2 = px.treemap(unique_words, 
                          path=['word'], 
                          values='freq', 
                          title='{nb} most Unique used words in {txt} reviews'.format(nb = str(nb_words), txt = score_label), 
                          color='freq',
                          color_continuous_scale='GnBu')
        st.plotly_chart(fig2)
    
    st.markdown("---") 
    st.markdown("***Visualization of words analysis for each category of comments***")
    output_selectbox = st.selectbox('Select which category of comment you want to display:', ['Positive', 'Negative'])
    output_slider = st.slider('Number of words to display:', 10, 100, 20)
    output_multiselect = st.multiselect('Select type of TAG to display: ', ['ADV', 'ADJ', 'NOUN', 'VERB'])
    show_tree_map(df=df1_fr, nb_words=output_slider, score_label=output_selectbox, pos_tag=output_multiselect)
    
###############    DATA PROCESSING
elif rad == "Data processing":  
    st.header('Data processing')
    st.subheader('Features selection')
    st.markdown("Since the text preprocessing is an essential step in building an efficient machine learning model, we tried to identify features that connotate a sentiment and are usually removed during the process but could improve the algorithm performance.  \n"
                "In addition to the number of words and sentences that correlated significantly with the ratings, we counted **punctuations** (exclamation and interrogation marks and ellipsis), **capslocks**, and **negative words** (ne,pas,ni,jamais,aucune,aucun,rien,sans,plus,n') and then try to identify any strong correlation between all the features and the rating.  \n"
                "Categorical features were dummy-encoded and numerical features were normalized using MinMaxScaler.  \n"
                "We kept only features displaying higher correlation with the target feature rating.")
    if st.button('Click if you want to see the features correlation heatmap'):
        img = Image.open('feats_heatmap.png')
        st.image(img, caption=None, width=None, use_column_width=None, clamp=False, channels="RGB", output_format="auto")
    st.markdown ("### Processed data")
    df_feats=pd.read_csv('feats_minmaxscaled.csv', index_col=0)
    df_feats.drop('No_stopwords_joined',axis=1, inplace=True)
    df_feats.rename(columns={'spacy_lemmatized_j':'Lemma'}, inplace=True)
    nb_rows=st.number_input("Choose the number of rows you want to display:", 0, 19000, 5)
    st.write(df_feats.head(nb_rows))



####################  MODELING
elif rad == "Modeling":
    st.header('Modeling')
    st.subheader('Data preparation')
    st.markdown("We used the lemmatized reviews, alone, or in combination with features that were highly correlated with the sentiment (word, sentence and negation counts, source).  \n")
    st.markdown ("Since machine learning algorithms cannot process text directly, the reviews need to be converted into numbers, or more precisely vectors of numbers. We used a popular and simple method of feature extraction with text data, called the **bag-of-words (BOW)**. It is a representation of text that describes the occurrence of words within a text. To reduce the dimensionality of the resulting matrix, it is usually preceded by a step of text preprocessing, as described in the previous part.  \n"
    "We used an advanced variant of the BOW that used the **term frequency–inverse document frequency (or Tf–Idf)**. Basically, the value of a word increases proportionally to count, but it is inversely proportional to the frequency of the word in the text. We used the text cleaned and lemmatized to feed the Tf-Idf."  \n
    "We also tried to improve the models performance by defining a pipeline that allowed us to select the best features and the best hyperparameters.")
    st.subheader('Machine Learning and Deep Learning models')
    st.markdown("We predicted the sentiment (positive or negative) of reviews with 4 classification  models,  namely :  \n"
                ">- **Gradient Boosting**  \n"
                ">- **CatBoosting**  \n"
                ">- **Logistic Regression**  \n"
                ">- **Support Vector Machine (SVM)**  \n")
    img = Image.open('modeling2.png')
    st.image(img, caption=None, width=None, use_column_width=None, clamp=False, channels="RGB", output_format="auto")
    st.write("As for the deep learning models we used a dense neural network including an embeddding layer, a MaxPooling layer and two dense layers. Only the reviews were used as inputs (Lemma).  \n"
                "We also incorporated a **fastText** pre-trained model to create an embedding matrix so as to initialize the embedding layer. [FastText] (https://fasttext.cc/) is an open-source library, developed by the Facebook AI Research lab and has proven to be very efficient on many NLP problems, such as semantic similarity detection and text classification.")   
    
   
    
###############     EVALUATION
elif rad == "Evaluation":    
   

    
    # def main():
    st.header('Predicting Sentiment Reviews')
    st.sidebar.header('Model Selection Panel')
    #st.subheader('Results Analysis')
    st.sidebar.markdown('Choose your model')
        #@st.cache(allow_output_mutation=True)
        #@st.cache(persist=True)
    #Prep data for ML models:
    df=pd.read_csv('feats_minmaxscaled.csv', index_col=0)
    to_keep=['spacy_lemmatized_j','rating','nb_words','negation','sentences_count',"company_ShowRoom","company_VeePee","source_TrustPilot","source_TrustedShop"]
    feats=df[to_keep]
    feats_train, feats_test, y_train, y_test = train_test_split(feats.drop(['rating'], axis=1), feats.rating, test_size=0.2, random_state=49)

    vectorizer = TfidfVectorizer( max_features=10000, ngram_range=(1,2))

    X_train_text = vectorizer.fit_transform(feats_train.spacy_lemmatized_j)
    X_test_text = vectorizer.transform(feats_test.spacy_lemmatized_j)

    X_train = hstack((X_train_text, feats_train.drop('spacy_lemmatized_j', axis=1).values))
    X_test = hstack((X_test_text, feats_test.drop('spacy_lemmatized_j', axis=1).values))

    class_names = ['Negative', 'Positive']
    

    # Definition of function resulting score computation of Algorithm mentionned in parameter
    def scores (model,X_train,y_train,X_test,y_test, y_pred):
    #Takes ML classifier, training and validation sets, and predictions and returns scores (accuracy, recall, f1, precision).
        score_train=model.score(X_train,y_train)                  # accuracy for training set
        score_test=model.score(X_test, y_test)                    # accuracy for validation set
        precision=precision_score(y_test, y_pred,average='macro') # precision for the validation set
        recall=recall_score(y_test, y_pred,average='macro')       # recall for the validation set
        f1=f1_score(y_test, y_pred,average='macro')               # f1 for the validation set
        return score_train, score_test, precision,recall,f1

    # Defintion of function resulting score computation of Algorithm mentionned in parameter   
    def Confusion_matrix(title,y_test,y_pred):
    #Takes 3 arguments (Title, target and predictions) and returns a visualization of the confusion matrix.
        cm = pd.crosstab(y_test, y_pred, rownames=['Actual'], colnames=['Predicted'])
        col1, col2, col3= st.columns([1,2, 1])
        with col2:
            fig=plt.figure(figsize=(4,3))
            sns.heatmap(cm, annot=True, fmt="d", cmap='Blues')
            plt.title(title)
            st.pyplot(fig)
        return cm    
        
    #st.sidebar.subheader('Select your Classifier')
    classifier = st.sidebar.selectbox('Classifier:', ('CatBoost', 'Logistic Regression','Logistic Regression - Pipeline' ,'Gradient Boosting', 'Gradient Boosting - Pipeline', 'Support Vector Machine', 'Support Vector Machine - Pipeline','Dense Neural Network', 'Dense Neural Network - fastText'))

        
    if classifier == 'Gradient Boosting':
        metrics = st.sidebar.multiselect('Select your metrics:', ('Confusion Matrix', 'Precision-Recall Curve', 'Classification report'))
        #st.sidebar.button('Classify', key='1')
        st.subheader('Gradient Boosting Results')
        GBC = GradientBoostingClassifier(n_estimators=100, learning_rate=1.0, max_depth=1, random_state=49)
        GBC.fit(X_train, y_train)
        #GBC = joblib.load("GBC_model.joblib")
        #accuracy = model.score(X_test, y_test)
        y_pred_GBC = GBC.predict(X_test)
        score_train_GBC, score_test_GBC,precision_GBC,recall_GBC,f1_GBC=scores(GBC,X_train,y_train,X_test,y_test, y_pred_GBC)
        st.write('Accuracy: ', score_test_GBC.round(2)*100,'%')
        st.write('Precision: ', precision_GBC.round(2))
        st.write('Recall: ', recall_GBC.round(2))
        
        if 'Confusion Matrix' in metrics:

            Confusion_matrix('Gradient Boosting confusion matrix',y_test, y_pred_GBC)
             

        if 'Precision-Recall Curve' in metrics:
            y_score = GBC.predict_proba(X_test)[:, 1]
            #calculate precision and recall
            precision, recall, thresholds = precision_recall_curve(y_test, y_score)

            #create precision recall curve
            col1, col2, col3= st.columns([1,2, 1])
            with col2:
                fig, ax = plt.subplots(figsize=(2, 2))
                ax.plot(recall, precision, color='purple')

            # add axis labels to plot
                ax.set_title('Precision-Recall Curve')
                ax.set_ylabel('Precision')
                ax.set_xlabel('Recall')
                st.pyplot(fig)
        else:
            
            class_rep_GBC=pd.DataFrame(classification_report(y_test, y_pred_GBC, output_dict=True))
            st.markdown("Classification report for GradientBoosting Classifier:") 
            st.write(class_rep_GBC.round(3))  
            
    if classifier == 'Gradient Boosting - Pipeline':
        metrics = st.sidebar.multiselect('Select your metrics:', ('Confusion Matrix', 'Precision-Recall Curve', 'Classification report'))
        #st.sidebar.button('Classify', key='1')
        st.subheader('Gradient Boosting - Pipeline Results')
        sel = SelectKBest( k=3000)
        sel.fit(X_train,y_train)
        #Training and predictions
        X_train_sel = sel.transform(X_train)
        X_test_sel = sel.transform(X_test)
        GBC_pipe = GradientBoostingClassifier(n_estimators=500, learning_rate=0.3, random_state=49,
                                              loss='exponential', max_features='auto')
        GBC_pipe.fit(X_train_sel, y_train)
        y_pred_GBC_pipe=GBC_pipe.predict(X_test_sel)
        score_train_GBC_pipe, score_test_GBC_pipe,precision_GBC_pipe,recall_GBC_pipe,f1_GBC_pipe=scores(GBC_pipe,X_train_sel,y_train,X_test_sel,y_test, y_pred_GBC_pipe)
        st.write('Accuracy: ', score_test_GBC_pipe.round(2)*100,'%')
        st.write('Precision: ', precision_GBC_pipe.round(2))
        st.write('Recall: ', recall_GBC_pipe.round(2))
        
        if 'Confusion Matrix' in metrics:

            Confusion_matrix('Gradient Boosting - Pipeline confusion matrix',y_test, y_pred_GBC_pipe)
             

        if 'Precision-Recall Curve' in metrics:
            y_score = GBC_pipe.predict_proba(X_test_sel)[:, 1]
            #calculate precision and recall
            precision, recall, thresholds = precision_recall_curve(y_test, y_score)

            #create precision recall curve
            col1, col2, col3= st.columns([1,2, 1])
            with col2:
                fig, ax = plt.subplots(figsize=(2, 2))
                ax.plot(recall, precision, color='purple')

            # add axis labels to plot
                #ax.set_title('Precision-Recall Curve')
                st.markdown("Classification report for GradientBoosting Classifier:")
                ax.set_ylabel('Precision')
                ax.set_xlabel('Recall')
                st.pyplot(fig)
        else:
                
                class_rep_GBC_pipe=pd.DataFrame(classification_report(y_test, y_pred_GBC_pipe, output_dict=True))
                st.markdown("Classification report for GradientBoostingClassifier - Pipeline:") 
                st.write(class_rep_GBC_pipe.round(3))
    
    elif classifier == 'Logistic Regression':
    
        metrics = st.sidebar.multiselect('Select your metrics:', ('Confusion Matrix', 'Precision-Recall Curve', 'Classification report'))
        #st.sidebar.button('Classify', key='1')
        st.subheader('Logistic Regression Results')
        LR = joblib.load("LR_model.joblib")
        #LR = LogisticRegression()
        #LR.fit(X_train, y_train)
        y_pred_LR=LR.predict(X_test)

        score_train_LR, score_test_LR,precision_LR,recall_LR,f1_LR=scores(LR,X_train,y_train,X_test,y_test, y_pred_LR)

        #accuracy = model.score(X_test, y_test)
        st.write('Accuracy: ', score_test_LR.round(2)*100,'%')
        st.write('Precision: ', precision_LR.round(2))
        st.write('Recall: ', recall_LR.round(2))
        
        if 'Confusion Matrix' in metrics:

            Confusion_matrix ("Logistic Regression Classifier Confusion Matrix ",y_test, y_pred_LR)
             

        if 'Precision-Recall Curve' in metrics:
            y_score = LR.predict_proba(X_test)[:, 1]
            #calculate precision and recall
            precision, recall, thresholds = precision_recall_curve(y_test, y_score)

            #create precision recall curve
            col1, col2, col3= st.columns([1,2, 1])
            with col2:
                fig, ax = plt.subplots(figsize=(2, 2))
                ax.plot(recall, precision, color='purple')

            # add axis labels to plot
                ax.set_title('Precision-Recall Curve')
                ax.set_ylabel('Precision')
                ax.set_xlabel('Recall')
                st.pyplot(fig)
        else:
            
            class_rep_LR=pd.DataFrame(classification_report(y_test, y_pred_LR, output_dict=True)).round(3)
            st.markdown("Classification report for Logistic Regression Classifier:") 
            st.write(class_rep_LR)  

    elif classifier == 'Logistic Regression - Pipeline':
    
        metrics = st.sidebar.multiselect('Select your metrics:', ('Confusion Matrix', 'Precision-Recall Curve', 'Classification report'))
        #st.sidebar.button('Classify', key='1')
        st.subheader('Logistic Regression - Pipeline Results')
        LR_pipe = joblib.load("LR_pipe_model.joblib")
        #LR_pipe = LogisticRegression(class_weight='balanced', penalty='l2', solver='lbfgs',C=1)
        #LR_pipe.fit(X_train, y_train)
        y_pred_LR_pipe=LR_pipe.predict(X_test)

        score_train_LR_pipe, score_test_LR_pipe,precision_LR_pipe,recall_LR_pipe,f1_LR_pipe=scores(LR_pipe,X_train,y_train,X_test,y_test, y_pred_LR_pipe)
        #accuracy = model.score(X_test, y_test)
        st.write('Accuracy: ', score_test_LR_pipe.round(2)*100,'%')
        st.write('Precision: ', precision_LR_pipe.round(2))
        st.write('Recall: ', recall_LR_pipe.round(2))
        
        if 'Confusion Matrix' in metrics:

            Confusion_matrix ("Logistic Regression - Pipeline Classifier Confusion Matrix ",y_test, y_pred_LR_pipe)
             

        if 'Precision-Recall Curve' in metrics:
            y_score = LR_pipe.predict_proba(X_test)[:, 1]
            #calculate precision and recall
            precision, recall, thresholds = precision_recall_curve(y_test, y_score)

            #create precision recall curve
            col1, col2, col3= st.columns([1,2, 1])
            with col2:
                fig, ax = plt.subplots(figsize=(2, 2))
                ax.plot(recall, precision, color='purple')

            # add axis labels to plot
                ax.set_title('Precision-Recall Curve')
                ax.set_ylabel('Precision')
                ax.set_xlabel('Recall')
                st.pyplot(fig)
        else:
            
            class_rep_LR_pipe=pd.DataFrame(classification_report(y_test, y_pred_LR_pipe, output_dict=True)).round(3)
            st.markdown("Classification report for Logistic Regression - Pipeline Classifier:") 
            st.write(class_rep_LR_pipe)  

    elif classifier == 'CatBoost':
        metrics = st.sidebar.multiselect('Select your metrics:', ('Confusion Matrix', 'Precision-Recall Curve', 'Classification report'))
        #st.sidebar.button('Classify', key='1')
        st.subheader('CatBoost Results')
        #CBC =CatBoostClassifier(iterations=100, random_seed=42, )
        #Training and predictions
        #CBC.fit(X_train, y_train)
        CBC = joblib.load("CBC_model.joblib")

        y_pred_CBC=CBC.predict(X_test)
        score_train_CBC, score_test_CBC,precision_CBC,recall_CBC,f1_CBC=scores(CBC,X_train,y_train,X_test,y_test, y_pred_CBC)
        st.write('Accuracy: ', score_test_CBC.round(2)*100,'%')
        st.write('Precision: ', precision_CBC.round(2))
        st.write('Recall: ', recall_CBC.round(2))
        
        if 'Confusion Matrix' in metrics:

            Confusion_matrix ("CatBoost Classifier Confusion Matrix ",y_test, y_pred_CBC)
             

        if 'Precision-Recall Curve' in metrics:
            y_score = CBC.predict_proba(X_test)[:, 1]
            #calculate precision and recall
            precision, recall, thresholds = precision_recall_curve(y_test, y_score)

            #create precision recall curve
            col1, col2, col3= st.columns([1,2, 1])
            with col2:
                fig, ax = plt.subplots(figsize=(2, 2))
                ax.plot(recall, precision, color='purple')

            # add axis labels to plot
                ax.set_title('Precision-Recall Curve')
                ax.set_ylabel('Precision')
                ax.set_xlabel('Recall')
                st.pyplot(fig)
        else:
            
            class_rep_CBC=pd.DataFrame(classification_report(y_test, y_pred_CBC, output_dict=True)).round(3)
            st.markdown("Classification report for  CatBoost Classifier:") 
            st.write(class_rep_CBC)
            
            
    elif classifier == 'Support Vector Machine':
    
        metrics = st.sidebar.multiselect('Select your metrics:', ('Confusion Matrix', 'Precision-Recall Curve', 'Classification report'))
        #st.sidebar.button('Classify', key='1')
        st.subheader('Support Vector Machine Results')
        SVC = joblib.load("SVC_model.joblib")
        #SVC=SVC()
        #SVC.fit(X_train, y_train)
        y_pred_SVC=SVC.predict(X_test)
        score_train_SVC, score_test_SVC,precision_SVC,recall_SVC,f1_SVC=scores(SVC,X_train,y_train,X_test,y_test, y_pred_SVC)

        #accuracy = model.score(X_test, y_test)
        st.write('Accuracy: ', score_test_SVC.round(2)*100,'%')
        st.write('Precision: ', precision_SVC.round(2))
        st.write('Recall: ', recall_SVC.round(2))
        
        if 'Confusion Matrix' in metrics:

            Confusion_matrix ("Support Vector Machine Classifier Confusion Matrix  ",y_test, y_pred_SVC)             

        if 'Precision-Recall Curve' in metrics:
            y_score = SVC.predict_proba(X_test)[:, 1]
            #calculate precision and recall
            precision, recall, thresholds = precision_recall_curve(y_test, y_score)

            #create precision recall curve
            col1, col2, col3= st.columns([1,2, 1])
            with col2:
                fig, ax = plt.subplots(figsize=(2, 2))
                ax.plot(recall, precision, color='purple')

            # add axis labels to plot
                ax.set_title('Precision-Recall Curve')
                ax.set_ylabel('Precision')
                ax.set_xlabel('Recall')
                st.pyplot(fig)
        else:
            
            class_rep_SVC=pd.DataFrame(classification_report(y_test, y_pred_SVC, output_dict=True)).round(3)
            st.markdown("Classification report for Support Vector Classifier:") 
            st.write(class_rep_SVC)

    elif classifier == 'Support Vector Machine - Pipeline':
    
        metrics = st.sidebar.multiselect('Select your metrics:', ('Confusion Matrix', 'Precision-Recall Curve', 'Classification report'))
        #st.sidebar.button('Classify', key='1')
        st.subheader('Support Vector Machine -Pipeline Results')
        #SVC_pipe = joblib.load("C:/Users/celin/Documents/cours/formation_DatascienTest_2022_bootcamp/projet_satisfaction_client/SVC_pipe_model.joblib")
        SVC_pipe = joblib.load("SVC_pipe_model.joblib")
        y_pred_SVC_pipe=SVC_pipe.predict(X_test)
        score_train_SVC_pipe, score_test_SVC_pipe, precision_SVC_pipe, recall_SVC_pipe, f1_SVC_pipe=scores(SVC_pipe,X_train,y_train,X_test,y_test, y_pred_SVC_pipe)
        #accuracy = model.score(X_test, y_test)
        st.write('Accuracy: ', score_test_SVC_pipe.round(2)*100,'%')
        st.write('Precision: ', precision_SVC_pipe.round(2))
        st.write('Recall: ', recall_SVC_pipe.round(2))
        
        if 'Confusion Matrix' in metrics:

            Confusion_matrix ("Support Vector Classifier - Pipeline Confusion Matrix  ",y_test, y_pred_SVC_pipe)             

        if 'Precision-Recall Curve' in metrics:
            y_score = SVC_pipe.predict_proba(X_test)[:, 1]
            #calculate precision and recall
            precision, recall, thresholds = precision_recall_curve(y_test, y_score)

            #create precision recall curve
            col1, col2, col3= st.columns([1,2, 1])
            with col2:
                fig, ax = plt.subplots(figsize=(2, 2))
                ax.plot(recall, precision, color='purple')

            # add axis labels to plot
                ax.set_title('Precision-Recall Curve')
                ax.set_ylabel('Precision')
                ax.set_xlabel('Recall')
                st.pyplot(fig)
        else:
            
            class_rep_SVC_pipe=pd.DataFrame(classification_report(y_test, y_pred_SVC_pipe, output_dict=True))
            st.markdown("Classification report for Support Vector Classifier - Pipeline:") 
            st.write(class_rep_SVC_pipe.round(3))    
    
    
    
    elif classifier == 'Dense Neural Network':
        metrics = st.sidebar.multiselect('Select your metrics:', ('Confusion Matrix', 'Loss/Precision/Accuracy Curves', 'Classification report'))
        #st.sidebar.button('Classify', key='1')
        st.subheader('Dense Neural Network Results')
        st.write('Accuracy: ', 0.89*100,'%')
        st.write('Precision: ', 91.81)
        st.write('Recall: ', 87.49)
        
        if 'Confusion Matrix' in metrics:
        #confusion matrix
            st.markdown("Confusion matrix for Dense Neural Network:") 
            img = Image.open('DNN_cm.png')
            st.image(img, caption=None, width=None, use_column_width=None, clamp=False, channels="RGB", output_format="auto")
        
        if 'Loss/Precision/Accuracy Curves' in metrics:
            img = Image.open('DNN_history.png')
            st.image(img, caption=None, width=None, use_column_width=None, clamp=False, channels="RGB", output_format="auto")
        else:
            #classification report
            class_rep_DNN = pd.read_csv('class_rep_DNN.csv', index_col=0)
            st.markdown("Classification report for Dense Neural Network:") 
            st.write (class_rep_DNN)


    elif classifier == 'Dense Neural Network - fastText':
        metrics = st.sidebar.multiselect('Select your metrics:', ('Confusion Matrix', 'Loss/Precision/Accuracy Curves', 'Classification report'))
        #st.sidebar.button('Classify', key='1')
        st.subheader('Dense Neural Network Results')
        st.write('Accuracy: ', 0.89*100,'%')
        st.write('Precision: ', 0.92)
        st.write('Recall: 0.88', 0.88)
        
        if 'Confusion Matrix' in metrics:
        #confusion matrix
            st.markdown("Confusion matrix for Dense Neural Network - fastText:") 
            img = Image.open('DNN_ft_cm.png')
            st.image(img, caption=None, width=None, use_column_width=None, clamp=False, channels="RGB", output_format="auto")
       
        if 'Loss/Precision/Accuracy Curves' in metrics:
            img = Image.open('DNN_ft_history.png')
            st.image(img, caption=None, width=None, use_column_width=None, clamp=False, channels="RGB", output_format="auto")
        else:
            #classification report
            class_rep_DNN_ft = pd.read_csv('class_rep_DNN_lem_ft.csv', index_col=0)
            st.markdown("Classification report for Dense Neural Network - fastText:") 
            st.write (class_rep_DNN_ft)

  
    
    
###############     CONCLUSION  
elif rad == "Conclusion & Perspectives":
    st.header('Conclusion')
    if st.button('Click if you want to see the scoring metrics for all models tested'):

        #Display dataframe of all scores
        df_all_scores=pd.read_csv('df_all_scores.csv', index_col=0)
        st.markdown("Performance results for all models")
        st.write(df_all_scores)
    #Visualization (barplot) of the scores obtained for the different models:
        df_graph=df_all_scores.reset_index().melt('index',var_name='cols',value_name='vals')

        fig, ax = plt.subplots(figsize=(12,5))
        sns.barplot(x = 'vals', y = 'cols' , data = df_graph, hue = 'index', palette='Blues')
        ax.set(xlabel='Score Value',ylabel='Classifier')
        ax.set_xlim(0.7, 0.95)
        plt.legend(bbox_to_anchor=(1,1))
        st.pyplot(fig)
    st.markdown("In  this  project,  we  compared 4 supervised machine learning approaches :  Gradient Boosting, CatBoosting, SVM and Logistic Regression and 2 deep learning models. Reviews were preprocesssed and prepared using  various  NLP  techniques  including  stopwords  removal,  word lemmatization,  TF-IDF  vectorization and word embedding.  \n"
                "Our  experimental  approaches  studied  the accuracy,  precision,  recall,  and  F1  score , focusing on the precision metrics so as to minimize the false positives."
                "Overall,  all  our  models  were  able  to  classify  negative  and  positive  reviews  with good  accuracy  and  precision (minimum of 86%) with SVC outperforming the other classifiers, including the Dense Neural Networks models with scoring metrics reaching 90% (accuracy, precision recall and f1-score). However, we were able to reach a precision of 93% using a fastText classifier (preliminary data). Fasttext using shallow neural network, we might be able to improve our deep learning models performance by opting for a simpler architecture.  \n"
                "There are few other options that we have not tried, since lots of NLP tools are dedicated to the English language. Among them, CamemBERT, whish is state-of-the-art language model for French based on RoBERTa architecture.  \n"
                "All supervised machine learning algorithms  performed  better  in  term  of classifying positive sentiment, with systematically lower precision and F1-scores for the negative class. This might be due to the reduced proportion  of  negative reviews  or the fact we included the 3-star reviews in the negative class, which could slighltly skew the distinction between negative and positive sentiment.  \n"
                "Future work would focus on optimizing our models for a multiclass classification problem so as to predict more accurately the star rating (4 classes by removing the 3-star/'neutral' rating class).")    
    st.header('Perspectives')
    st.markdown("If we try to project further results of our work and put it into more business perspective, a valuable insight we could obtain, it would be to understand what are the reasons of satisfaction or dissatisfaction from customers. This could help the company to monitor its activity completed with additionnal key performance indicators (KPI), tracking evolution per region of good/bad comments for each of these main categories identified.  \n"
                "We could also complete this analysis by deep diving into specific comments where interaction has been initiated with the customer. (Is the answer appropriate? Is there any continuation ? How is the situation resolved personally? etc...).  \n"
                "As a consequence this would help the company to engage concrete action plan internally that would improve its operational efficiency, and in-fine its digital reputation."
                "In this report we already tried to serve this purpose in different ways, notably:  \n"
                ">- our work with the reviews POStagging showed that we can retrieve easily features that would allow us not only to identify more precisely customers'pain points, but also to study their seasonal and yearly trends/changes.  \n"
                ">- In a different approach, we also initated a mapping defining all main categories and sub-categories using Graph Theory and NetworkX library. This first exercice showed us that the analysis of the community and relationship between each other could be also an interesting insight.(Is there issue common to other categories? To re-phrase it from the opposite angle: 'If we solve this issue, this will tackle two problems in one').  \n"
                ">- Finally we focused on the reviews written in French as they represented more than 89% of our dataset, it would be interesting to collect more reviews that are in the top 5 languages.  \n"
                "In a nutshell, this project is just the beginning of a bigger AI application and, with further development effort, could spark off the interest of retail companies.")
    
    df_netx = pd.read_csv('df_relationship_final_postag.csv',index_col=0)
    df_netx.rename(columns={'word': 'sub_category'}, inplace=True)
    
    # Function ensuring vizualisation of Relationship
    def relationship_chart(source):
        G = nx.from_pandas_edgelist(source, source = "main_category", target="sub_category", edge_attr="value", create_using = nx.Graph())
        communities = best_partition(G)
        nx.set_node_attributes(G, communities, 'group')
        com_net = Network(notebook=True, width="700px", height="700px", bgcolor="#222222", font_color='white')
        com_net.from_nx(G)
        com_net.show("mapping.html")
        
        HtmlFile = open("mapping.html", 'r', encoding='utf-8')
        source_code = HtmlFile.read() 
        components.html(source_code, height = 900,width=900)
        
    st.markdown("---")    
    check =  st.checkbox('Example of visualization of main & sub categories using NetworkX and POS tagging')
    if check:
        radio_options=['Verbs', 'Nouns', 'Adverbs']
        radio_status = st.radio("Select which POS tagging to see visualization of categories of comments:", radio_options)
    
        if (radio_status == 'Verbs'):
            df_nx = df_netx[df_netx.tag == 'VERB']
            df_nx = df_nx.head(150)
            relationship_chart(df_nx)
        
        elif (radio_status == 'Nouns'):
            df_nx = df_netx[df_netx.tag == 'NOUN']
            df_nx = df_nx.head(150)
            relationship_chart(df_nx)
        
        elif (radio_status == 'Adverbs'):
            df_nx = df_netx[df_netx.tag == 'ADV']
            df_nx = df_nx.head(150)
            relationship_chart(df_nx)
    

