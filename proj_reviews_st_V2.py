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

import scikitplot as skplt
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import precision_recall_curve
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import classification_report
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import classification_report
from catboost import CatBoostClassifier
from sklearn.naive_bayes import BernoulliNB,MultinomialNB
from sklearn.svm import SVC 


#Creation of a dataframe with with the data from the file "reviews_trust.csv":
df=pd.read_csv('reviews_trust.csv', index_col=0)

#Setting option to show max rows and max columns
pd.set_option("display.max_columns",None)
pd.set_option("display.max_rows", None)


#Sidebar creation: 
rad = st.sidebar.radio("Menu",["Project presentation", "Explorative Data Analysis", "Data processing", "Modeling", "Conclusion & Perspectives"])

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
    st.markdown('### 1. Data description and preparation')
    st.markdown('> #### 1.1. Dataset discovery')


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
    st.markdown('> #### 1.2. Cleaning of Duplicates & NaN')
    
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
    st.markdown("### 2. Data Visualization")
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
    
###############EDA
elif rad == "Modeling":    

    
    #from nltk.corpus import stopwords
    from sklearn.model_selection import train_test_split
    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import classification_report
    from sklearn.metrics import f1_score
    from sklearn.metrics import precision_score
    from sklearn.metrics import recall_score
    from sklearn.metrics import classification_report
    from sklearn.metrics import confusion_matrix
    from sklearn.metrics import plot_confusion_matrix, plot_roc_curve
    from catboost import CatBoostClassifier
    from sklearn.naive_bayes import BernoulliNB,MultinomialNB
    from sklearn.svm import SVC
    from scipy.sparse import hstack
    from sklearn.metrics import precision_score, recall_score
    import scikitplot as skplt
    
    # def main():
    st.title('Predicting Sentiment Reviews')
    st.sidebar.title('Model Selection Panel')
    st.subheader('Results Analysis')
    st.sidebar.markdown('Choose your model')
        #@st.cache(allow_output_mutation=True)
        #@st.cache(persist=True)
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
        
        
    st.sidebar.subheader('Select your Classifier')
    classifier = st.sidebar.selectbox('Classifier', ('Catboosting', 'Logistic Regression', 'Gradient Boosting', 'Gradient Boosting - Pipeline', 'Support Vector Machine', 'Support Vector Machine - Pipeline'))

        
    if classifier == 'Gradient Boosting':
        metrics = st.sidebar.multiselect('Select your metrics?', ('Confusion Matrix', 'Precision-Recall Curve'))
        st.sidebar.button('Classify', key='1')
        st.subheader('Gradient Boosting Results')
        model = GradientBoostingClassifier(n_estimators=100, learning_rate=1.0, max_depth=1, random_state=49)
        model.fit(X_train, y_train)
        accuracy = model.score(X_test, y_test)
        y_pred = model.predict(X_test)
        st.write('Accuracy: ', accuracy.round(2)*100,'%')
        st.write('Precision: ', precision_score(y_test, y_pred, labels=class_names).round(2))
        st.write('Recall: ', recall_score(y_test, y_pred, labels=class_names).round(2))
        
        if 'Confusion Matrix' in metrics:

            confusion_matrix = confusion_matrix(y_test, y_pred)

            fig, ax = plt.subplots(figsize=(3, 3))
            ax.matshow(confusion_matrix, cmap=plt.cm.Blues, alpha=0.3)
            for i in range(confusion_matrix.shape[0]):
                for j in range(confusion_matrix.shape[1]):
                    ax.text(x=j, y=i,s=confusion_matrix[i, j], va='center', ha='center', size='xx-large')
            
            plt.xlabel('Predictions', fontsize=10)
            plt.ylabel('Actuals', fontsize=10)
            plt.title('Confusion Matrix', fontsize=10)
            st.pyplot(fig)

        else:
            y_score = model.predict_proba(X_test)[:, 1]
            #calculate precision and recall
            precision, recall, thresholds = precision_recall_curve(y_test, y_score)

            #create precision recall curve
            fig, ax = plt.subplots(figsize=(3, 3))
            ax.plot(recall, precision, color='purple')

            # add axis labels to plot
            ax.set_title('Precision-Recall Curve')
            ax.set_ylabel('Precision')
            ax.set_xlabel('Recall')
            st.pyplot(fig)

    elif classifier == 'Logistic Regression':
        st.sidebar.subheader('Model Parameters')
        metrics = st.sidebar.multiselect('Select your metrics?', ('Confusion Matrix', 'Precision-Recall Curve'))
        st.sidebar.button('Classify', key='2')
        st.subheader('Logistic Regression Results')
        model = LogisticRegression(class_weight='balanced', penalty='l2', solver='lbfgs',C=1)
        model.fit(X_train, y_train)
        accuracy = model.score(X_test, y_test)
        y_pred = model.predict(X_test)
        st.write('Accuracy: ', accuracy.round(2)*100,'%')
        st.write('Precision: ', precision_score(y_test, y_pred, labels=class_names).round(2))
        st.write('Recall: ', recall_score(y_test, y_pred, labels=class_names).round(2))

        if 'Confusion Matrix' in metrics:

            confusion_matrix = confusion_matrix(y_test, y_pred)

            fig, ax = plt.subplots(figsize=(3, 3))
            ax.matshow(confusion_matrix, cmap=plt.cm.Blues, alpha=0.3)
            for i in range(confusion_matrix.shape[0]):
                for j in range(confusion_matrix.shape[1]):
                    ax.text(x=j, y=i,s=confusion_matrix[i, j], va='center', ha='center', size='xx-large')
            
            plt.xlabel('Predictions', fontsize=10)
            plt.ylabel('Actuals', fontsize=10)
            plt.title('Confusion Matrix', fontsize=10)
            st.pyplot(fig)

        else:
            y_score = model.predict_proba(X_test)[:, 1]
            #calculate precision and recall
            precision, recall, thresholds = precision_recall_curve(y_test, y_score)

            #create precision recall curve
            fig, ax = plt.subplots(figsize=(3, 3))
            ax.plot(recall, precision, color='purple')

            # add axis labels to plot
            ax.set_title('Precision-Recall Curve')
            ax.set_ylabel('Precision')
            ax.set_xlabel('Recall')
            st.pyplot(fig)

            # fig = plt.subplots(figsize=(3, 3))
            # skplt.metrics.plot_roc_curve(y_test, y_pred[:,:2])
            # st.pyplot(fig)




    
    
    
    
###############CONCLUSION  
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
    st.markdown("In  this  project,  we  compared 4 supervised machine learning approaches :  Gradient Boosting, CatBooting, SVM and Logistic Regression and 2 deep learning models. Reviews were preprocesssed and prepared using  various  NLP  techniques  including  stopwords  removal,  word lemmatization,  TF-IDF  vectorization and word embedding.  \n"
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
    
    
    

