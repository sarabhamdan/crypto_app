import streamlit as st
from requests.exceptions import ConnectionError, Timeout, TooManyRedirects
# import json
import orjson
import pandas as pd
import numpy as np
import datetime
from datetime import date, timedelta
import re
from requests import Request, Session
import requests
from bs4 import BeautifulSoup
import datetime
from dateutil.relativedelta import relativedelta
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
import time
from pycoingecko import CoinGeckoAPI
import emoji
import gate_api
from gate_api.exceptions import ApiException, GateApiException
from st_aggrid import AgGrid
from st_aggrid.grid_options_builder import GridOptionsBuilder
from st_aggrid import JsCode
from st_aggrid.shared import GridUpdateMode
import plotly.express as px
import hydralit_components as hc
from IPython.core.display import HTML
import yfinance as yf
import plotly.graph_objects as go
from autots import AutoTS

st.set_page_config(layout="wide")
#app logo
st.image("https://i.imgur.com/KylBVCy.png", use_column_width=True)
#set navigation menu
menu_data= [
    {'icon':"fas fa-home",'label': "Home"},
    {'icon':"fas fa-layer-group",'label': "Meta Data"},
    {'icon':"far fa-chart-bar",'label': "Forecast Analysis"},
    {'icon':"fas fa-stream",'label': "Real Time Forecasts"},
    {'icon':"fas fa-hand-holding-usd",'label': "Wallet Tracker"},
    {'icon':"fas fa-laptop", 'label': 'Machine Learning'},
    {'icon':"fas fa-code", 'label': 'Methodology'}]
menu_id= hc.nav_bar(menu_definition= menu_data, first_select=0, sticky_nav=True, hide_streamlit_markers=False,
        override_theme= {'menu_background':'gray', 'txc_active':'red'})#sticky_mode='sticky'
#edit footer
page_style= """
    <style>
    footer{
        visibility: visible;
        }
    footer:after{
        content: 'Developed by Sara Bou Hamdan';
        display:block;
        position:relative;
        color:red;
    }
    </style>"""
st.markdown(page_style, unsafe_allow_html=True)

@st.cache
def load_meta_data():
    df= pd.read_csv("meta_data.csv")
    df= df.fillna('N/A')
    df= df.rename(columns={'platform':'blockchain'})
    #clean and tranform data
    df['tag-names']= df['tag-names'].apply(lambda x: x.replace("]",""))
    df['tag-names']= df['tag-names'].apply(lambda x: x.replace("[",""))
    df['date_added']= df.date_added.apply(lambda x: datetime.datetime.strptime(x,"%Y-%m-%d")).dt.date
    df.drop("announcement", axis=1, inplace= True)
    # df['date_added']= df.date_added.apply(lambda x: datetime.datetime.strptime(x,"%d-%m-%y")).dt.date
    df.columns = df.columns.str.title()
    return df    

@st.cache(allow_output_mutation=True)
def load_data():
    df = pd.read_csv('all_forecasts.csv') 
    # transform forecast data
    df.drop(['tags','last_updated.1'],axis=1, inplace=True)
    df= df.rename(columns={'market_cap': "verified_market_cap", 'circulating_supply': 'verified_circulating_supply'})
    df['platform']= df['platform'].fillna('N/A')
    # convert date column to datetime format
    df['date_added']= df.date_added.apply(lambda x: datetime.datetime.strptime(x,"%Y-%m-%d")).dt.date
    #create market cap & supply columns that use self-reported figures when verified data is unavailable
    df['market_cap']= df.apply(lambda row: row['verified_market_cap'] if row['verified_market_cap'] != 0 else row['self_reported_market_cap'], axis=1)
    df['circulating_supply']= df.apply(lambda row: row['verified_circulating_supply'] if row['verified_circulating_supply'] != 0 else row['self_reported_circulating_supply'], axis=1)
    #create a new field to indicate whether data used is verified or self-reported
    types= []
    for i in range(0, len(df)):
        if df.verified_market_cap[i] != 0:
            types.append("Verified")
        elif (df.self_reported_market_cap[i] != 0) & (np.isnan(df.self_reported_market_cap[i]) == False):
            types.append("Self-Reported")
        else:
            types.append("N/A")
    df['data_source']= types
    df['market_cap'].fillna(-1.00, inplace=True) #differentiate cases where market cap is unavailable
    # calculate forecasts prices of Wallet Investor, Gov Capital, & CryptoPredictions.com from scraped % change
    change_cols=[col for col in df.columns if ("CP" in col) or ("WI" in col) or ("GC" in col)]
    for col in change_cols:
        df[col]= (df[col]/100 +1) * df.price
    return df

meta_data= load_meta_data()
forecasts= load_data()

#create dict columns that contain the scraped forecasts by period
cols_3m=[col for col in forecasts.columns if "3m" in col]
cols_1y=[col for col in forecasts.columns if "1y" in col]
cols_5y=[col for col in forecasts.columns if ("5y" in col) or ("3y" in col) or ("4y" in col)]
cols_summary={'short-term':cols_3m,'medium-term':cols_1y,'long-term':cols_5y}

@st.cache
def forecast_analysis(data,period = 'long-term'):
    #calculate min, max, mean, median prices
    data['Min_Price']= data[cols_summary[period]].min(axis=1)
    data['Max_Price']= data[cols_summary[period]].max(axis=1)
    data['Avg_Price']= data[cols_summary[period]].mean(axis=1)
    data['Median_Price']= data[cols_summary[period]].median(axis=1)    
    #calculate % changes with respect to price
    data['Min_Change']= (data['Min_Price'] / data['price'] - 1)
    data['Max_Change']= (data['Max_Price'] / data['price'] - 1)  
    data['Avg_Change']= (data['Avg_Price'] / data['price'] - 1)
    data['Median_Change']= (data['Median_Price'] / data['price'] - 1)
    #calculate top 5% and 10% thresholds
    min_90, max_90, avg_90, median_90 = [x for x in data[['Min_Change', 'Max_Change', 'Avg_Change','Median_Change']].quantile(0.9, axis=0)]
    min_95, max_95, avg_95, median_95 = [x for x in data[['Min_Change', 'Max_Change', 'Avg_Change','Median_Change']].quantile(0.95, axis=0)]
    #generate trading recommendation
    data['Action']= data.apply(lambda row: "Strong Buy" if ((row['Min_Change']>= min_95) 
                                         and (row['Max_Change']>= max_95) and (row['Avg_Change']>= avg_95)
                                         and (row['Median_Change']>= median_95) and (row['market_cap'] > 0)) else ("Buy" if ((row['Min_Change']>= min_90) 
                                         and (row['Max_Change']>= max_90) and (row['Avg_Change']>= avg_90)
                                         and (row['Median_Change']>= median_90)and (row['market_cap'] > 0)) else ("Strong Buy - Swappable" if ((row['Min_Change']>= min_95) 
                                         and (row['Max_Change']>= max_95) and (row['Avg_Change']>= avg_95) and (row['Median_Change']>= median_95)and (row['market_cap'] == -1.00))
                                          else ("Buy - Swappable" if ((row['Min_Change']>= min_90) and (row['Max_Change']>= max_90) and (row['Avg_Change']>= avg_90) 
                                          and (row['Median_Change']>= median_90)and (row['market_cap'] == -1.00)) else "Avoid"))), axis=1)

@st.cache
def get_tags(df):
    tags_options=[]
    for i in range(0, len(df)):
        coin_tags=[sub.replace("'", "").strip() for sub in df['Tag-Names'][i].split(",")]
        tags_options= tags_options+coin_tags
    unique_tags= sorted(list(set(tags_options)))
    return unique_tags
#get the unique categories of cryptos provided by CoinMarketCap
unique_tags= get_tags(meta_data)

@st.cache
def fit_autots(data, forecast_length):
    model = AutoTS(forecast_length=forecast_length, frequency='infer', ensemble='simple', drop_data_older_than_periods=200,num_validations=2, verbose=0)
    model = model.fit(data, date_col='Date', value_col='Close', id_col=None)
    return model

if menu_id == "Machine Learning":
    st.subheader("Crypto Price Prediction using AutoTS Model")
    coins, date_s, length= st.columns(3)
    name= coins.selectbox("Select the crypto of interest:", options= sorted(meta_data.Name))
    start_date= date_s.date_input("Start date of historical data for prediction model:", value= datetime.date(2020,1,1),
                min_value= datetime.date(2016,1,1),max_value= datetime.datetime.today())
    forecast_length= length.selectbox("Forecast length period in days:", options=[10,30])
    m = st.markdown("""
        <style>
        div.stButton > button:first-child {
            background-color: gray;
            color:#ffffff;
        }
        div.stButton > button:hover {
            background-color: red;
            color:#ffffff;
            }
        </style>""", unsafe_allow_html=True)
    run= st.button("Run")

    if run:
        symbol= meta_data[meta_data.Name == name]['Symbol'].tolist()[0]
        ticker= symbol.upper()+"-USD"
        #set dates to get historical price data
        today = datetime.datetime.today().strftime('%Y-%m-%d')
        start_date = (start_date+ relativedelta(days=+1)).strftime('%Y-%m-%d')
        #get data from yfinance on crypto prices
        try:
            data = yf.download(ticker,start_date, today)
            data.reset_index(inplace=True)
            hist, div,future= st.columns([1,0.1,1])
            with hist:
                #visualize historical price movement
                # plot the open price
                x = data["Date"]
                y = data["Close"]
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=x, y=y,marker = {'color' : 'gray'}))
                # Set title
                fig.update_layout(
                    title_text="Time Series Plot of "+symbol.upper()+" Historical Close Prices",
                    yaxis={'showgrid': False},xaxis={'showgrid': False}, width=600, height=400
                )
                fig.update_layout(
                    xaxis=dict(
                        rangeselector=dict(
                            buttons=list(
                                [
                                    dict(count=1, label="1m", step="month", stepmode="backward"),
                                    dict(count=6, label="6m", step="month", stepmode="backward"),
                                    dict(count=1, label="YTD", step="year", stepmode="todate"),
                                    dict(count=1, label="1y", step="year", stepmode="backward"),
                                    dict(step="all"),
                                ]
                            )
                        ),
                        rangeslider=dict(visible=False),
                        type="date",
                    )
                )
                st.write(fig)
            with future:
                #train AutoTS model to predict future price movement
                #define & fit model:
                with st.spinner("Finding the best prediction model for you using AutoTS... For accurate results, run can take up to 5 mins!"):
                    model = fit_autots(data, forecast_length=forecast_length)
                #get prediction
                prediction = model.predict()
                forecast = prediction.forecast
                forecast.reset_index(inplace=True)
                forecast.rename(columns= {'index':'Date', "Close": "Prediction"},inplace=True)
                #calculate price change in coming period
                change= (forecast['Prediction'].tolist()[-1]/data['Close'].tolist()[-1]) -1
                if change >0:
                    marker= {'color' : 'green'}
                else:
                    marker= {'color' : 'red'}
                #visualize predicted price movement
                # plot the open price
                x = forecast["Date"]
                y = forecast["Prediction"]
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=x, y=y, marker = marker))
                # Set title
                fig.update_layout(
                    title_text="Time Series Plot of "+symbol.upper()+" Predicted Close Prices",
                    yaxis={'showgrid': False},xaxis={'showgrid': False}, width=600, height=400
                )
                st.write(fig)
            if change>0:
                conclusion= """<p style="background-color:rgb(194,229,211,0.5);text-align:left; color:#0d0d0c;font-size:16px;border-radius:0%;">The price of {name} is expected to increase by {change}% in {period} days.
            </p>""".format(name= name, change= np.round(change*100,decimals=2), period= forecast_length)
            else:
                conclusion= """<p style="background-color:rgb(255,204,203,0.5);text-align:left; color:#0d0d0c;font-size:16px;border-radius:0%;">The price of {name} is expected to decrease by {change}% in {period} days.
            </p>""".format(name= name, change= np.round(change*100,decimals=2),period= forecast_length)                
            st.markdown(conclusion, unsafe_allow_html=True)
        except:
            st.error("Error: The symbol entered is incorrect or data is unavailable for the crypto from Yahoo Finance. Please try another option.")
    notes, div= st.columns(2)
    with notes.expander("About AutoTS"):
        st.write("")
        st.markdown("""
        <p>
            AutoTS is an automated time series forecasting method that runs historical data through different models for time series prediction.
            Some of the features of AutoTS are:
            <li>
                - Around 20 pre-defined models like ARIMA, ETS, VECM etc. are available to use with thousands of possible hyperparameters.<br>
                - It works on finding the best model according to the seasonality and trend of the data by genetic programming. <br>
                - AutoTS can handle both Univariate and Multi-Variate Datasets.<br>
                - AutoTS itself clears the data from any NaN columns or outliers which are present.<br>
            </li>
        </p>
        """,
        unsafe_allow_html=True)



if menu_id == "Home":
    
    # convert links to html tags 
    def path_to_image_html(path):
        return '<img src="'+ path + '" width="50" >'
    #create tables of Top Gainers & Losers of the week
    summary_cols=['id','name','symbol','price','percent_change_7d','volume_24h']
    @st.cache
    def create_tables(data, ascending = False):
        df= data.sort_values(by=['percent_change_7d'], ascending=ascending)[summary_cols][:5]
        df= pd.merge(df, meta_data[['Id','Logo']].rename(columns={'Id':'id'}), how='left')
        df.columns= df.columns.str.title()
        #format and style df
        df.rename(columns={"Logo":" ", "Name":"   "}, inplace=True)
        df['Percent_Change_7D']= df['Percent_Change_7D'].apply(lambda x: "{0:,}%".format(np.round(x, decimals=2)))
        df['Price']= df['Price'].apply(lambda x: "$ {:,.6f}".format(np.round(x, decimals=6)))
        df['Volume_24H']= df['Volume_24H'].apply(lambda x: "$ {:,.2f}".format(np.round(x, decimals=6)))        
        # Create the dictionariy to be passed as formatters
        format_dict = {}
        format_dict[' '] = path_to_image_html
        df.rename(columns={'Percent_Change_7D': 'Change 7D'},inplace=True)
        col_order= [" ",'   ','Price', 'Change 7D', 'Volume_24H']
        #convert table to HTML for formatting
        html_table= df[col_order].to_html(border=0,index=False,justify = 'center',escape=False, formatters=format_dict)
        html_table= html_table.replace("<thead>", '<thead style="background-color: rgb(255,204,203,0.5)">')
        html_table= html_table.replace("<tr>", '<tr align="center" style="background-color: mintcream">')
        html_table= html_table.replace("<td>", '<td style="border-left:0px; border-right:0px">')
        html_table= html_table.replace("<th>", '<th style="border-left:0px; border-right:0px">')
        round_style= """
        <html>
        <head>
            <style>
            th{
            border-radius: 0px;
            }
            </style>
            </head>
            <body>"""
        html_table= round_style +html_table +"</body></html>"
        return html_table        

    gainers, losers= st.columns(2)
    with gainers:
        st.subheader("Top Gainers of the Week")
        html_table= create_tables(forecasts,ascending=False)
        st.write(HTML(html_table))
    with losers:
        st.subheader("Top Losers of the Week")
        html_table2= create_tables(forecasts,ascending=True)
        st.write(HTML(html_table2))        
    st.write("")
    st.subheader("Latest News from CryptoNews.com")
    #webscrape CryptoNews to get latest articles
    @st.cache
    def get_news():
        link= "https://cryptonews.com/"
        response= requests.get(link)
        soup= BeautifulSoup(response.content,'html.parser')
        articles= soup.find("section", {'class': "category_contents_details"}).find_all('article')
        images=[]
        titles=[]
        links=[]
        for art in articles:
            #get image
            image=str(art.find('div', {'class': 'img-sized'}))
            image= re.findall(r'\bsrc=".+\b"', image)[0].replace('src="', "").replace('"',"")
            images.append(image)
            #get title
            title = art.find('a',{'class': 'article__title'}).get_text()
            titles.append(title)
            #get article link
            article_link = str(art.find('a',{'class': 'article__title'}))
            article_link = re.findall(r'\bhref=".+">', article_link)[0].replace('href="', "").replace('"',"").replace(">","")
            article_link= link + article_link
            links.append(article_link)
        news= pd.DataFrame({"image": images,"title":titles,"link":links})
        return news
    news= get_news()
    #create the display of the scraped articles in the app
    for i in range(0,len(news)):
        empt,image, block,empt= st.columns([0.25,0.38,1.4,0.25])
        with image:
            st.image(news.image[i])
        with block:
            card=f"""
                    <html>
                    <body>
                        <div class="card">
                            <div class="card-body">
                                <h5 class="card-title">{news.title[i]}</h5>
                                <a href="{news.link[i]}" class="card-link">Read more</a>
                            </div>
                        </div>
                    </body>
                    </html>"""
            st.write("")
            st.write("")
            st.write("")
            st.markdown(card, unsafe_allow_html = True)

if menu_id == "Meta Data":
    st.header("Meta Data of 8,000+ Cryptos")
    tag_select, noth= st.columns(2)
    #create filter option for crypto category
    filter_tag= tag_select.multiselect("Search by category", options= unique_tags, default= None,help='Check cryptos in different categories such as metaverse, defi, stable coins, etc...')
    if filter_tag == []:
        meta_summary= meta_data.drop(["Id","Category","Logo","Tags", "Tag-Names", "Status","Chat","Facebook","Message_Board","Notice"], axis=1)
    else:
        meta_summary= meta_data.copy()
        #transform the category column values in the meta data to list
        tags_records=[]
        for i in range(0, len(meta_summary)):
            coin_tags= meta_summary['Tag-Names'][i].split(",")
            coin_tags=[sub.replace("'", "").strip() for sub in coin_tags]
            tags_records.append(coin_tags)
        tags_df= pd.DataFrame({'Tag-Names': tags_records})
        meta_summary = meta_summary.drop('Tag-Names', axis=1)
        meta_summary= pd.concat([meta_summary, tags_df], axis=1)
        #get cryptos whose category match user filter
        match_tag=[]
        for i in range(0, len(meta_summary)):
            result= any(elem in meta_summary['Tag-Names'][i] for elem in filter_tag)
            match_tag.append(result)
        meta_summary['match']=match_tag
        meta_summary= meta_summary[meta_summary['match'] == True]
        meta_summary.drop(["Id","Category","Logo","Tags","Tag-Names","Status","Chat","Facebook","Message_Board","Notice"], axis=1, inplace=True)
    #display final meta data table using Ag Grid 
    gb = GridOptionsBuilder.from_dataframe(meta_summary)
    gb.configure_pagination(paginationPageSize=25,paginationAutoPageSize= False)
    #format columns with links to become clickable
    gb.configure_columns(column_names=["Website", "Twitter","Explorer", "Reddit","Technical_Doc"],
                            cellRenderer=JsCode('''function(params) {if (params.value != "N/A") {return '<a href="' + params.value + '" target="_blank">'+ params.value+'</a>'}}'''),
                            width=300)
    gridOptions = gb.build()
    AgGrid(meta_summary, gridOptions= gridOptions, enable_enterprise_modules=True, allow_unsafe_jscode= True, height=700)

if menu_id == "Forecast Analysis":
    st.header("Forecasts Overview of 8,000+ Cryptos")
    categories, platforms= st.columns(2)
    #create filter option for crypto category
    filter_tag= categories.multiselect("Category", options= unique_tags, default= None, key= 41, help='Select cryptos in different categories such as metaverse, defi, stable coins, etc...')
    if filter_tag != []:
        #get ids of cryptos meeting tag search
        meta_df= meta_data.copy()
        tags_records=[]
        for i in range(0, len(meta_df)):
            coin_tags= meta_df['Tag-Names'][i].split(",")
            coin_tags=[sub.replace("'", "").strip() for sub in coin_tags]
            tags_records.append(coin_tags)
        tags_df= pd.DataFrame({'Tag-Names': tags_records})
        meta_df = meta_df.drop('Tag-Names', axis=1)
        meta_df= pd.concat([meta_df, tags_df], axis=1)
        match_tag=[]
        for i in range(0, len(meta_df)):
            result= any(elem in meta_df['Tag-Names'][i] for elem in filter_tag)
            match_tag.append(result)
        meta_df['match']=match_tag
        meta_df= meta_df[meta_df['match'] == True]
        matchs_ids= meta_df['Id'].values.tolist()
        #filter data based on category selected
        filter1_df= forecasts[forecasts.id.isin(matchs_ids)]
    else:
        filter1_df= forecasts

    #create blockchain type filter options
    platforms_df_filter = filter1_df.platform.value_counts().rename_axis('blockchain').reset_index(name='counts')
    top_platforms= ((platforms_df_filter.blockchain[:11])).tolist()
    other_platforms= (platforms_df_filter.blockchain[11:]).tolist()
    if other_platforms != []:
        options= ["All"]+ top_platforms + ["Other"]
    elif len(top_platforms) == 1:
        options= top_platforms
    else:
        options= ["All"]+ top_platforms
    options.sort(reverse= False)
    if "All" in options:
        default = "All"
    else:
        default= options
    blockchain = platforms.multiselect("Blockchain Type", options, default= default, help="Filter results based on the crypto's blockchain platform such as ethereum, solana, etc...", key= 2)

    if "All" in blockchain:
        blockchain= (filter1_df.platform.unique()).tolist()
    if "Other" in blockchain:
        blockchain= other_platforms

    filter1_df= filter1_df[filter1_df.platform.isin(blockchain)]
    filter1_df.dropna(subset= ["CP_3m", "WI_3m", "GC_3m"], how= 'all', inplace= True)
    ## create filters for market cap & date_added
    cap1, cap2, date1, date2= st.columns(4)
    if filter_tag != []:
        value_lower= np.nanmin(filter1_df.market_cap)
    else:
        value_lower= -1.00
    cap_lower= cap1.number_input(label= 'Min. Market Cap', min_value= -1.00, max_value= np.nanmax(filter1_df.market_cap), value= value_lower,step= 1000000.00,help="Value of -1 represents cryptos whose market cap is unknown.")
    cap_upper= cap2.number_input(label= 'Max. Market Cap', min_value= cap_lower, max_value= np.nanmax(filter1_df.market_cap), value= np.nanmax(filter1_df.market_cap),step= 1000000.00)
    date_min= date1.date_input("Min. Date Added", min_value= filter1_df.date_added.min(), max_value= filter1_df.date_added.max(), value= filter1_df.date_added.min())
    date_max= date2.date_input("Max. Date Added", min_value= date_min, max_value= filter1_df.date_added.max(), value= filter1_df.date_added.max())
    
    periods, trading_sug= st.columns(2)
    info_msg, chng_checkbox= st.columns(2)

    #select forecast analysis period for calculations
    user_period = periods.selectbox("Select Forecast Analysis Period", options=('short-term','medium-term','long-term'),index=2, help= "Short-term: 3 months - Medium-term: 1 year - Long-term: 3 to 5 years", key= 1)
    #perform forecast analysis on selected period
    forecast_analysis(period= user_period, data= forecasts)

    #create trading recommendation filter
    action= trading_sug.selectbox(label="Trading Recommendation",options= ["All"]+ sorted((forecasts.Action.unique()).tolist()), index=0)

    #add note to user on option to select cryptos from table
    info= """<p style="background-color:rgb(255,204,203,0.5);text-align:left; color:#0d0d0c;font-size:16px;border-radius:0%;">Select any crypto to view summary dashboard below!
            </p>
        """
    info_msg.markdown(info, unsafe_allow_html=True)

    #add legend on forecasts references
    legend= """<i><text style="text-align:left; color:gray;font-size:13px;">Forecasts References: Cp- CryptoPredictions.com | Wi - Wallet Investor | Gc - Gov Capital
                </text></i>
            """
    st.markdown(legend,unsafe_allow_html=True)
    
    #add checkbox to allow user to add % change columns to data viewed
    with chng_checkbox:
        # st.write("")
        show_change= st.checkbox("Display price % change calculation columns", key=42)
    if show_change:
        figs_to_display=['name','symbol','platform','date_added','Action','price', 'market_cap', 'data_source'] +['Min_Price', 'Max_Price', 'Avg_Price',
        'Median_Price','Min_Change', 'Max_Change', 'Avg_Change','Median_Change'] + cols_summary[user_period] + ['id']
    else:
        figs_to_display=['name','symbol','platform','date_added','Action','price', 'market_cap', 'data_source'] +['Min_Price', 'Max_Price', 'Avg_Price',
            'Median_Price'] + cols_summary[user_period] + ['id']

    ## display final filtered data
    if "All" in action:
        mask= (forecasts['platform'].isin(blockchain)) & (forecasts['date_added'] >= date_min) & (forecasts['date_added'] <= date_max) & (forecasts['market_cap'] >= cap_lower) & (forecasts['market_cap'] <= cap_upper)
    else:
        mask= (forecasts['platform'].isin(blockchain)) & (forecasts['date_added'] >= date_min) & (forecasts['date_added'] <= date_max) & (forecasts['market_cap'] >= cap_lower) & (forecasts['market_cap'] <= cap_upper) & (forecasts['Action'] == action)

    filtered_df = forecasts.loc[mask]
    if filter_tag != []:
        filtered_df= filtered_df[filtered_df.id.isin(matchs_ids)]
    filtered_df.dropna(subset= cols_summary[user_period], how= 'all', inplace= True)
    filtered_df.reset_index(drop=True,inplace=True)

    # set Jscode color styler to color Action when recommendation is buy or strong buy
    cellsytle_jscode = JsCode(
            """
        function(params) {
            if (params.value.includes('Strong Buy')) {
                return {
                    'backgroundColor': 'limegreen'
                }
            } else {
                if (params.value.includes('Buy')) {
                return {
                    'color': 'black',
                    'backgroundColor': 'lightgreen'
                }
                }
            }
        };
        """
        )
    # create value formatter to round Market Cap in Ag Grid table
    currency_formatter1= JsCode(
        """
        function (params) {
            if ((params.value != 'nan') && (params.value != '-1.00')){
                var sansDec = params.value.toFixed(2);
                var formatted = sansDec.replace(/\B(?=(\d{3})+(?!\d))/g, ',');
                return '$' + `${formatted}`;
            } else {
                return params.value
            }      
        };
        """
    )
    # create value formatter to round other numeric columns in Ag Grid table based on varying decimals
    currency_formatter2= JsCode(
        """
        function (params) {
            if (params.value != 'nan'){
                var sansDec = params.value.toFixed(2);
                if (sansDec != 0.00){
                    var formatted = sansDec.replace(/\B(?=(\d{3})+(?!\d))/g, ',');
                    return '$' + `${formatted}`;
                } else {
                    var sansDec2 = params.value.toFixed(3);
                    if (sansDec2 != 0.000) {
                        return '$' + `${sansDec2}`;
                    } else{
                        var sansDec3 = params.value.toFixed(4);
                        if (sansDec3 != 0.0000) {
                            return '$' + `${sansDec3}`;
                        } else {
                            var sansDec4 = params.value.toFixed(5);
                            if (sansDec4 != 0.00000) {
                                return '$' + `${sansDec4}`;
                            } else {
                                return '$' + `${params.value}`;
                            }
                        }
                    }
                } 
            } else {
                return params.value
            }      
        };
        """
    )
    def round_num(x):
        if (np.round(x, decimals = 2) != 0.00):
            rounded= np.round(x, decimals = 2)
        elif (np.round(x, decimals = 3) != 0.000):
            rounded= np.round(x, decimals = 3)
        elif (np.round(x, decimals = 4) != 0.0000):
            rounded= np.round(x, decimals = 4)
        elif (np.round(x, decimals = 5) != 0.00000):
            rounded= np.round(x, decimals = 5)
        elif (np.round(x, decimals = 6) != 0.000000):
            rounded= np.round(x, decimals = 6)        
        else:
            rounded= x
        return rounded
    def round_million(x):
        if np.round(x/1000000,decimals=2) >= 1.0:
            round_x= np.round(x/1000000,decimals=2)
            x_met= "{:,.2f}".format(round_x)+" M"
        elif np.round(x/1000000,decimals=3) > 0.001:
            round_x= np.round(x/1000000,decimals=3)
            x_met= "{:,.3f}".format(round_x)+" M"
        else: 
            round_x= np.round(x,decimals=2)
            x_met= "{:,.2f}".format(round_x) 
        return x_met

    years_period={"short-term": "3 months", "medium-term": "1 year", "long-term": "3-5 years"}

    df_show= filtered_df[figs_to_display]
    df_show.rename(columns={'platform':'blockchain'}, inplace=True)
    df_show.columns = df_show.columns.str.title()
    #build final table in Ag Grid
    gb = GridOptionsBuilder.from_dataframe(df_show.drop(['Id'], axis=1))
    gb.configure_pagination(paginationPageSize=25,paginationAutoPageSize= False)
    gb.configure_side_bar(columns_panel = False,filters_panel= True)
    gb.configure_columns(column_names=["Price", "Min_Price", "Max_Price", "Avg_Price", "Median_Price"], type=["numericColumn","numberColumnFilter"], valueFormatter=currency_formatter2)
    cols_summary_formatted= [x.title() for x in cols_summary[user_period]]
    gb.configure_columns(column_names= cols_summary_formatted, type=["numericColumn","numberColumnFilter"], valueFormatter=currency_formatter2)
    gb.configure_column("Market_Cap", type=["numericColumn","numberColumnFilter"], valueFormatter=currency_formatter1)
    gb.configure_column("Action", cellStyle=cellsytle_jscode)
    #format price change columns if included to show "%" sign
    if show_change:
        gb.configure_columns(column_names=['Min_Change', 'Max_Change', 'Avg_Change','Median_Change'],
                        cellRenderer=JsCode('''function(params) {return (params.value * 100).toFixed(1) + '%'}'''))
    gb.configure_selection(selection_mode="single", use_checkbox=True) #allow selection of rows
    gb.configure_default_column(groupable=True, value=True, enableRowGroup=True, editable=False)
    gridOptions = gb.build()
    data = AgGrid(df_show, gridOptions= gridOptions, enable_enterprise_modules=True, allow_unsafe_jscode= True, 
            update_mode=GridUpdateMode.SELECTION_CHANGED, height=700)
    
    #define the selected row data in the table
    selected_row = data["selected_rows"]
    selected_row = pd.DataFrame(selected_row) 
    #create dashboard to be displayed when crypto is selected
    if len(selected_row) != 0:
        id_sel= selected_row.Id.tolist()[0]
        pic, header, empt1= st.columns([0.5,1,6])
        logo= meta_data[meta_data.Id == id_sel]['Logo'].tolist()[0]
        with pic:
            st.image(logo)
        with header:
            coin_name= selected_row.Name.tolist()[0]
            st.markdown("### "+coin_name)
            source = forecasts[forecasts.id == id_sel]['data_source'].tolist()[0]
            st.markdown("Data Source: " + source)

        metric1, metric2,metric3,metric4,metric5= st.columns(5)
        with metric1:  #show price & change
            price_met= "$"+f"{round_num(selected_row.Price.tolist()[0]):,}"
            percent_change_30d= f"{np.round(forecasts[forecasts.id == id_sel]['percent_change_30d'].tolist()[0],decimals=1):,}"+"%"
            if forecasts[forecasts.id == id_sel]['percent_change_30d'].tolist()[0] == 0:
                delta_color="off"
            else:
                delta_color="normal"
            st.metric("Price | 30d Change", value=price_met, delta= percent_change_30d, delta_color= delta_color)
        
        with metric2:  #show volume & change
            volume= forecasts[forecasts.id == id_sel]['volume_24h'].tolist()[0]                
            volume_change_24h= f"{np.round(forecasts[forecasts.id == id_sel]['volume_change_24h'].tolist()[0],decimals=1):,}"+"%"
            if forecasts[forecasts.id == id_sel]['volume_change_24h'].tolist()[0] == 0:
                delta_color="off"
            else:
                delta_color="normal"
            st.metric("Volume | 24h Change", value="$"+round_million(volume), delta= volume_change_24h,delta_color= delta_color)

        with metric3:  #show market cap
            mrkt_cap= selected_row.Market_Cap.tolist()[0]
            if mrkt_cap == -1.00:
                cap_met= np.nan
            else:
                cap_met= "$"+round_million(mrkt_cap)
            st.metric("Market Cap", value=cap_met)
        
        with metric4: #show max supply
            max_supply= forecasts[forecasts.id == id_sel]['max_supply'].tolist()[0]
            if np.isnan(max_supply):
                max_sup_met= max_supply
            else:
                max_sup_met= round_million(max_supply)
            st.metric("Max Supply", value=max_sup_met)
        
        with metric5:  #show circulating supply
            circulating_supply= np.round(forecasts[forecasts.id == id_sel]['circulating_supply'].tolist()[0],decimals=0)
            cir_sup_met= round_million(circulating_supply)
            st.metric("Circulating Supply", value=cir_sup_met)

        plot, div, links= st.columns([2,0.1,1])
        with links:
            #create filter to allow user to visualize different period in dashboard
            periods_options=['short-term','medium-term','long-term']
            period_filter = st.radio("Explore other forecast periods:", options=periods_options,index= periods_options.index(user_period), help= "Short-term: 3 months - Medium-term: 1 year - Long-term: 3 to 5 years")
            #create list of links
            url_icons={'Website': "fa fa-globe", "Twitter": "fa fa-twitter", "Reddit": "fa fa-reddit-alien", 'Technical_Doc': "fa fa-object-group",
                        'Explorer': "fa fa-qrcode"}
            col_links=['Website', 'Twitter','Reddit','Explorer','Technical_Doc']
            coin_links= meta_data[meta_data.Id == id_sel][col_links].values.tolist()[0]
            link_dict= dict(zip(col_links,coin_links))
            link_text=[]
            for key, value in link_dict.items():
                if value != "N/A":
                    link_text.append('<i class="'+url_icons[key]+'"></i><text> </text><text> </text>'+'<a href="'+str(value)+'" class="card-link"> '+str(key)+"</a>")
            link_text= '<p> </p>'.join(link_text)
            string= f"""
                    <html>
                    <head>
                        <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/4.7.0/css/font-awesome.min.css">
                    </head>
                    <body>
                        <div class="card" style="width: 18rem;">
                            <div class="card-body">
                                <h5 class="card-title">Useful Links</h5>
                                {link_text}
                                <p> </p>
                            </div>
                        </div>
                    </body>
                    </html>"""
            st.markdown(string, unsafe_allow_html = True)
            #display list of categories of selected crypto
            coin_tag= meta_data[meta_data.Id == id_sel]['Tag-Names'].tolist()[0]
            tags_text= f"""
                    <div class="card" >
                        <div class="card-body">
                            <h5 class="card-title">Category</h5>
                            <p> {coin_tag} </p>
                        </div>
                    </div>"""
            if tags_text != 'N/A':
                st.markdown(tags_text, unsafe_allow_html = True)

        with plot: #visualize forecasts per source
            coin_forecasts= forecasts[forecasts.id == id_sel][cols_summary[period_filter]]
            coin_forecasts.columns= coin_forecasts.columns.str.title()
            coin_forecasts= coin_forecasts.T.reset_index()
            coin_forecasts.columns= coin_forecasts.columns.astype(str)
            coin_forecasts.rename(columns={coin_forecasts.columns[1]: "Forecast"}, inplace=True)
            coin_forecasts['Source']= coin_forecasts['index'].apply(lambda x: re.sub(r"Dc.+", "Digital Coin Price", x))
            coin_forecasts['Source']= coin_forecasts['Source'].apply(lambda x: re.sub(r"Tb.+", "Trading Beasts", x))
            coin_forecasts['Source']= coin_forecasts['Source'].apply(lambda x: re.sub(r"Cp.+", "CryptoPredictions.com", x))
            coin_forecasts['Source']= coin_forecasts['Source'].apply(lambda x: re.sub(r"Wi.+", "Wallet Investor", x))
            coin_forecasts['Source']= coin_forecasts['Source'].apply(lambda x: re.sub(r"Gc.+", "GovCapital", x))
            plt_header= "Expected Price of "+ coin_name + " in "+ years_period[period_filter]
            st.markdown("##### "+plt_header)
            fig = px.bar(coin_forecasts, y='Forecast', x='Source')
            fig.update_yaxes(title='')
            fig.update_xaxes(title='')
            fig.update_traces(marker_color='lightskyblue')
            fig.add_hline(y= selected_row.Price.tolist()[0], line_dash= "dash", annotation_text= "Current Price")
            fig.update_layout(xaxis={'categoryorder':'total ascending', 'showgrid': False}, yaxis={'showgrid': False}, showlegend= True)
            st.write(fig)
        #display "Notice" is available from CoinMarketCap
        coin_notice= meta_data[meta_data.Id == id_sel]['Notice'].values.tolist()[0]
        if coin_notice != "N/A":
            notice_header= """
                    <html>
                    <head>
                        <link rel="stylesheet" href="https://fonts.googleapis.com/icon?family=Material+Icons">
                    </head>
                    <body>
                        <div class="card" style="width: 18rem;">
                            <div class="card-body">
                                <i class="material-icons" style="color:red;">warning</i><text><b>     Notice from CoinMarketCap:  </b></text>
                            </div>
                        </div>
                    </body>
                    </html>"""
            st.markdown(notice_header, unsafe_allow_html = True)
            st.write(coin_notice)

    st.write("Data last updated:", datetime.datetime.strptime(forecasts.last_updated[0],"%Y-%m-%dT%H:%M:%S.000Z"),"(GMT)")
    
    #create data download button
    @st.cache
    def convert_df(df):
        # IMPORTANT: Cache the conversion to prevent computation on every rerun
        return df.to_csv(index= False).encode('utf-8')
    csv_data= convert_df(filtered_df[figs_to_display])
    st.download_button(label="Export table as CSV", data=csv_data, file_name='crypto_forecasts.csv', mime='text/csv')

### SINGLE CRYPTO SECTION - REAL TIME FORECASTS & PRICES
if menu_id== "Real Time Forecasts":
    st.header("Real Time Forecasts & Recommendations")
    # get recent prices of selected cryptos
    coin_cap_key= '1500a447-eea6-4433-abce-a28f4072eece'

    # transform ids list to string to match query format
    def list_to_str(l):
        str1=","
        return (str1.join([str(elem) for elem in l]))
    @st.cache(allow_output_mutation=True)
    def get_quotes(select_ids):
        # transform list of selected ids to string for query
        ids_str= list_to_str(select_ids)
        url = 'https://pro-api.coinmarketcap.com/v2/cryptocurrency/quotes/latest'
        parameters = {
        'id': ids_str,
        'convert':'USD'
        }
        headers = {
        'Accepts': 'application/json',
        'X-CMC_PRO_API_KEY': coin_cap_key,
            'Accept-Encoding': 'deflate, gzip' 
        }
        session = Session()
        session.headers.update(headers)
        try:
            response = session.get(url, params=parameters)
            data = orjson.loads(response.text)
        except (ConnectionError, Timeout, TooManyRedirects) as e:
            print(e)
        # convert listing to dataframe
        df=pd.DataFrame(data['data'].values())
        # decompose the quote field to obtain data on prices and changes
        quotes_dic=[]
        for quote in df.quote.tolist():
            quotes_dic.append(quote['USD'])
        df.drop('quote',axis=1, inplace= True)
        df=pd.concat([df, pd.DataFrame(quotes_dic)], axis=1)
        #clean and transform data
        df.drop(['num_market_pairs','tags','last_updated','is_active','is_fiat'],axis=1, inplace=True)
        df.rename(columns={'market_cap': "verified_market_cap", 'circulating_supply': 'verified_circulating_supply'},inplace=True)
        df['market_cap']= df.apply(lambda row: row['verified_market_cap'] if row['verified_market_cap'] != 0 else row['self_reported_market_cap'], axis=1)
        df['circulating_supply']= df.apply(lambda row: row['verified_circulating_supply'] if row['verified_circulating_supply'] != 0 else row['self_reported_circulating_supply'], axis=1)
        types= []
        for i in range(0, len(df)):
            if df.verified_market_cap[i] != 0:
                types.append("Verified")
            elif (df.self_reported_market_cap[i] != 0) & (np.isnan(df.self_reported_market_cap[i]) == False):
                types.append("Self-Reported")
            else:
                types.append("N/A")
        df['data_source']= types
        df['market_cap'].fillna(0, inplace=True)
        df['date_added']= df.date_added.apply(lambda x: x.split("T")[0])
        df['date_added']= df.date_added.apply(lambda x: datetime.datetime.strptime(x,"%Y-%m-%d")).dt.date
        df['platform']= df['platform'].fillna('N/A')
        df['platform']= df['platform'].apply(lambda x: x['name'] if x != "N/A"  else x)
        df['identifier']=df.name.str.lower()+"-"+df.symbol
        if sum(df.name.isin(['USD Coin'])) == 1:
            df.loc[forecasts.name == "USD Coin", 'identifier'] = "usd//coin-USDC"
        return df

    def get_forecasts_by_coin(data):
        ## Scraping Crypto Forecasts from Digital Coin Price
        forecast_dc_3m=[]
        forecast_dc_1y=[]
        forecast_dc_5y=[]
        for coin in data.slug:
            link= "https://digitalcoinprice.com/forecast/"+ coin
            response = requests.get(link)
            try:
                soup= BeautifulSoup(response.content, 'html.parser')
                st.write(soup)
                try:
                    #get needed table columns for up to 5 years of data
                    # results= soup.find('div',{'class': "forecast_yearlyTable__1ESY_"}).find('table', {'class': 'dcpTable m-0 table'}).find('tbody').find_all('tr')[:6]
                    results= soup.find_all('table', {'class': 'dcpTable'})[2].find('tbody').find_all('tr')[:6]
                    #scrape table data
                    months=[i for i in range(1,13)]
                    columns=['Month']
                    nums=[months]
                    for result in results:
                        #get column headers
                        try:
                            columns.append(result.find_all("td")[0].contents[2])
                        except:
                            columns.append(np.nan)
                        #get monthly forecast
                        values=[]
                        for td in result.find_all("td")[1:]:
                            try:
                                values.append(float(td.find('span', {'class':'CurrencyPrice'}).get_text().strip("$").replace(",",""))) #returns forecasted price
                                #values.append(float(td.contents[1].get_text().strip("%"))) # returns forecasted change in price
                            except:
                                values.append(np.nan)
                        nums.append(values)
                    #reshape data into dataframe
                    df= pd.DataFrame(list(zip(*nums)),columns=columns).melt(id_vars='Month', var_name= 'Year', value_name='Forecast')
                    #create date column to use for forecast calculations
                    df['date']= (df.Year.astype(str)+ "-"+df.Month.astype(str)).apply(lambda x: datetime.datetime.strptime(x,"%Y-%m")).dt.date
                    # calculate forecast_3m
                    Forecast_3m = float(df[df.date == ((date.today() + relativedelta(months=+3)).replace(day=1))].Forecast)
                    # calculate forecast_1y
                    Forecast_1y = float(df[df.date == ((date.today() + relativedelta(months=+12)).replace(day=1))].Forecast)
                    # calculate forecast_5y
                    Forecast_5y = float(df[df.date == ((date.today() + relativedelta(months=+60)).replace(day=1))].Forecast)
                    forecast_dc_3m.append(Forecast_3m)
                    forecast_dc_1y.append(Forecast_1y)
                    forecast_dc_5y.append(Forecast_5y)
                except:
                    forecast_dc_3m.append(np.nan)
                    forecast_dc_1y.append(np.nan)
                    forecast_dc_5y.append(np.nan)
            except:
                forecast_dc_3m.append(np.nan)
                forecast_dc_1y.append(np.nan)
                forecast_dc_5y.append(np.nan)
        # add DC data to listings table
        data['DC_3m']=forecast_dc_3m
        data['DC_1y']=forecast_dc_1y
        data['DC_5y']=forecast_dc_5y

        ## Scraping Crypto Forecasts from Trading Beasts
        forecast_tb_3m=[]
        forecast_tb_1y=[]
        forecast_tb_3y=[]
        for coin in data.slug:
            link= "https://tradingbeasts.com/price-prediction/"+ coin
            response = requests.get(link)
            try:
                soup= BeautifulSoup(response.content, 'html.parser')
                try:
                    #get tables
                    results=soup.find_all('table',{'class': 'table-striped'})
                    # set headers reference
                    headers=['Month', 'Minimum Price', 'Maximum Price', 'Average Price', 'Change']
                    # get headers and confirm they match our script
                    link_header=[x.get_text() for x in results[0].find('thead').find_all('th')]
                    records=[]
                    if headers==link_header:
                        for table in results:
                            # get records per row for each month of the year
                            for row in table.find('tbody').find_all('tr'):
                                records.append([td.get_text() for td in row.find_all('td')])
                        #combine records into df
                        df=pd.DataFrame(records,columns=headers)
                        #transform numerical columns
                        cols=['Minimum Price', 'Maximum Price', 'Average Price', 'Change']
                        df[cols]=df[cols].apply(lambda x: x.str.strip().str.strip(' %').str.replace(',',"").astype(float))
                        #transform date column
                        df['Month']= df['Month'].apply(lambda x: datetime.datetime.strptime(x,"%B %Y")).dt.date 
                        # calculate forecast_3m
                        Forecast_3m = float(df[df.Month == ((date.today() + relativedelta(months=+3)).replace(day=1))]['Average Price'])
                        # calculate forecast_1y
                        Forecast_1y = float(df[df.Month == ((date.today() + relativedelta(months=+12)).replace(day=1))]['Average Price'])
                        # calculate forecast_5y
                        Forecast_5y = float(df[df.Month == df.Month.max()]['Average Price'])
                        forecast_tb_3m.append(Forecast_3m)
                        forecast_tb_1y.append(Forecast_1y)
                        forecast_tb_3y.append(Forecast_5y)
                    else:
                        slug2= coin.split("-")[0]
                        link2= "https://tradingbeasts.com/price-prediction/"+ slug2
                        response2 = requests.get(link2)
                        try:
                            soup2= BeautifulSoup(response2.content, 'html.parser')
                            try:
                                #get tables
                                results2=soup2.find_all('table',{'class': 'table-striped'})
                                # set headers reference
                                headers=['Month', 'Minimum Price', 'Maximum Price', 'Average Price', 'Change']
                                # get headers and confirm they match our script
                                link_header2=[x.get_text() for x in results2[0].find('thead').find_all('th')]
                                records2=[]
                                if headers==link_header2:
                                    for table in results2:
                                        # get records per row for each month of the year
                                        for row in table.find('tbody').find_all('tr'):
                                            records2.append([td.get_text() for td in row.find_all('td')])
                                    #combine records into df
                                    df2=pd.DataFrame(records2,columns=headers)
                                    #transform numerical columns
                                    cols2=['Minimum Price', 'Maximum Price', 'Average Price', 'Change']
                                    df2[cols2]=df2[cols2].apply(lambda x: x.str.strip().str.strip(' %').str.replace(',',"").astype(float))
                                    #transform date column
                                    df2['Month']= df2['Month'].apply(lambda x: datetime.datetime.strptime(x,"%B %Y")).dt.date 
                                    # calculate forecast_3m
                                    Forecast_3m = float(df[df.Month == ((date.today() + relativedelta(months=+3)).replace(day=1))]['Average Price'])
                                    # calculate forecast_1y
                                    Forecast_1y = float(df[df.Month == ((date.today() + relativedelta(months=+12)).replace(day=1))]['Average Price'])
                                    # calculate forecast_5y
                                    Forecast_5y = float(df[df.Month == df.Month.max()]['Average Price'])
                                    forecast_tb_3m.append(Forecast_3m)
                                    forecast_tb_1y.append(Forecast_1y)
                                    forecast_tb_3y.append(Forecast_5y)
                                else:
                                    slug3= coin.split("-")[-1]
                                    link3= "https://tradingbeasts.com/price-prediction/"+ slug3
                                    response3 = requests.get(link3)
                                    try:
                                        soup3= BeautifulSoup(response3.content, 'html.parser')
                                        try:
                                            #get tables
                                            results3=soup3.find_all('table',{'class': 'table-striped'})
                                            # set headers reference
                                            headers=['Month', 'Minimum Price', 'Maximum Price', 'Average Price', 'Change']
                                            # get headers and confirm they match our script
                                            link_header3=[x.get_text() for x in results3[0].find('thead').find_all('th')]
                                            records3=[]
                                            if headers==link_header3:
                                                for table in results3:
                                                    # get records per row for each month of the year
                                                    for row in table.find('tbody').find_all('tr'):
                                                        records3.append([td.get_text() for td in row.find_all('td')])
                                                #combine records into df
                                                df3=pd.DataFrame(records3,columns=headers)
                                                #transform numerical columns
                                                cols3=['Minimum Price', 'Maximum Price', 'Average Price', 'Change']
                                                df3[cols3]=df3[cols3].apply(lambda x: x.str.strip().str.strip(' %').str.replace(',',"").astype(float))
                                                #transform date column
                                                df3['Month']= df3['Month'].apply(lambda x: datetime.datetime.strptime(x,"%B %Y")).dt.date 
                                                # calculate forecast_3m
                                                Forecast_3m = float(df[df.Month == ((date.today() + relativedelta(months=+3)).replace(day=1))]['Average Price'])
                                                # calculate forecast_1y
                                                Forecast_1y = float(df[df.Month == ((date.today() + relativedelta(months=+12)).replace(day=1))]['Average Price'])
                                                # calculate forecast_5y
                                                Forecast_5y = float(df[df.Month == df.Month.max()]['Average Price'])
                                                forecast_tb_3m.append(Forecast_3m)
                                                forecast_tb_1y.append(Forecast_1y)
                                                forecast_tb_3y.append(Forecast_5y)             
                                            else:
                                                forecast_tb_3m.append(np.nan)
                                                forecast_tb_1y.append(np.nan)
                                                forecast_tb_3y.append(np.nan)
                                        except:
                                            forecast_tb_3m.append(np.nan)
                                            forecast_tb_1y.append(np.nan)
                                            forecast_tb_3y.append(np.nan)                                        
                                    except:
                                        forecast_tb_3m.append(np.nan)
                                        forecast_tb_1y.append(np.nan)
                                        forecast_tb_3y.append(np.nan)                                        
                            except:
                                forecast_tb_3m.append(np.nan)
                                forecast_tb_1y.append(np.nan)
                                forecast_tb_3y.append(np.nan)                            
                        except:
                            forecast_tb_3m.append(np.nan)
                            forecast_tb_1y.append(np.nan)
                            forecast_tb_3y.append(np.nan)                                       
                except:
                    forecast_tb_3m.append(np.nan)
                    forecast_tb_1y.append(np.nan)
                    forecast_tb_3y.append(np.nan)            
            except:
                forecast_tb_3m.append(np.nan)
                forecast_tb_1y.append(np.nan)
                forecast_tb_3y.append(np.nan)       
        # add TB data to listings table
        data['TB_3m']=forecast_tb_3m
        data['TB_1y']=forecast_tb_1y
        data['TB_3y']=forecast_tb_3y

        ## Scraping Crypto Forecasts from CryptoPredictions.com
        forecast_cp_3m=[]
        forecast_cp_1y=[]
        forecast_cp_4y=[]
        for coin in data.slug:
            link= "https://cryptopredictions.com/"+ coin +"/"
            response = requests.get(link)
            try:
                soup= BeautifulSoup(response.content, 'html.parser')
                try:
                    #get tables of monthly forecasts per year
                    results=soup.find_all('table')
                    # set headers reference
                    headers=['Month', 'Minimum price', 'Maximum price', 'Average price', 'Change']
                    # get headers and confirm they match our script
                    link_header=[x.get_text() for x in results[0].find('thead').find_all('th')]
                    records=[]
                    if headers==link_header:
                        for table in results:
                            # get records per row for each month of the year
                            for row in table.find('tbody').find_all('tr'):
                                records.append([td.get_text() for td in row.find_all('td')])
                        #combine records into df
                        df=pd.DataFrame(records,columns=headers)
                        #transform numerical columns
                        cols=['Minimum price', 'Maximum price', 'Average price', 'Change']
                        df[cols]=df[cols].apply(lambda x: x.str.strip().str.strip(' %').str.strip("$").str.replace(',',"").astype(float))
                        #transform date column
                        df['Month']= df['Month'].apply(lambda x: datetime.datetime.strptime(x,"%B %Y")).dt.date 
                        # calculate forecast_3m
                        Forecast_3m = float(df[df.Month == ((date.today() + relativedelta(months=+3)).replace(day=1))]['Change'])
                        # calculate forecast_1y
                        Forecast_1y = float(df[df.Month == ((date.today() + relativedelta(months=+12)).replace(day=1))]['Change'])
                        # calculate forecast_5y
                        Forecast_5y = float(df[df.Month == df.Month.max()]['Change'])             
                        forecast_cp_3m.append(Forecast_3m)
                        forecast_cp_1y.append(Forecast_1y)
                        forecast_cp_4y.append(Forecast_5y)
                    else:
                        forecast_cp_3m.append(np.nan)
                        forecast_cp_1y.append(np.nan)
                        forecast_cp_4y.append(np.nan)
                except:
                        forecast_cp_3m.append(np.nan)
                        forecast_cp_1y.append(np.nan)
                        forecast_cp_4y.append(np.nan)            
            except:
                forecast_cp_3m.append(np.nan)
                forecast_cp_1y.append(np.nan)
                forecast_cp_4y.append(np.nan)       

        # add CP data to listings table
        data['CP_3m']=forecast_cp_3m
        data['CP_1y']=forecast_cp_1y
        data['CP_4y']=forecast_cp_4y

    def get_forecast_tables(select_identifier):
        ## Scraping Crypto Forecasts from Wallet Investor
        identifier_wi=[]
        forecast_wi_3m=[]
        forecast_wi_1y=[]
        forecast_wi_5y=[]
        #set header reference as consistency check when scraping
        headers_wi= ['Name','7d Forecast','3m Forecast', '1y Forecast', '5y Forecast', 'Price Graph (1y)']
        for i in range(1,109):
            website ="https://walletinvestor.com/forecast?page="+str(i)+"&per-page=100"
            response = requests.get(website)
            soup= BeautifulSoup(response.content, 'html.parser')
            link_headers= [x.get_text().strip() for x in soup.find('table', {'class': 'table-condensed'}).find('thead').find_all('th')]
            if headers_wi==link_headers:
                results= soup.find('table', {'class': 'table-condensed'}).find('tbody').find_all('tr')
                for result in results:
                    #name
                    try:
                        name, ticker= result.find('a',{'class':['green crypto-ticker-name','red crypto-ticker-name']}).contents
                        ticker= ticker.get_text()
                        name= name.lower()
                        identifier_wi.append(name+"-"+ticker)
                    except:
                        identifier_wi.append(np.nan)
                    #forecast_3m
                    try:
                        forecast_wi_3m.append(float(result.find('td',{'class':'table-cell-label kv-align-right kv-align-middle w0', 'data-col-seq':'2'})
                                        .get_text().strip(" %").strip("+")))
                    except:
                        forecast_wi_3m.append(np.nan)
                    #forecast_1y
                    try:
                        forecast_wi_1y.append(float(result.find('td',{'class':'table-cell-label kv-align-right kv-align-middle w0', 'data-col-seq':'3'})
                                        .get_text().strip(" %").strip("+")))
                    except:
                        forecast_wi_1y.append(np.nan)
                    #forecast_5y
                    try:
                        forecast_wi_5y.append(float(result.find('td',{'class':'table-cell-label kv-align-right kv-align-middle w0', 'data-col-seq':'4'})
                                        .get_text().strip(" %").strip("+")))
                    except:
                        forecast_wi_5y.append(np.nan)
            else:
                pass        
            check =  all(elem in identifier_wi  for elem in select_identifier) #stop scraping when all cryptos selected are found
            if check == True:
                break 
        wallet_investor= pd.DataFrame({'identifier': identifier_wi,"WI_3m":forecast_wi_3m,
                                "WI_1y":forecast_wi_1y,"WI_5y":forecast_wi_5y}).dropna(how='all').drop_duplicates()

        ## Scraping Crypto Forecasts from Gov Capital
        identifier_gc=[]
        forecast_gc_14d=[]
        forecast_gc_3m=[]
        forecast_gc_6m=[]
        forecast_gc_1y=[]
        forecast_gc_5y=[]
        #set header reference as consistency check when scraping
        headers_gc= ['Name', '14d Forecast', '3m Forecast', '6m Forecast', '1y Forecast', '5y Forecast']
        for i in range(1,222):
            website ="https://gov.capital/crypto/page/"+str(i)+"/"
            response = requests.get(website)
            soup= BeautifulSoup(response.content, 'html.parser')
            #get headers from scraped page
            link_headers= [x.get_text().strip() for x in soup.find('table', {'class': 'table-striped'}).find('thead').find_all('th')]
            if headers_gc==link_headers:    
                results= soup.find('table', {'class': 'table-striped'}).find('tbody').find_all('tr')
                for result in results:
                    #name
                    try:
                        name,symbol= result.find_all('a')[0].get_text().strip().split("(",maxsplit =1)
                        name= name.strip().lower()
                        symbol= symbol.strip(')')
                        identifier_gc.append(name+'-'+symbol)
                    except:
                        identifier_gc.append(np.nan)
                    #get forecasts
                    try:
                        forecast_14ds, forecast_3ms, forecast_6ms, forecast_1ys, forecast_5ys=[float(x.get_text().strip().strip(" %").strip("+")) for x in result.find_all('a')[1:]]
                        forecast_gc_14d.append(forecast_14ds)
                        forecast_gc_3m.append(forecast_3ms)
                        forecast_gc_6m.append(forecast_6ms)
                        forecast_gc_1y.append(forecast_1ys)
                        forecast_gc_5y.append(forecast_5ys) 
                    except:
                        forecast_gc_14d.append(np.nan)
                        forecast_gc_3m.append(np.nan)
                        forecast_gc_6m.append(np.nan)
                        forecast_gc_1y.append(np.nan)
                        forecast_gc_5y.append(np.nan)
            else:
                pass  
            check =  all(elem in identifier_gc  for elem in select_identifier) #stop scraping when all cryptos selected are found
            if check == True:
                break
        # combine in Dataframe
        gov_capital= pd.DataFrame({'identifier':identifier_gc,"GC_3m":forecast_gc_3m, "GC_1y":forecast_gc_1y,
                                "GC_5y":forecast_gc_5y}).dropna(how='all').drop_duplicates().drop_duplicates(subset='identifier')

        # merge data sources with ready tables: Wallet Investor & Gov Capital
        df1= pd.merge(wallet_investor,gov_capital, how='outer')
        return df1
    
    @st.cache
    def get_indicators(data):
        #get technical indicators from Trading View
        option = webdriver.ChromeOptions()
        option.add_argument('headless')
        driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()),options=option)
        rsi=[]
        macd=[]
        adx=[]
        for symbol in data.symbol:
            url="https://www.tradingview.com/symbols/"+symbol+"USD/technicals/"
            driver.get(url)
            time.sleep(2)
            html = driver.page_source
            soup = BeautifulSoup(html, 'html.parser')
            try:
                results=soup.find_all('table')[:2]
                final=pd.DataFrame(columns=['Name','Value','Action'])
                for table in results:
                    rows=table.find_all('tr')
                    # get headers
                    headers=[x.get_text() for x in rows[0].find_all('th')]
                    #get info
                    records=[]
                    for row in rows[1:]:
                        records.append([x.get_text() for x in row.find_all('td')])
                    df=pd.DataFrame(records,columns=headers)
                    final= pd.concat([final,df])
                rsi.append(float((final[final.Name.isin(['Relative Strength Index (14)'])]['Value']).tolist()[0].replace('−', '-')))
                macd.append(float((final[final.Name.isin(['MACD Level (12, 26)'])]['Value']).tolist()[0].replace('−', '-')))
                adx.append(float((final[final.Name.isin(['Average Directional Index (14)'])]['Value']).tolist()[0].replace('−', '-')))
            except:
                rsi.append(np.nan)
                macd.append(np.nan)
                adx.append(np.nan)
        driver.close()
        data['RSI_14d']= rsi
        data['MACD'] = macd
        data['ADX'] = adx

    @st.cache(allow_output_mutation=True)
    def get_forecasts(data, select_identifier):
        get_forecasts_by_coin(data= data)
        df1 = get_forecast_tables(select_identifier)
        # merge all forecasts
        final =pd.merge(data,df1, how="left")
        # calculate forecasts prices of Wallet Investor and Gov Capital and CryptoPredictions.com from scraped % change
        change_cols=[col for col in final.columns if ("CP" in col) or ("WI" in col) or ("GC" in col)]
        for col in change_cols:
            final[col]= (final[col]/100 +1) * final.price
        return final

    cols_3m_sin=list(set([col for col in forecasts.columns if "3m" in col]+['TB_3m']+['DC_3m']))
    cols_1y_sin=list(set([col for col in forecasts.columns if "1y" in col]+['TB_1y']+['DC_1y']))
    cols_5y_sin=list(set([col for col in forecasts.columns if ("5y" in col) or ("3y" in col) or ("4y" in col)]+['TB_3y']+['DC_5y']))
    cols_summary_sin={'short-term':cols_3m_sin,'medium-term':cols_1y_sin,'long-term':cols_5y_sin}

    # @st.cache
    def forecast_analysis_single(selection_data,full_data, period = 'long-term'):
        #rerun forecast analysis on entire data based on period selected
        forecast_analysis(period= period, data= full_data)
        #calculate min, max, mean, median prices
        selection_data['Min_Price']= selection_data[cols_summary_sin[period]].min(axis=1)
        selection_data['Max_Price']= selection_data[cols_summary_sin[period]].max(axis=1)
        selection_data['Avg_Price']= selection_data[cols_summary_sin[period]].mean(axis=1)
        selection_data['Median_Price']= selection_data[cols_summary_sin[period]].median(axis=1)    
        # calculate % changes with respect to price
        selection_data['Min_Change']= (selection_data['Min_Price'] / selection_data['price'] - 1)
        selection_data['Max_Change']= (selection_data['Max_Price'] / selection_data['price'] - 1) 
        selection_data['Avg_Change']= (selection_data['Avg_Price'] / selection_data['price'] - 1)
        selection_data['Median_Change']= (selection_data['Median_Price'] / selection_data['price'] - 1)
        #calculate top 5% and 10% thresholds from full universe data 
        min_90, max_90, avg_90, median_90 = [x for x in full_data[['Min_Change', 'Max_Change', 'Avg_Change','Median_Change']].quantile(0.9, axis=0)]
        min_95, max_95, avg_95, median_95 = [x for x in full_data[['Min_Change', 'Max_Change', 'Avg_Change','Median_Change']].quantile(0.95, axis=0)]
        #generate trading recommendation for the selected crypto
        selection_data['Action']= selection_data.apply(lambda row: "Strong Buy" if ((row['Min_Change']>= min_95) 
                                         and (row['Max_Change']>= max_95) and (row['Avg_Change']>= avg_95)
                                         and (row['Median_Change']>= median_95)and (row['market_cap'] > 0)) else ("Buy" if ((row['Min_Change']>= min_90) 
                                         and (row['Max_Change']>= max_90) and (row['Avg_Change']>= avg_90)
                                         and (row['Median_Change']>= median_90)and (row['market_cap'] > 0)) else ("Strong Buy - Swappable" if ((row['Min_Change']>= min_95) 
                                         and (row['Max_Change']>= max_95) and (row['Avg_Change']>= avg_95) and (row['Median_Change']>= median_95)and (row['market_cap'] == -1.00))
                                          else ("Buy - Swappable" if ((row['Min_Change']>= min_90) and (row['Max_Change']>= max_90) and (row['Avg_Change']>= avg_90) 
                                          and (row['Median_Change']>= median_90)and (row['market_cap'] == -1.00)) else "Avoid"))), axis=1)
    categories, noth = st.columns(2)
    #create filter for crypto category
    filter_tag= categories.multiselect("Category of Interest:", options= unique_tags, default= None, key= 40,help='Select cryptos in different categories such as metaverse, defi, stable coins, etc...')
    if filter_tag != []:
        #get ids of cryptos meeting tag search
        meta_df= meta_data.copy()
        tags_records=[]
        for i in range(0, len(meta_df)):
            coin_tags= meta_df['Tag-Names'][i].split(",")
            coin_tags=[sub.replace("'", "").strip() for sub in coin_tags]
            tags_records.append(coin_tags)
        tags_df= pd.DataFrame({'Tag-Names': tags_records})
        meta_df = meta_df.drop('Tag-Names', axis=1)
        meta_df= pd.concat([meta_df, tags_df], axis=1)
        match_tag=[]
        for i in range(0, len(meta_df)):
            result= any(elem in meta_df['Tag-Names'][i] for elem in filter_tag)
            match_tag.append(result)
        meta_df['match']=match_tag
        meta_df= meta_df[meta_df['match'] == True]
        matchs_ids= meta_df['Id'].values.tolist()

        filter2_df= forecasts[forecasts.id.isin(matchs_ids)]
    else:
        filter2_df= forecasts

    filter2_df.dropna(subset= ["CP_3m", "WI_3m", "GC_3m"], how= 'all', inplace= True)
    ## create blockchain type filter options
    platforms_df_filter = filter2_df.platform.value_counts().rename_axis('blockchain').reset_index(name='counts')
    top_platforms= ((platforms_df_filter.blockchain[:11])).tolist() # displayed by order
    other_platforms= (platforms_df_filter.blockchain[11:]).tolist()
    if other_platforms != []:
        options= ["All"]+ top_platforms + ["Other"]
    elif len(top_platforms) == 1:
        options= top_platforms
    else:
        options= ["All"]+ top_platforms
    options.sort(reverse= False)
    if "All" in options:
        default = "All"
    else:
        default= options

    form=st.form("filter")
    periods2, platforms2= form.columns(2)
    #select forecast analysis period for calculations
    user_period2 = periods2.selectbox("Select Forecast Analysis Period", options=('short-term','medium-term','long-term'),index=2, help= "Short-term: 3 months - Medium-term: 1 year - Long-term: 3 to 5 years", key = 3)
    #create blockchain type filter
    blockchain2 = platforms2.multiselect("Blockchain Type", options, default= default, help="Filter results based on the crypto's blockchain platform such as ethereum, solana, etc...", key = 4)
    if "All" in blockchain2:
        blockchain2= (filter2_df.platform.unique()).tolist()
    if "Other" in blockchain2:
        blockchain2= other_platforms
    if filter_tag != []:
        value_lower= np.nanmin(filter2_df.market_cap)
    else:
        value_lower= -1.00
    #create filters for market cap & date_added
    cap1_single, cap2_single, date1_single, date2_single= form.columns(4)
    cap_lower2= cap1_single.number_input(label= 'Min. Market Cap', min_value= -1.00, max_value= filter2_df.market_cap.max(), value= value_lower,step= 1000000.00, help="Value of -1 represents cryptos whose market cap is unknown.", key=7)
    cap_upper2= cap2_single.number_input(label= 'Max. Market Cap', min_value= cap_lower2, max_value= filter2_df.market_cap.max(), value= filter2_df.market_cap.max(),step= 1000000.00, key=8)
    date_min2= date1_single.date_input("Min. Date Added", min_value= filter2_df.date_added.min(), max_value= filter2_df.date_added.max(), value= filter2_df.date_added.min())
    date_max2= date2_single.date_input("Max. Date Added", min_value= date_min2, max_value= filter2_df.date_added.max(), value= filter2_df.date_added.max())

    if cap_lower2 == -1.00:
        st.write("Filtering results for cryptos having a market cap up to \${}.".format('{0:,.2f}'.format(cap_upper2)))
    else:
        st.write("Filtering results for cryptos having a market cap ranging from \${} to \${}.".format('{0:,.2f}'.format(cap_lower2), '{0:,.2f}'.format(cap_upper2)))
    #create form submit button:
    form.form_submit_button("Apply filters")
    #filter table based on user input in widgets
    mask2= (forecasts['platform'].isin(blockchain2)) & (forecasts['date_added'] >= date_min2) & (forecasts['date_added'] <= date_max2) & (forecasts['market_cap'] >= cap_lower2) & (forecasts['market_cap'] <= cap_upper2)
    filtered_df2= forecasts.loc[mask2]
    if filter_tag != []:
        filtered_df2= filtered_df2[filtered_df2.id.isin(matchs_ids)]

    names, indicator= st.columns([1.75,1])
    selection = names.multiselect("Click here to select from the list of cryptos meeting the desired criteria:", options= (np.sort(filtered_df2.name.unique())).tolist(), default= None)
    with indicator:
        st.write("")
        st.write("")
        # indicators= st.checkbox(label="Technical Indicators", value = False, help="Include metrics scraped from Trading View on relative strength index - 14 days (RSI), average directional index (ADX), and moving average convergence-divergence (MACD).")
        show_change= st.checkbox("Display price % change calculation columns", key=43)

    submit= st.button("Get forecasts")

    if submit:
        with st.spinner("Please wait... Data scraping in process! Maximum waiting time is around 15 mins."):
            # get CMC ids of selected cryptos
            selected_ids= (filtered_df2[filtered_df2['name'].isin(selection)].id).tolist()
            # get identifier of selected cryptos
            select_identifier= (filtered_df2[filtered_df2['name'].isin(selection)].identifier).tolist()
            listings= get_quotes(select_ids=selected_ids)
            single_forecasts = get_forecasts(data= listings, select_identifier= select_identifier)

            # if indicators:
            #     get_indicators(data= single_forecasts)

            ## perform forecast analysis on selected period
            forecast_analysis_single(period= user_period2,selection_data= single_forecasts, full_data= forecasts)

            #create styler for each numeric column
            styler={'Market_Cap': '{:,.2f}'}
            price_cols= ['Price', 'Min_Price', 'Max_Price', 'Avg_Price','Median_Price'] + [x.title() for x in cols_summary_sin[user_period2]]
            change_cols=['Min_Change', 'Max_Change', 'Avg_Change','Median_Change']
            for col in price_cols:
                styler[col]= '{:,.5f}'
            if show_change:
                figs_to_display2=['name','symbol','platform','date_added','price', 'market_cap', 'data_source','Action','Min_Price', 'Max_Price', 'Avg_Price',
                'Median_Price','Min_Change', 'Max_Change', 'Avg_Change','Median_Change'] + cols_summary_sin[user_period2]
                for col in change_cols:
                    styler[col]= '{:.2%}'
            else:
                figs_to_display2=['name','symbol','platform','date_added','price', 'market_cap', 'data_source','Action','Min_Price', 'Max_Price', 'Avg_Price',
                'Median_Price'] + cols_summary_sin[user_period2]
            # if indicators:
            #     figs_to_display2= figs_to_display2 + ['RSI_14d','MACD','ADX']
            #     indicators_cols=['RSI_14d','MACD','ADX']
            #     for col in indicators_cols:
            #         styler[col]= '{:,.2f}'

            final_single= single_forecasts[figs_to_display2]
            final_single.rename(columns={'platform': 'blockchain'}, inplace=True)
            final_single.columns= final_single.columns.str.title()
            # if indicators:
            #     final_single.rename(columns={'Rsi_14D': 'RSI_14d', 'Macd':'MACD', 'Adx':'ADX'}, inplace= True)
            legend= """<i><text style="text-align:left; color:gray;font-size:13px;">Forecasts References: Cp- CryptoPredictions.com | Wi - Wallet Investor | Gc - Gov Capital | Dc - Digital Coin Price | Tb - Trading Beasts
                </text></i>
                """
            st.markdown(legend,unsafe_allow_html=True)
            st.write(final_single.set_index('Name').style.format(styler))

### WALLET TRACKER
if menu_id == "Wallet Tracker":
    st.header("What have you earned so far in each of your investments?")
    st.markdown("""This wallet tracker is powered by CoinGecko & Gate.io API.  
                   To track your portfolio's profit/loss by crypto, download the below template and fill the amount invested and received in each crypto.  
                   Search the list by both name and symbol as in some cases, one of the fields might not be available.  
                   DO NOT EDIT ANY OF THE ELEMENTS UNDER "ID", "SYMBOL", and "NAME".
                   """)

    cg = CoinGeckoAPI()
    api_client = gate_api.ApiClient()
    # Create an instance of the API class
    api_instance = gate_api.SpotApi(api_client)

    # prepare template for portofolio tracker
    @st.cache
    def prepare_template():
        # get coins list from CoinGecko
        coins_list= cg.get_coins_list()
        coins_list= pd.DataFrame(coins_list)
        coins_list.rename(columns={'id':'ID', 'symbol':"Symbol", "name": "Name"},inplace=True)
        coins_list['USD Investment']= np.nan
        coins_list['Received']=np.nan
        coins_list['Symbol']= coins_list['Symbol'].str.upper()
        coins_list["Source"]= "CoinGecko"
        matches=[]
        for i in range(0, len(coins_list)):
            match= re.findall(r'\d*\.*\d[Xx]', coins_list['Name'][i])
            if match == []:
                matches.append("ok")
            else:
                matches.append("remove")        
        coins_list['check']=matches
        coins_list.drop(coins_list[coins_list['check'] == "remove"].index, inplace=True)
        # # get coins list from Gate.io exchange
        # try:
        #     # List all currency pairs on Gate.io
        #     api_response = api_instance.list_currency_pairs()
        #     base=[]
        #     for i in range(0, len(api_response)):
        #         if api_response[i].quote == "USDT":
        #             base.append(api_response[i].base)
        #     gate_list= pd.DataFrame({"Symbol": base})
        #     gate_list['Source'] = "Gate.io"
        #     # save only the symbols not available on CoinGecko
        #     gate_list = gate_list[gate_list.Symbol.isin(coins_list.Symbol) == False]
        #     #get coins names through scraping
        #     option = webdriver.ChromeOptions()
        #     option.add_argument('headless')
        #     driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()),options=option)
        #     url= "https://www.gate.io/marketlist?tab=usdt"
        #     driver.get(url)
        #     time.sleep(1)
        #     html = driver.page_source
        #     # response= requests.get(url)
        #     soup= BeautifulSoup(html, 'html.parser')
        #     rows= soup.find("table").find("tbody").find_all("tr")
        #     symbols=[]
        #     names=[]
        #     for row in rows:
        #         symbol= row.find_all("td")[0].find("span",{"class": "curr_a"}).get_text().strip()
        #         name= row.find_all("td")[0].find("span",{"class": "cname"}).get_text().strip()
        #         symbols.append(symbol)
        #         names.append(name)
        #     gate_names= pd.DataFrame({"Name": names,"Symbol": symbols})
        #     gate_list=pd.merge(gate_list,gate_names,how="left")
        #     gate_list.dropna(subset=['Name'], inplace=True)
        #     gate_list.reset_index(drop=True, inplace=True)
        #     gate_list['Name']= gate_list.Name.apply(lambda x: re.sub(r'\b\d+X\s', "",x))
        #     gate_list['ID']= gate_list['Name'].str.lower() + "-" + gate_list['Symbol'].str.lower()
        #     gate_list.drop_duplicates(inplace=True)
        #     driver.quit()
        # except GateApiException as ex:
        #     print("Gate api exception, label: %s, message: %s\n" % (ex.label, ex.message))
        # except ApiException as e:
        #     print("Exception when calling DeliveryApi->list_delivery_contracts: %s\n" % e)
        # # combine data from both sources
        # all_coins= pd.concat([coins_list,gate_list])
        all_coins= coins_list
        return all_coins

    with st.spinner("Please wait... Preparing portfolio template..."):
        all_coins = prepare_template()

    #create data download button
    @st.cache
    def download_template(df):
        df= df[["Source","ID","Name", "Symbol", "USD Investment", "Received"]].sort_values(by="Name", ascending= True)
        return df.to_csv(index=False).encode('utf-8')
    template= download_template(all_coins)
    st.download_button(label="Download Template", data=template, file_name='portfolio_template.csv', mime='text/csv')
    uploader, empt= st.columns(2)
    uploaded_file = uploader.file_uploader("Upload portfolio summary here:", type= 'csv')

    if uploaded_file is not None:
        portfolio= pd.read_csv(uploaded_file)
        # drop coins not invested in
        portfolio.dropna(subset=['Received'], inplace=True)
        portfolio = portfolio.groupby(by=["Name", "Symbol", "ID","Source"], as_index =False).sum()
        # split portfolio by source to get current prices
        gecko= portfolio[portfolio.Source == "CoinGecko"]
        gateio= portfolio[portfolio.Source == "Gate.io"]
        if gecko.empty == False:
            # get current prices of portfolio cryptos
            live_prices= cg.get_price(ids=list(gecko.ID), vs_currencies='usd')
            # extract current price from current data
            current_price=[]
            for price in list(live_prices.values()):
                current_price.append(price['usd'])
            # add current price to portfolio table
            gecko['Current Price']= current_price
        if gateio.empty == False:
            try:
                # List all ticker prices
                api_response = api_instance.list_tickers()
                curreny_pair=[]
                last_price=[]
                for ticker in api_response:
                    curreny_pair.append(ticker.currency_pair)
                    last_price.append(float(ticker.last))
                gate_tickers = pd.DataFrame({"pair": curreny_pair, "price":last_price})
                # gate_tickers= gate_tickers[gate_tickers.pair.str.contains("_USDT")]
                price=[]
                for coin in gateio.Symbol:
                    coin_price = float(gate_tickers[gate_tickers.pair == (coin + "_USDT")].price)
                    price.append(coin_price)
                gateio["Current Price"]= price
            except GateApiException as ex:
                print("Gate api exception, label: %s, message: %s\n" % (ex.label, ex.message))
            except ApiException as e:
                print("Exception when calling DeliveryApi->list_delivery_contracts: %s\n" % e)
        # recombine 2 tables into 1 portfolio
        portfolio= pd.concat([gecko, gateio])

        # calculate current investment value
        portfolio['Current Value']= portfolio['Current Price'] * portfolio['Received']
        # calculate profit or loss
        portfolio['Profit (Loss)']= portfolio['Current Value'] - portfolio['USD Investment']
        # add icon
        sign =[]
        for x in portfolio['Profit (Loss)']:
            if x >= 0:
                sign.append(emoji.emojize(":sparkle:", language='alias'))
            else:
                sign.append(emoji.emojize(":heavy_exclamation_mark:", language='alias'))
        portfolio[" "]= sign
        portfolio['USD Investment']= portfolio['USD Investment'].fillna(0)
        #display results
        results = portfolio[["Symbol",'Name', 'USD Investment','Received', "Current Price","Current Value", "Profit (Loss)", 
                    " "]].sort_values(by="Profit (Loss)", ascending= False)
        results['Received']=results['Received'].apply(lambda x: '{0:,.2f}'.format(x))
        results['USD Investment']=results['USD Investment'].apply(lambda x: '{0:,.2f}'.format(x))
        results['Current Price']=results['Current Price'].apply(lambda x: '{0:,.6f}'.format(x))
        results['Current Value']=results['Current Value'].apply(lambda x: '{0:,.3f}'.format(x))
        results['Profit (Loss)']=results['Profit (Loss)'].apply(lambda x: '{0:,.3f}'.format(x))
        #convert table to html for formatting
        html_table= results.to_html(border=0,index=False,justify = 'center',escape=False)
        html_table= html_table.replace("<thead>", '<thead style="background-color: rgb(255,204,203,0.5)">')
        html_table= html_table.replace("<tr>", '<tr align="center" style="background-color: mintcream">')
        html_table= html_table.replace("<td>", '<td style="border-left:0px; border-right:0px">')
        html_table= html_table.replace("<th>", '<th style="border-left:0px; border-right:0px">')
        round_style= """
        <html>
        <head>
            <style>
            th{
            border-radius: 0px;
            }
            </style>
            </head>
            <body>"""
        html_table= round_style +html_table +"</body></html>"
        st.write(HTML(html_table))
        #create data download button
        @st.cache
        def download_tracker(df):
            df= df.drop([" ", "ID"], axis= 1)
            return df.to_csv(index=False).encode('utf-8')
        tracker= download_tracker(portfolio)
        st.write("")
        investment, value, col1, col4= st.columns(4)

        value_metric= "$"+"{0:,.2f}".format(np.nansum(portfolio['Current Value']))
        investment_metric= "$"+"{0:,.2f}".format(np.nansum(portfolio['USD Investment']))
        with investment:
            st.metric("Total Investment", value=investment_metric)
        with value:
            st.metric("Portfolio Value", value=value_metric, delta= "{0:,.2f}".format(np.nansum(portfolio['Profit (Loss)'])))
        st.write("")
        st.download_button(label="Export as CSV", data=tracker, file_name='wallet_tracker.csv', mime='text/csv')

if menu_id == "Methodology":
    st.subheader("Forecast Analysis")
    with st.expander("Data Sources"):
        st.write("")
        st.markdown("""<h6><u>Actual Price Data</u></h6>
        <p>
            The collection of cryptocurrencies to be analyzed was obtained through <a href="https://coinmarketcap.com/api/">CoinMarketCap</a> API as a result of merging two datasets:
            <li>
            -	Top 5,000 cryptos based on market capitalization<br>
            -	Top 5,000 cryptos based on date added, i.e., youngest coins in the market that might hold the greatest potential for investment
            </li>
        </p>
            <p>The data included information on each currency’s blockchain type, price, market capitalization, volume, and price changes over several timeframes.
            </p><p>
            Note that CoinMarketCap only reports data under the market capitalization field when the currency’s circulating supply has been verified by their team through due diligence processes initiated with the currency’s project team. 
            The reason for which the API provides self-reported market cap and circulating supply data. We have used the latter when verified information was not available. 
            Cases where market cap is unknown reprerent cryptos available for purchase only through swaps and were assigned a market cap value of -1.
        </p>
        <h6><u>Forecasts Data</u></h6>
        <p>
            Information on forecasted prices was collected through web scraping the following sources:
            <li>
            - <a href="https://walletinvestor.com/forecast?page=1&per-page=100">Wallet Investor</a><br>
            - <a href="https://gov.capital/crypto/">Gov Capital</a><br>
            - <a href="https://cryptopredictions.com/">CryptoPredictions.com</a><br>
            - <a href="https://digitalcoinprice.com/forecast">Digital Coin Price</a> (results may not be always available due to Captcha block from source)<br>
            - <a href="https://tradingbeasts.com/crypto/">Trading Beasts</a> (only available in Real Time Forecasts section as full data running process is very slow)
            </li>
        </p>
        <p>
            The forecasts available cover a period of 3 months (short-term), 1 year (medium-term), and 3 to 5 years (long-term).
        """,
        unsafe_allow_html=True)
    with st.expander("Trading Recommendation System"):
        st.write("")
        st.markdown("""
        <p>
        The trading recommendation system applied by the application is based on comparative analysis between both the forecasts available and with the current price.<br>
        The analysis performed between the forecasts available per crypto considers all scenarios of the forecasted price whereby we calculate the minimum, maximum, average and median forecasted price among the forecasts sources used for each crypto for each period (3 months, 1 year, and 5 years).
        The expected percentage change in price is calculated per each of the metrics listed compared to the current market price.<br>
        </p>
        <p>
        The following trading recommendations are then determined based on set rules:
        <li>
            - <b> Strong Buy </b>:<br>
            This recommendation is generated for a cryptocurrency whose expected percentage changes in price fall in the <b>top 5%</b> of the expected price changes of the entire universe under study for all four calculated metrics.<br>
            - <b> Buy</b>:  <br>
            This recommendation is generated for a cryptocurrency whose expected percentage changes in price fall in the <b>top 10%</b> of the expected price changes of the entire universe under study for all four calculated metrics.<br>
            - <b> Avoid</b>: <br>
            This recommendation is generated for a cryptocurrency that does not match the above conditions in all expected percentage changes scenarios.
        </li>
        </p>
        <p>
        The system can tailor the recommendations explained above to satisfy the interests of short, medium, or long term investors.
        </p>
        <p>
        Note that in the cases where market cap is unknown, i.e. the crypto is classified as purchasable through swaps only, a sub-level to the “Strong Buy” and “Buy” recommendations is available to distinguish 
        these cryptos whose expected price changes are in the top 5% or 10% of the universe but their market cap is unknown with two types of recommendations: “Strong Buy – Swappable” or “Buy – Swappable”.
        """,
        unsafe_allow_html=True)
    
    st.subheader("Wallet Tracker")
    with st.expander("Data Sources"):
        st.write("")
        st.markdown("""
        <p>
        The portfolio template is compiled from the lists of cryptocurrenices obtained from the APIs of <a href="https://www.coingecko.com/en/api/documentation">Coin Gecko </a>and 
        <a href="https://www.gate.io/docs/developers/apiv4/en/">Gate.io</a>.</p>
        """,
        unsafe_allow_html=True)
    with st.expander("Portfolio Analysis"):
        st.write("")
        st.markdown("""
        <p>
        The wallet tracker we developed relies on the following assumptions:
        <li>
            - The user has purchased his cryptocurrencies with fiat currency (USD) or USDT<br>
            - 1 USD = 1 USDT<br>
        </li>
        </p>
        <p>
        Based on the user input in the uploaded portfolio template, the tracker obtains the current prices of the select cryptocurrencies from Coin Gecko and Gate.io APIs.<br>
        The current value in USD of the investment made is calculated as current price x amount received (in crypto).<br>
        The final profit/loss on investment is implied by comparing the initial USD investment with the current value.</p>
        <p>
        Note that the portfolio template accepts data entries with missing USD investment.<br>
        In case of duplicate trades, the wallet tracker calculates the sum of investments and amounts received in a cryptocurrency.<br>
        In addition, the "ID" field in the template is unique and can be used by the user to update the portfolio template downloaded from the app in excel as new cryptos are added to the list.
        """,
        unsafe_allow_html=True)

