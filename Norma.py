# -*- coding: utf-8 -*-
"""
Created on Thu Mar 10 19:21:07 2022

@author: Kasi
"""

import scipy
import streamlit as st  
import pandas as pd  
import plotly.express as px 
import base64  
from io import StringIO, BytesIO  
from scipy.stats import shapiro
from scipy.stats import kstest
from scipy.stats import anderson
from scipy.stats import normaltest
import statsmodels
from scipy import stats
from scipy.stats import norm
import matplotlib.pyplot as plt
from statsmodels.stats.diagnostic import lilliefors
import numpy as np
import statistics
import statsmodels.api as sm
import pylab as py


st.set_page_config(page_title='Norma, The Normality Assessment Tool')
st.title('Norma 📈')
st.subheader('Please select the Excel file')

    # -- File Uploader

uploaded_file = st.file_uploader('Please select the Excel file', type='xlsx')
if uploaded_file:

    df = pd.read_excel(uploaded_file, engine='openpyxl')
    df_t = df.select_dtypes(include= 'object')
    df_c = df.drop([col for col in df.columns if col in df.columns and col in df_t.columns], axis=1)

    
    # -- Selecting Variable
    
    list1 = df_c.columns.values.tolist()
    
    button1 = st.radio("Which variable do you want to analyze?", (list1))
    
    new_df = df_c[button1].dropna()
    
    st.markdown("""---""")
    
    # -- Centeral Tendency Indices
    
    std = statistics.stdev(new_df)
    std = "{:.3f}".format(std)
    std = float(std)
    
    mu = statistics.mean(new_df)     
    mu = "{:.3f}".format(mu)    
    mu = float(mu)    
    
    st.write('Mean:', mu,
             'Standard Deviation:', std,
             'Sample Size:', len(new_df))
    
    st.markdown("""---""")
    
    # -- Alpha
    
    alpha = st.number_input('Insert value of the alpha:', min_value=0.00, max_value=1.00, value=0.05, help='Between 0 and 1')

    st.markdown("""---""")    

    # -- Normality Tests
    
    st.subheader('Results of Tests of Normality:')
   
    col1, col2 = st.columns(2)
    
    with col1:
        
        ssw, psw = shapiro(new_df)
        ssw = "{:.3f}".format(ssw)
        psw = "{:.5f}".format(psw)
        ssw = float(ssw)
        psw = float(psw)
        st.write('🔷 Shapiro-Wilk:')
        st.write('statistic:', ssw)
        st.write('p-value:', psw)
    
        if psw > alpha:
            st.write('✅ Sample looks NORMAL (failed to reject H0)')
        else:
            st.write('❌ Sample does NOT look Normal (rejected H0)')
        st.write('')
        st.write('')  
    
    
        sks, pks = kstest(new_df, 'norm')
        sks = "{:.3f}".format(sks)
        pks = "{:.5f}".format(pks)
        sks = float(sks)
        pks = float(pks)
        st.write('🔷 Kolmogorov-Smirnov:')
        st.write('statistic:', sks)
        st.write('p-value:', pks)

        if pks > alpha:
            st.write('✅ Sample looks NORMAL (failed to reject H0)')
        else:
            st.write('❌ Sample does NOT look Normal (rejected H0)') 
        st.write('')
        st.write('')      
        
    
        sda, pda = normaltest(new_df)  
        sda = "{:.3f}".format(sda)
        pda = "{:.5f}".format(pda)
        sda = float(sda)
        pda = float(pda)
        st.write('🔷 D’Agostino’s K^2:')
        st.write('statistic:', sda)
        st.write('p-value:', pda)    
        if pda > alpha:
            st.write('✅ Sample looks NORMAL (failed to reject H0)')
        else:
            st.write('❌ Sample does NOT look Normal (rejected H0)') 
        st.write('')
        st.write('') 


        res = stats.cramervonmises(new_df, 'norm')
        scvm, pcvm = res.statistic, res.pvalue
        scvm = "{:.3f}".format(scvm)
        pcvm = "{:.5f}".format(pcvm)
        scvm = float(scvm)
        pcvm = float(pcvm)
        st.write('🔷 Cramér–von Mises:')
        st.write('statistic:', scvm)
        st.write('p-value:', pcvm)    
        if pcvm > alpha:
            st.write('✅ Sample looks NORMAL (failed to reject H0)')
        else:
            st.write('❌ Sample does NOT look Normal (rejected H0)') 
        st.write('')
        st.write('') 
        
        

    with col2:
        
        st.write('🔷 Anderson-Darling:')
        result = anderson(new_df)
        sad = result.statistic
        sad = "{:.3f}".format(sad)
        sad = float(sad)
        st.write('statistic:', sad)
        p = 0
        for i in range(len(result.critical_values)):
            sl, cv = result.significance_level[i], result.critical_values[i]
            sl = str(sl)
            cv = "{:.5f}".format(cv)
            cv = float(cv)
            if result.statistic < result.critical_values[i]:
                #sl = str(sl)
                st.write('%', sl,':', cv, '✅ Sample looks NORMAL (failed to reject H0)')
            else:
                st.write('%', sl,':', cv, '❌ Sample does NOT look Normal (rejected H0)')
        st.write('')
        st.write('') 

        
        sll, pll = (statsmodels.stats.diagnostic.lilliefors(new_df, dist='norm', pvalmethod='table'))
        sll = "{:.3f}".format(sll)
        pll = "{:.5f}".format(pll)
        sll = float(sll)
        pll = float(pll)
        st.write('🔷 Lilliefors:')
        st.write('statistic:', sll)
        st.write('p-value:', pll)    
        if pll > alpha:
            st.write('✅ Sample looks NORMAL (failed to reject H0)')
        else:
            st.write('❌ Sample does NOT look Normal (rejected H0)') 
        st.write('')
        st.write('')                 


        sjb, pjb = stats.jarque_bera(new_df)
        sjb = "{:.3f}".format(sjb)
        pjb = "{:.5f}".format(pjb)
        sjb = float(sjb)
        pjb = float(pjb)
        st.write('🔷 Jarque-Bera:')
        st.write('statistic:', sjb)
        st.write('p-value:', pjb)    
        if pjb > alpha:
            st.write('✅ Sample looks NORMAL (failed to reject H0)')
        else:
            st.write('❌ Sample does NOT look Normal (rejected H0)') 
        st.write('')
        st.write('') 
        
    st.markdown("""---""")
             
    # -- Skewness and Kurtosis Indices    
        
    st.subheader('Skewness and Kurtosis Indices:')
    std = statistics.stdev(new_df) 
    mu = statistics.mean(new_df) 
    std = "{:.3f}".format(std)
    mu = "{:.3f}".format(mu)
    std = float(std)
    mu = float(mu)    
    st.write('Mean:', mu,
             'Standard Deviation:', std,
             'Sample Size:', len(new_df))
        
    std = statistics.stdev(new_df)    
    skew = scipy.stats.skew(new_df)
    kurt = scipy.stats.kurtosis(new_df)
    
    s1 = abs(skew/std)
    k1 = abs(kurt/std)
    
    s1 = "{:.3f}".format(s1)
    k1 = "{:.3f}".format(k1)
    s1 = float(s1)
    k1 = float(k1)  
    
    st.write('|Skewness/Standard Deviation| =', s1)
    if s1 < 2:
        st.write('✅ Sample looks NORMAL (failed to reject H0)')
    else:
        st.write('❌ Sample does NOT look Normal (rejected H0)') 
    st.write('')
    st.write('')
    
    st.write('|kurtosis/Standard Deviation| =', k1)
    if k1 < 2:
        st.write('✅ Sample looks NORMAL (failed to reject H0)')
    else:
        st.write('❌ Sample does NOT look Normal (rejected H0)') 
    st.write('')
    st.write('')
    
    st.markdown("""---""")
    
    # -- Graphs
        
    st.subheader('Graphs:')
    
    bins = st.number_input('Insert the number of groups for the histogram:', min_value=0, max_value=100, value=10, help='A non negative number')

    # -- Histogram
    
    fig, ax = plt.subplots()
    ax.hist(new_df, bins=bins, density=True)
    plt.show() 
    
    x = np.linspace(mu - 3*std, mu + 3*std, 100)
    ncur = plt.plot(x, stats.norm.pdf(x, mu, std))
    plt.show()
    st.subheader('Histogram:')
    st.pyplot(fig, ncur)
    
    st.markdown("""---""")

    # -- QQ Plot

    qq = sm.qqplot(new_df, line='45', fit=True, dist=stats.norm)
    plt.show()
    st.subheader('Q-Q Plot:')
    st.pyplot(qq)
    
    st.markdown("""---""")
    
    # -- Box Plot
    
    box = plt.figure(figsize =(10, 7))
    plt.boxplot(new_df)
    plt.show() 
    st.subheader('Box Plot:')
    st.pyplot(box)
    
    st.markdown("""---""")
    
    # -- End
    
    st.write('Any problems with Norma? Feel free to contact me at: kasra.j1218  [at]  gmail  [dot]  com')
    
    url = 'https://www.linkedin.com/in/kasra-jafari-1b127a132/'
    st.write('Made by: [Kasra Jafari](%s)' % url)
