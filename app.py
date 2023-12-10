import streamlit as st
from gensim.summarization import summarize
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lex_rank import LexRankSummarizer
import pandas as pd
import warnings
warnings.filterwarnings('ignore')
import matplotlib.pyplot as plt
#mport seaborn as sns
import matplotlib
matplotlib.use('Agg')
from rouge import Rouge
import altair as alt



def sumy_summarizer(docx,num = 2):
    parser = PlaintextParser.from_string(docx,Tokenizer("english"))
    lex_summerizer = LexRankSummarizer()
    summary = lex_summerizer(parser.document,num)
    summary_list = [str(sentence) for sentence in summary]
    result = ' '.join(summary_list)
    return result

#Evaluate Summary

def evaluate_summary(summary,reference):
    r = Rouge()
    eval_score = r.get_scores(summary,reference)
    eval_score_df = pd.DataFrame(eval_score[0])
    return eval_score_df




def main():

    st.title('Summerizer App')
    menu= ['Home',"About"]
    choice = st.sidebar.selectbox("Menu",menu)
    
    if choice =="Home":
        st.subheader('Summarzation')
        raw_text = st.text_area('Enter Text Here')
        button = st.button('Summarize')
        
        if button==True:
            with st.expander('Original Text'):

               st.write(raw_text)
            
            c1,c2 =st.columns(2)
            col_height = 500
            with c1:
               # st.markdown(f'<div style="height: {col_height}px;"></div>', unsafe_allow_html=True)
                with st.expander('LexRank Summury'):
                     
                     my_summary = sumy_summarizer(raw_text)
                     document_len = {'Original': len(raw_text), 'Summary': len(my_summary)}

                     st.write(document_len)
                     st.write(my_summary)
                     st.info("Rouge Score")
                     eval_df= evaluate_summary(my_summary,raw_text)
                     st.dataframe(eval_df.T)
                     eval_df['metrics'] = eval_df.index
                     c22 = alt.Chart(eval_df).mark_bar().encode(x='metrics',y='rouge-1')
                      
                     c22 = c22.properties(width=300,
                                          height = 200) 
                     st.altair_chart(c22)
                        
                    

                    
            with c2:
                    #st.markdown(f'<div style="height: {col_height}px;"></div>', unsafe_allow_html=True)
                    with st.expander('TextRanker Summury'):
                      my_summary = summarize(raw_text)
                      document_len = {'Original': len(raw_text), 'Summary': len(my_summary)}
                      st.write(document_len)
                      st.write(my_summary)
                      #st.write(document_len)
                      st.markdown("<br>", unsafe_allow_html=True)
                      st.info('Rouge Score')
                      eval_df = evaluate_summary(my_summary,raw_text)
                      #document_len = {'Original': len(raw_text), 'Summary': len(my_summary)}

                      
                      #st.write(my_summary)
                    #   st.info("Rouge Score")
                      #eval_df= evaluate_summary(my_summary,raw_text)
                      st.dataframe(eval_df.T)
                      eval_df['metrics'] = eval_df.index
                     
                      c11 = alt.Chart(eval_df).mark_bar().encode(x='metrics',y='rouge-1')
                      
                      c11 = c11.properties(width=300,height = 220) 
                      st.altair_chart(c11)
                        



if __name__=='__main__':
    main()

