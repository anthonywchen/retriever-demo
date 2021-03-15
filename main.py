import json
import copy

import pandas as pd
import streamlit as st
import torch
from transformers import DPRContextEncoder, DPRQuestionEncoder, BertTokenizer


@st.cache()
def load_retriever():
    ctx_encoder = DPRContextEncoder.from_pretrained('facebook/dpr-ctx_encoder-single-nq-base')
    ques_encoder = DPRQuestionEncoder.from_pretrained('facebook/dpr-question_encoder-single-nq-base')
    return ctx_encoder, ques_encoder


def load_tokenizer():
    return BertTokenizer.from_pretrained('bert-base-uncased')


@st.cache(suppress_st_warning=True)
def load_dataset():
    dataset = json.load(open('retrieval_dpr_50.json'))
    # Only get the top 100 questions to save cache space
    dataset = {dataset[k]['question']: dataset[k]
               for k in list(dataset.keys())[:100]}

    return dataset


@st.cache()
def get_passage_score(title, passage, question):
    ctx_encoder, ques_encoder = load_retriever()
    tokenizer = load_tokenizer()

    ctx_inputs = tokenizer(title, passage, return_tensors='pt')
    ques_inputs = tokenizer(question, return_tensors='pt')

    ctx_embed = ctx_encoder(**ctx_inputs).pooler_output.squeeze()
    ques_embed = ques_encoder(**ques_inputs).pooler_output.squeeze()
    score = torch.dot(ctx_embed, ques_embed).item()
    return round(score, 4)


def main():
    st.set_page_config(layout="wide")
    st.title('DPR Retrieval Demo')
    dataset = load_dataset()

    st.sidebar.write('Demo uses DPR single-nq-base models')
    question = st.sidebar.selectbox('Pick a question', list(dataset.keys()))
    st.sidebar.write(question)

    left_col, right_col = st.beta_columns(2)
    # Print retrieved documents
    left_col.write('Retrieve Wikipedia Passages')
    num_retr_docs = left_col.slider('Number of passages to retrieve', 10, 50)

    # Add scores for the documents
    retr_docs = copy.deepcopy(dataset[question]['contexts'][:num_retr_docs])
    for context_dict in retr_docs:
        title, passage = context_dict['text'].split('\n')
        title = title.replace('"', '')
        passage = passage.replace('""', '"')
        context_dict['title'] = title
        context_dict['passage'] = passage
        context_dict['score'] = get_passage_score(title, passage, question)
        del context_dict['docid']
        del context_dict['text']

    retr_docs = sorted(retr_docs, key=lambda k: k['score'], reverse=True)
    left_col.table(pd.DataFrame.from_dict(retr_docs))

    # Custom inputs
    right_col.write('Get score of a custom title and passage:')
    title = right_col.text_input('Put in a passage title')
    passage = right_col.text_input('Put in a passage')
    score = get_passage_score(title, passage, question)
    right_col.write('Score: ' + str(score))


if __name__ == "__main__":
    main()