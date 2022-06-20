from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
# for importing and manuplating data
import pickle
import numpy as np
import pandas as pd
import string
import csv
# for fitting model
# from sklearn.feature_extraction.text import CountVectorizer
# from sklearn.tree import DecisionTreeClassifier
# from sklearn.feature_extraction.text import TfidfTransformer, TfidfVectorizer
# from sklearn.pipeline import Pipeline
from transformers import AutoTokenizer, AutoModelForQuestionAnswering
import torch
# connecting with google, request data and apllying nlp
import requests
from lxml import html
from bs4 import BeautifulSoup
import string
import urllib.request
from urllib.request import urlopen
import re
import spacy
# from spacy.lang.en.stop_words import STOP_WORDS
# from sklearn.feature_extraction.text import CountVectorizer
# from sklearn.tree import DecisionTreeClassifier
# from sklearn.feature_extraction.text import TfidfTransformer, TfidfVectorizer
# from sklearn.pipeline import Pipeline
from bs4 import BeautifulSoup, NavigableString, Tag
import requests
import torch
from torch.jit import script, trace
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
from googlesearch import search
#import search
import csv
import random
import re
import os
import unicodedata
import codecs
from io import open
import itertools
import math
from django.shortcuts import render
from rest_framework.views import APIView
import json
from django.http import JsonResponse
from rest_framework import status
import spacy
from scispacy.abbreviation import AbbreviationDetector
import numpy as np
from symspellpy.symspellpy import SymSpell
from rest_framework.decorators import api_view
from rest_framework.response import Response


def cleaner(x):
    return [a for a in (''.join([a for a in x if a not in string.punctuation])).lower().split()]


def createResponse(msg_type, answer, url1, url2, price):
    response = []
    response.append(msg_type)
    response.append(answer)
    response.append(url1)
    response.append(url2)
    response.append(price)
    return response


@api_view(['POST'])
def interact(request):
    input_query = request.data['message']
    response = []

    USE_CUDA = torch.cuda.is_available()
    device = torch.device("cuda" if USE_CUDA else "cpu")

    def listToString(s):
        str1 = ""
        for ele in s:
            str1 += ele + " "
        return str1

    def cleaner(x):
        return [a for a in (''.join([a for a in x if a not in string.punctuation])).lower().split()]

    Pipe = pickle.load(open("model_v1.pk", "rb"))

    def spacy_process_i(text):
        doc = nlp(text)
        lemma_list = []
        for token in doc:
            lemma_list.append(token.lemma_)
        filtered_sentence = []
        for word in lemma_list:
            lexeme = nlp.vocab[word]
            if lexeme.is_stop == False:
                filtered_sentence.append(word)

        punctuations = "?:!.,;"
        for word in filtered_sentence:
            if word in punctuations:
                filtered_sentence.remove(word)

        counter_string = listToString(filtered_sentence)
        pp_string = counter_string.replace("-PRON-", "")

        return pp_string

    MAX_LENGTH = 20  # Maximum sentence length
    PAD_token = 0  # Used for padding short sentences
    SOS_token = 1  # Start-of-sentence token
    EOS_token = 2  # End-of-sentence token
    scripted_searcher = torch.jit.load('scripted_chatbot.pth')

    class Voc:
        def __init__(self, name):
            self.name = name
            self.trimmed = False
            self.word2index = {}
            self.word2count = {}
            self.index2word = {PAD_token: "PAD",
                               SOS_token: "SOS", EOS_token: "EOS"}
            self.num_words = 3  # Count SOS, EOS, PAD

        def addSentence(self, sentence):
            for word in sentence.split(' '):
                self.addWord(word)

        def addWord(self, word):
            if word not in self.word2index:
                self.word2index[word] = self.num_words
                self.word2count[word] = 1
                self.index2word[self.num_words] = word
                self.num_words += 1
            else:
                self.word2count[word] += 1

        # Remove words below a certain count threshold
        def trim(self, min_count):
            if self.trimmed:
                return
            self.trimmed = True
            keep_words = []
            for k, v in self.word2count.items():
                if v >= min_count:
                    keep_words.append(k)

            print('keep_words {} / {} = {:.4f}'.format(
                len(keep_words), len(self.word2index), len(
                    keep_words) / len(self.word2index)
            ))
            # Reinitialize dictionaries
            self.word2index = {}
            self.word2count = {}
            self.index2word = {PAD_token: "PAD",
                               SOS_token: "SOS", EOS_token: "EOS"}
            self.num_words = 3  # Count default tokens
            for word in keep_words:
                self.addWord(word)

    # Lowercase and remove non-letter characters

    def normalizeString(s):
        s = s.lower()
        s = re.sub(r"([.!?])", r" \1", s)
        s = re.sub(r"[^a-zA-Z.!?]+", r" ", s)
        return s

    # Takes string sentence, returns sentence of word indexes

    def indexesFromSentence(voc, sentence):
        try:
            print("one")
            return [voc.word2index[word] for word in sentence.split(' ')] + [EOS_token]
        except:
            print("two")
            return None

    def evaluate(searcher, voc, sentence, max_length=MAX_LENGTH):
        # Format input sentence as a batch
        # words -> indexes
        indexes_batch = [indexesFromSentence(voc, sentence)]
        # Create lengths tensor
        print(indexes_batch)
        lengths = torch.tensor([len(indexes) for indexes in indexes_batch])
        # Transpose dimensions of batch to match models' expectations
        input_batch = torch.LongTensor(indexes_batch).transpose(0, 1)
        # Use appropriate device
        input_batch = input_batch.to(device)
        lengths = lengths.to(device)
        # Decode sentence with searcher
        tokens, scores = searcher(input_batch, lengths, max_length)
        # indexes -> words
        decoded_words = [voc.index2word[token.item()] for token in tokens]
        return decoded_words

    # Evaluate inputs from user input (stdin)

    def evaluateInput(searcher, voc):
        input_sentence = ''
        while(1):
            try:
                # Get input sentence
                input_sentence = input('> ')
                # Check if it is quit case
                if input_sentence == 'q' or input_sentence == 'quit':
                    break
                # Normalize sentence
                input_sentence = normalizeString(input_sentence)
                # Evaluate sentence
                output_words = evaluate(searcher, voc, input_sentence)
                # Format and print response sentence
                output_words[:] = [x for x in output_words if not (
                    x == 'EOS' or x == 'PAD')]
                print('Bot:', ' '.join(output_words))

            except KeyError:
                print("Error: Encountered unknown word.")

    # Normalize input sentence and call evaluate()
    def evaluateExample(sentence, searcher, voc):
        print("> " + sentence)
        # Normalize sentence
        input_sentence = normalizeString(sentence)
        # Evaluate sentence
        output_words = evaluate(searcher, voc, input_sentence)
        output_words[:] = [x for x in output_words if not (
            x == 'EOS' or x == 'PAD')]
        print('Bot:', ' '.join(output_words))
    corpus_name = "cornell movie-dialogs corpus"
    model_name = 'cb_model'
    attn_model = 'dot'
    checkpoint_iter = 4000
    loadFilename = '4000_checkpoint.tar'
    checkpoint = torch.load(loadFilename, map_location=torch.device('cpu'))
    voc = Voc(corpus_name)
    voc.__dict__ = checkpoint['voc_dict']
    scripted_searcher.to(device)
    scripted_searcher.eval()

    def search_web(query):
        search_result_list1 = []
        search_result_list2 = []
        search_result_list3 = []
        disease = query
        website = ['mayo clinic', 'apollo clinic', 'cdc']
        query1 = disease + " " + website[0]
        # print(query1)
        search_result_list1.append(
            list(search(query1, tld="co.in", num=3, stop=5, pause=2)))
        # print(search_result_list1)
        query2 = disease + " " + website[1]
        # print(query2)
        search_result_list2.append(
            list(search(query2, tld="co.in", num=3, stop=5, pause=2)))
        # print(search_result_list2)
        query3 = disease + " " + website[2]
        # print(query3)
        search_result_list3.append(
            list(search(query3, tld="co.in", num=3, stop=5, pause=2)))
        # print(search_result_list3)

        return search_result_list1, search_result_list2, search_result_list3

    def mayo_info(html):
        soup = BeautifulSoup(html, 'html.parser')
        list1 = []
        for header in soup.find_all('p'):
            nextNode = header
            if(True):
                # print("heaad"+header.text.strip())
                list1.append(header.text.strip())
                while True:
                    nextNode = nextNode.nextSibling
                    if nextNode is None:
                        break
                    if isinstance(nextNode, NavigableString):
                        # list1.append(nextNode.strip())
                        print("")
                        # print("bye")
                        # print(nextNode.strip())
                    if isinstance(nextNode, Tag):
                        if nextNode.name == "p":
                            break
                        #print (nextNode.get_text(strip=True).strip())
                        # print(list1)
                        # print("hei")
                        # list1.append(nextNode.get_text(strip=True).strip())

        # print(list1)
        reference = listToString(list1)
        reference = reference.replace(
            ' and the triple-shield Mayo Clinic logo are trademarks of Mayo Foundation for Medical Education and Research.', '')
        reference = reference.replace(
            'Check out these best-sellers and special offers on books and newsletters from Mayo Clinic Press.', '')
        reference = reference.replace(
            'Mayo Clinic does not endorse companies or products. Advertising revenue supports our not-for-profit mission.', '')
        reference = reference.replace(
            'We are open for safe in-person care.', '')
        reference = reference.replace('Featured conditions', '')
        reference = reference.replace(
            'Any use of this site constitutes your agreement to the Terms and Conditions and Privacy Policy linked below.\n\nTerms and Conditions\nPrivacy Policy\nNotice of Privacy Practices\nNotice of Nondiscrimination\nManage Cookies', '')
        reference = reference.replace('MayoClinic.org', '')
        reference = reference.replace('Mayo Foundation ', '')
        reference = reference.replace(
            'This site complies with the  HONcode standard for trustworthy health information: verify here.', '')
        reference = reference.lower()
        reference = reference.replace('mayo', '')
        reference = reference.replace('clinic', '')
        reference = reference.replace('","', '')
        reference = reference.replace(
            'a single copy of these materials may be reprinted for noncommercial personal use only.', '')
        reference = reference.replace('  ', '')
        reference = reference.replace('"', '')
        # print(reference)
        flag = 0
        sol = []
        # traverse paragraphs from soup
        for data in soup.find_all("h2"):
            if(data.text.strip() == 'Prevention' or data.text.strip() == 'Overview' or data.text.strip() == 'Symptoms' or data.text.strip() == 'Causes' or data.text.strip() == 'Complications' or data.text.strip() == 'Risk factors' or data.text.strip() == 'Diagnosis' or data.text.strip() == 'Treatment'):
                sol.append(data.text.strip())
                sol.append("\n")
                # print(data.text.strip())
                para = data.find_next_sibling('ul')
                sol.append(para.text)
                # print(para.text)
                flag = 1

        if flag == 0:
            resultant = soup.findAll('h2')
            for i in range(len(resultant)):
                sol.append(resultant[i].text)

            # print(sol)

        sol = list(dict.fromkeys(sol))
        reference_counter = listToString(sol)

        #print("\nReference: ", *search_result_list)
        return reference, reference_counter

    def cdc_info(html):
        soup = BeautifulSoup(html, 'html.parser')
        list = []
        for header in soup.find_all('p'):
            nextNode = header
            if(True):
                # print("heaad"+header.text.strip())
                list.append(header.text.strip())
                while True:
                    nextNode = nextNode.nextSibling
                    if nextNode is None:
                        break
                    if isinstance(nextNode, NavigableString):
                        print("")
                        # list.append(nextNode.strip())
                    if isinstance(nextNode, Tag):
                        if nextNode.name == "p":
                            break
                        #print (nextNode.get_text(strip=True).strip())
                        # print(list)
                        # print("hi")
                        # list.append(nextNode.get_text(strip=True).strip())

        reference = listToString(list)
        reference = reference.replace(
            'Your email address will not be published. Required fields are marked', '')
        reference = reference.replace('Comment', '')
        reference = reference.replace('Name', '')
        reference = reference.replace('Website', '')
        reference = reference.replace('*', '')
        reference = reference.lower()
        reference = reference.replace('all rights reserved', '')
        # print(reference)
        return reference

    def apollo_info(html):
        soup = BeautifulSoup(html, 'html.parser')
        list = []
        for header in soup.find_all('p'):
            nextNode = header
            if(True):
                # print("heaad"+header.text.strip())
                list.append(header.text.strip())
                while True:
                    nextNode = nextNode.nextSibling
                    if nextNode is None:
                        break
                    if isinstance(nextNode, NavigableString):
                        print("")
                        # list.append(nextNode.strip())
                    if isinstance(nextNode, Tag):
                        if nextNode.name == "p":
                            break
                        #print (nextNode.get_text(strip=True).strip())
                        # print(list)
                        # print("hi")
                        # list.append(nextNode.get_text(strip=True).strip())

        reference = listToString(list)
        reference = reference.replace(
            'Your email address will not be published. Required fields are marked', '')
        reference = reference.replace('Comment', '')
        reference = reference.replace('Name', '')
        reference = reference.replace('Website', '')
        reference = reference.replace('*', '')
        reference = reference.lower()
        reference = reference.replace('all rights reserved', '')
        reference = reference.replace(
            'apollo hospitals enterprise limited', '')
        reference = reference.replace('apollo hospitals', '')
        reference = reference.replace(
            'https://www.apolloclinic.com/clinic-locator', '')
        reference = reference.replace('email', '')
        # print(reference)
        return reference

    def get_context(search_result_list1, search_result_list2, search_result_list3):
        url = search_result_list1[0]
        r = requests.get(url[0])
        html = r.text
        context_mayo, context_mayo2 = mayo_info(html)
        url2 = search_result_list2[0]
        r2 = requests.get(url2[0])
        html2 = r2.text
        context_apollo = apollo_info(html2)
        url3 = search_result_list3[0]
        r3 = requests.get(url3[0])
        html3 = r3.text
        context_cdc = cdc_info(html3)
        return context_apollo, context_mayo, context_mayo2, context_cdc

    nlp = spacy.load("en_core_web_sm")

    def cmp_check(wrn_cmp, symspell):
        max_edit_distance_lookup = 2
        sugs = symspell.lookup_compound(wrn_cmp,  max_edit_distance_lookup)
        for sug in sugs:
            return sug.term, sug.distance, sug.count

    def PP_spellings(x):
        path01 = "frequency_dictionary_en_82_765.txt"
        path02 = "frequency_bigramdictionary_en_243_342.txt"
        max_edit_distance_dictionary = 2
        prefix_length = 7
        symspell = SymSpell(max_edit_distance_dictionary, prefix_length)
        symspell.load_dictionary(corpus=path01, term_index=0, count_index=1)
        symspell.load_bigram_dictionary(
            corpus=path02, term_index=0, count_index=2)
        op, _, _ = cmp_check(x, symspell)
        return op

    def replace_acronyms(text, nlp):
        doc = nlp(text)
        altered_tok = [tok.text for tok in doc]
        for abrv in doc._.abbreviations:
            altered_tok[abrv.start] = str(abrv._.long_form)

        return(" ".join(altered_tok))

    def acro_pp(query):
        nlp1 = spacy.load("en_core_sci_sm")
        nlp1.add_pipe("abbreviation_detector")
        # Add the abbreviation pipe to the spacy pipeline.
        output = replace_acronyms(query, nlp1)
        del nlp1
        return output

    def spacy_process(text):
        doc = nlp(text)
        lemma_list = []
        for token in doc:
            lemma_list.append(token.lemma_)
        # print("Tokenize+Lemmatize:")
        # print(lemma_list)

        # removes stop-word like at , i , am ,etc

        # Filter the stopword
        #filtered_sentence =[]
        # for word in lemma_list:
            #lexeme = nlp.vocab[word]
            # if lexeme.is_stop == False:
            # filtered_sentence.append(word)

        # Remove punctuation
        # punctuations="?:!.,;"
        # for word in filtered_sentence:
            # if word in punctuations:
            # filtered_sentence.remove(word)

        counter_string = listToString(lemma_list)
        pp_string = counter_string.replace("-PRON-", "")

        return pp_string

    def spacy_process_gen(text):
        doc = nlp(text)
        lemma_list = []
        for token in doc:
            lemma_list.append(token.lemma_)
        # print("Tokenize+Lemmatize:")
        # print(lemma_list)

        # removes stop-word like at , i , am ,etc

        # Filter the stopword
        filtered_sentence = []
        for word in lemma_list:
            lexeme = nlp.vocab[word]
            if lexeme.is_stop == False:
                filtered_sentence.append(word)

        # Remove punctuation
        punctuations = "?:!.,;"
        for word in filtered_sentence:
            if word in punctuations:
                filtered_sentence.remove(word)

        counter_string = listToString(filtered_sentence)
        pp_string = counter_string.replace("-PRON-", "")

        return pp_string

    def qadiseasebert(question, text):
        tokenizer = AutoTokenizer.from_pretrained(
            "biobert_v1.1_pubmed_squad_v2", model_max_length=512)
        inputs = tokenizer(question, text, add_special_tokens=True,
                           truncation=True, return_tensors="pt")
        input_ids = inputs["input_ids"].tolist()[0]
        model = AutoModelForQuestionAnswering.from_pretrained(
            "biobert_v1.1_pubmed_squad_v2")
        outputs = model(**inputs)
        answer_start_scores = outputs.start_logits
        answer_end_scores = outputs.end_logits
        answer_start = torch.argmax(answer_start_scores)
        answer_end = torch.argmax(answer_end_scores) + 1
        answer = tokenizer.convert_tokens_to_string(
            tokenizer.convert_ids_to_tokens(input_ids[answer_start:answer_end]))
        answer = answer.replace("[CLS]", "")
        answer = answer.replace("[SEP]", "")
        answer = answer.replace(question, "")
        answer = answer.replace(spacy_process(question), "")
        del tokenizer
        del model
        return answer

    def txt_pp(question, text):
        counter = spacy_process(question)
        res = text.split()
        text = counter+" "+text
        return question, text

    def qadisease_c(query):
        list1, list2, list3 = search_web(query)
        context1, context2, context2_1, context3 = get_context(
            list1, list2, list3)
        _, ctxt1 = txt_pp(query, context1)
        # _,ctxt2=txt_pp(query,context2)
        # _,ctxt3=txt_pp(query,context2_1)
        # _,ctxt4=txt_pp(query,context3)
        ans1 = qadiseasebert(query, ctxt1)
        #ans2 = qadiseasebert(query,ctxt2)
        #ans3 = qadiseasebert(query,ctxt3)
        #ans4 = qadiseasebert(query,ctxt4)
        score = []
        ans_t = []
        counter_q = nlp(spacy_process(query))
        score.append(counter_q.similarity(nlp(ans1)))
        # score.append(counter_q.similarity(nlp(ans2)))
        # score.append(counter_q.similarity(nlp(ans3)))
        # score.append(counter_q.similarity(nlp(ans4)))
        ans_t.append(ans1)
        # ans_t.append(ans2)
        # ans_t.append(ans3)
        # ans_t.append(ans4)
        print(score)
        return score, ans_t

    def f_ans(score, ans):
        fallback = "Couldnt understand your query, Please try again"
        maxpos = score.index(max(score))
        if maxpos == 0:
            return ans[0]
        elif maxpos == 1:
            return ans[1]
        elif maxpos == 2:
            return ans[2]
        elif maxpos == 3:
            return ans[3]
        else:
            return fallback

    def reemovNestings(l):
        for i in l:
            if type(i) == list:
                reemovNestings(i)
            else:
                output.append(i)

    def servicecard(lst):
        lst_p = []

        def reemovNestings(l):
            for i in l:
                if type(i) == list:
                    reemovNestings(i)
                else:
                    lst_p.append(i)
        reemovNestings(lst)
        keywords_check = ["sales", "yoga", "excercise", "diet", "nutrition", "nutritional", "nutritionist", "dietician", "consultation", "treatment", "alternative treatment", "weight management", "muscle gain", "pcod", "cardiovascular disease", "renal", "anaemia", "gastrointestinal",
                          "disease", "ayurveda", "ayurvedic", "naturopathy", "homeopathy", "unani", "siddha", "cureya", "doctor", "blood pressure", "heart", "artery", "overweight", "underweight", "pcos", "thyroid", "cardiovascular", "dialysis", "typhoid", "influenza", "malaria", "aids"]
        set_a = set(lst_p)
        set_b = set(keywords_check)
        set_c = set_a & set_b
        output = list(set_c)
        output = listToString(output)
        return output

    counter = 1
    counter2 = 0
    msg_type = 0
    msg_track = []
    url1 = ""
    url2 = ""
    price = ""
    try:
        while(counter == 1):
            query = input_query
            print(query)
            query_c = PP_spellings(query)
            print(query_c)
            query_c = acro_pp(query_c)
            print(query_c)
            output = query
            if(output == "q" or output == "quit" or output == "end" or output == "exit"):
                counter = 0
                print("Thanks")
                break

            check_services = query_c.split()
            keywords_check = ["sales", "doctor", "yoga", "diet", "services", "nutrition", "nutritional", "nutritionist", "dietician", "consultation", "treatment", "alternative treatment", "weight management", "muscle gain"]
            set_a = set(check_services)
            set_b = set(keywords_check)
            set_c = set_a & set_b
            if(len(set_c) > 0):
                msg_type = 2
                answer = "service"
                url1 = "req1"
                url2 = "req2"
                price = 450
                print(answer)
                # return msg_type, answer, url1, url2, price
                return Response({'solutions': createResponse(msg_type, answer, url1, url2, price)})

            output = Pipe.predict([spacy_process_i(query_c)])[0]
            print(output)
            msg_track.append(spacy_process_i(query).split(" "))
            if(len(msg_track) < 3):
                if(output == "medical"):
                    msg_type = 1
                    op, at = qadisease_c(query)
                    answer = f_ans(op, at)
                    print(answer)
                    if(len(answer) == 0):
                        answer = "Couldnt understand your query, Please try again in simpler words"
                    # print(msg_type)
                    # return msg_type, answer, url1, url2, price
                    return Response({'solutions': createResponse(msg_type, answer, url1, url2, price)})
                elif(output == "general"):
                    try:
                        msg_type = 0
                        #query = re.sub(r'[^\w]', ' ', query)
                        print(query)
                        query = spacy_process_gen(query)
                        print(query)
                        answer = evaluate(scripted_searcher, voc, query)
                        print(answer)
                        # print(msg_type)
                        # return msg_type, answer, url1, url2, price
                        return Response({'solutions': createResponse(msg_type, answer, url1, url2, price)})
                    except:
                        try:
                            msg_type = 0
                            #query = re.sub(r'[^\w]', ' ', query)
                            print(query)
                            query_c2 = PP_spellings(query)
                            print(query_c2)
                            query_c2 = acro_pp(query_c2)
                            print(query_c2)
                            query = spacy_process_gen(query_c2)
                            print(query)
                            answer = evaluate(scripted_searcher, voc, query)
                            print(answer)
                            # print(msg_type)
                            return msg_type, answer, url1, url2, price
                        except:
                            fallback = "Couldnt understand your query, Please try again in simpler words"
                            # return 0, fallback, url1, url2, price
                            response.append(0)
                            response.append(fallback)
                            response.append(url1)
                            response.append(url2)
                            response.append(price)
                            return Response(response)
                elif(output == "service"):
                    msg_type = 2
                    answer = "service"
                    url1 = "req1"
                    url2 = "req2"
                    price = 450
                    print(answer)
                    # return msg_type, answer, url1, url2, price
                    return Response({'solutions': createResponse(msg_type, answer, url1, url2, price)})
            else:
                msg_type = 2
                answer = servicecard(msg_track)
                # print(msg_type)
                print("\nCARD-services\n")
                print(answer)
                answer = "service"
                url1 = "req1"
                url2 = "req2"
                price = 450
                # return msg_type, answer, url1, url2, price
                return Response({'solutions': createResponse(msg_type, answer, url1, url2, price)})
    except:
        fallback = "Couldnt understand your query, Please try again"
        # return 0, fallback, url1, url2, price
        response.append(0)
        response.append(fallback)
        response.append(url1)
        response.append(url2)
        response.append(price)
        return Response(response)

    # return msg_type, answer, url1, url2, price
        return Response({'solutions': createResponse(msg_type, answer, url1, url2, price)})


def tt(input_query):
    USE_CUDA = torch.cuda.is_available()
    device = torch.device("cuda" if USE_CUDA else "cpu")

    def listToString(s):
        str1 = ""
        for ele in s:
            str1 += ele + " "
        return str1

    def cleaner(x):
        return [a for a in (''.join([a for a in x if a not in string.punctuation])).lower().split()]

    Pipe = pickle.load(open("model_v1.pk", "rb"))

    def spacy_process_i(text):
        doc = nlp(text)
        lemma_list = []
        for token in doc:
            lemma_list.append(token.lemma_)
        filtered_sentence = []
        for word in lemma_list:
            lexeme = nlp.vocab[word]
            if lexeme.is_stop == False:
                filtered_sentence.append(word)

        punctuations = "?:!.,;"
        for word in filtered_sentence:
            if word in punctuations:
                filtered_sentence.remove(word)

        counter_string = listToString(filtered_sentence)
        pp_string = counter_string.replace("-PRON-", "")

        return pp_string

    MAX_LENGTH = 20  # Maximum sentence length
    PAD_token = 0  # Used for padding short sentences
    SOS_token = 1  # Start-of-sentence token
    EOS_token = 2  # End-of-sentence token
    scripted_searcher = torch.jit.load('scripted_chatbot.pth')

    class Voc:
        def __init__(self, name):
            self.name = name
            self.trimmed = False
            self.word2index = {}
            self.word2count = {}
            self.index2word = {PAD_token: "PAD",
                               SOS_token: "SOS", EOS_token: "EOS"}
            self.num_words = 3  # Count SOS, EOS, PAD

        def addSentence(self, sentence):
            for word in sentence.split(' '):
                self.addWord(word)

        def addWord(self, word):
            if word not in self.word2index:
                self.word2index[word] = self.num_words
                self.word2count[word] = 1
                self.index2word[self.num_words] = word
                self.num_words += 1
            else:
                self.word2count[word] += 1

        # Remove words below a certain count threshold
        def trim(self, min_count):
            if self.trimmed:
                return
            self.trimmed = True
            keep_words = []
            for k, v in self.word2count.items():
                if v >= min_count:
                    keep_words.append(k)

            print('keep_words {} / {} = {:.4f}'.format(
                len(keep_words), len(self.word2index), len(
                    keep_words) / len(self.word2index)
            ))
            # Reinitialize dictionaries
            self.word2index = {}
            self.word2count = {}
            self.index2word = {PAD_token: "PAD",
                               SOS_token: "SOS", EOS_token: "EOS"}
            self.num_words = 3  # Count default tokens
            for word in keep_words:
                self.addWord(word)

    # Lowercase and remove non-letter characters

    def normalizeString(s):
        s = s.lower()
        s = re.sub(r"([.!?])", r" \1", s)
        s = re.sub(r"[^a-zA-Z.!?]+", r" ", s)
        return s

    # Takes string sentence, returns sentence of word indexes

    def indexesFromSentence(voc, sentence):
        try:
            print("one")
            return [voc.word2index[word] for word in sentence.split(' ')] + [EOS_token]
        except:
            print("two")
            return None

    def evaluate(searcher, voc, sentence, max_length=MAX_LENGTH):
        # Format input sentence as a batch
        # words -> indexes
        indexes_batch = [indexesFromSentence(voc, sentence)]
        # Create lengths tensor
        print(indexes_batch)
        lengths = torch.tensor([len(indexes) for indexes in indexes_batch])
        # Transpose dimensions of batch to match models' expectations
        input_batch = torch.LongTensor(indexes_batch).transpose(0, 1)
        # Use appropriate device
        input_batch = input_batch.to(device)
        lengths = lengths.to(device)
        # Decode sentence with searcher
        tokens, scores = searcher(input_batch, lengths, max_length)
        # indexes -> words
        decoded_words = [voc.index2word[token.item()] for token in tokens]
        return decoded_words

    # Evaluate inputs from user input (stdin)

    def evaluateInput(searcher, voc):
        input_sentence = ''
        while(1):
            try:
                # Get input sentence
                input_sentence = input('> ')
                # Check if it is quit case
                if input_sentence == 'q' or input_sentence == 'quit':
                    break
                # Normalize sentence
                input_sentence = normalizeString(input_sentence)
                # Evaluate sentence
                output_words = evaluate(searcher, voc, input_sentence)
                # Format and print response sentence
                output_words[:] = [x for x in output_words if not (
                    x == 'EOS' or x == 'PAD')]
                print('Bot:', ' '.join(output_words))

            except KeyError:
                print("Error: Encountered unknown word.")

    # Normalize input sentence and call evaluate()
    def evaluateExample(sentence, searcher, voc):
        print("> " + sentence)
        # Normalize sentence
        input_sentence = normalizeString(sentence)
        # Evaluate sentence
        output_words = evaluate(searcher, voc, input_sentence)
        output_words[:] = [x for x in output_words if not (
            x == 'EOS' or x == 'PAD')]
        print('Bot:', ' '.join(output_words))
    corpus_name = "cornell movie-dialogs corpus"
    model_name = 'cb_model'
    attn_model = 'dot'
    checkpoint_iter = 4000
    loadFilename = '4000_checkpoint.tar'
    checkpoint = torch.load(loadFilename, map_location=torch.device('cpu'))
    voc = Voc(corpus_name)
    voc.__dict__ = checkpoint['voc_dict']
    scripted_searcher.to(device)
    scripted_searcher.eval()

    def search_web(query):
        search_result_list1 = []
        search_result_list2 = []
        search_result_list3 = []
        disease = query
        website = ['mayo clinic', 'apollo clinic', 'cdc']
        query1 = disease + " " + website[0]
        # print(query1)
        search_result_list1.append(
            list(search(query1, tld="co.in", num=3, stop=5, pause=2)))
        # print(search_result_list1)
        query2 = disease + " " + website[1]
        # print(query2)
        search_result_list2.append(
            list(search(query2, tld="co.in", num=3, stop=5, pause=2)))
        # print(search_result_list2)
        query3 = disease + " " + website[2]
        # print(query3)
        search_result_list3.append(
            list(search(query3, tld="co.in", num=3, stop=5, pause=2)))
        # print(search_result_list3)

        return search_result_list1, search_result_list2, search_result_list3

    def mayo_info(html):
        soup = BeautifulSoup(html, 'html.parser')
        list1 = []
        for header in soup.find_all('p'):
            nextNode = header
            if(True):
                # print("heaad"+header.text.strip())
                list1.append(header.text.strip())
                while True:
                    nextNode = nextNode.nextSibling
                    if nextNode is None:
                        break
                    if isinstance(nextNode, NavigableString):
                        # list1.append(nextNode.strip())
                        print("")
                        # print("bye")
                        # print(nextNode.strip())
                    if isinstance(nextNode, Tag):
                        if nextNode.name == "p":
                            break
                        #print (nextNode.get_text(strip=True).strip())
                        # print(list1)
                        # print("hei")
                        # list1.append(nextNode.get_text(strip=True).strip())

        # print(list1)
        reference = listToString(list1)
        reference = reference.replace(
            ' and the triple-shield Mayo Clinic logo are trademarks of Mayo Foundation for Medical Education and Research.', '')
        reference = reference.replace(
            'Check out these best-sellers and special offers on books and newsletters from Mayo Clinic Press.', '')
        reference = reference.replace(
            'Mayo Clinic does not endorse companies or products. Advertising revenue supports our not-for-profit mission.', '')
        reference = reference.replace(
            'We are open for safe in-person care.', '')
        reference = reference.replace('Featured conditions', '')
        reference = reference.replace(
            'Any use of this site constitutes your agreement to the Terms and Conditions and Privacy Policy linked below.\n\nTerms and Conditions\nPrivacy Policy\nNotice of Privacy Practices\nNotice of Nondiscrimination\nManage Cookies', '')
        reference = reference.replace('MayoClinic.org', '')
        reference = reference.replace('Mayo Foundation ', '')
        reference = reference.replace(
            'This site complies with the  HONcode standard for trustworthy health information: verify here.', '')
        reference = reference.lower()
        reference = reference.replace('mayo', '')
        reference = reference.replace('clinic', '')
        reference = reference.replace('","', '')
        reference = reference.replace(
            'a single copy of these materials may be reprinted for noncommercial personal use only.', '')
        reference = reference.replace('  ', '')
        reference = reference.replace('"', '')
        # print(reference)
        flag = 0
        sol = []
        # traverse paragraphs from soup
        for data in soup.find_all("h2"):
            if(data.text.strip() == 'Prevention' or data.text.strip() == 'Overview' or data.text.strip() == 'Symptoms' or data.text.strip() == 'Causes' or data.text.strip() == 'Complications' or data.text.strip() == 'Risk factors' or data.text.strip() == 'Diagnosis' or data.text.strip() == 'Treatment'):
                sol.append(data.text.strip())
                sol.append("\n")
                # print(data.text.strip())
                para = data.find_next_sibling('ul')
                sol.append(para.text)
                # print(para.text)
                flag = 1

        if flag == 0:
            resultant = soup.findAll('h2')
            for i in range(len(resultant)):
                sol.append(resultant[i].text)

            # print(sol)

        sol = list(dict.fromkeys(sol))
        reference_counter = listToString(sol)

        #print("\nReference: ", *search_result_list)
        return reference, reference_counter

    def cdc_info(html):
        soup = BeautifulSoup(html, 'html.parser')
        list = []
        for header in soup.find_all('p'):
            nextNode = header
            if(True):
                # print("heaad"+header.text.strip())
                list.append(header.text.strip())
                while True:
                    nextNode = nextNode.nextSibling
                    if nextNode is None:
                        break
                    if isinstance(nextNode, NavigableString):
                        print("")
                        # list.append(nextNode.strip())
                    if isinstance(nextNode, Tag):
                        if nextNode.name == "p":
                            break
                        #print (nextNode.get_text(strip=True).strip())
                        # print(list)
                        # print("hi")
                        # list.append(nextNode.get_text(strip=True).strip())

        reference = listToString(list)
        reference = reference.replace(
            'Your email address will not be published. Required fields are marked', '')
        reference = reference.replace('Comment', '')
        reference = reference.replace('Name', '')
        reference = reference.replace('Website', '')
        reference = reference.replace('*', '')
        reference = reference.lower()
        reference = reference.replace('all rights reserved', '')
        # print(reference)
        return reference

    def apollo_info(html):
        soup = BeautifulSoup(html, 'html.parser')
        list = []
        for header in soup.find_all('p'):
            nextNode = header
            if(True):
                # print("heaad"+header.text.strip())
                list.append(header.text.strip())
                while True:
                    nextNode = nextNode.nextSibling
                    if nextNode is None:
                        break
                    if isinstance(nextNode, NavigableString):
                        print("")
                        # list.append(nextNode.strip())
                    if isinstance(nextNode, Tag):
                        if nextNode.name == "p":
                            break
                        #print (nextNode.get_text(strip=True).strip())
                        # print(list)
                        # print("hi")
                        # list.append(nextNode.get_text(strip=True).strip())

        reference = listToString(list)
        reference = reference.replace(
            'Your email address will not be published. Required fields are marked', '')
        reference = reference.replace('Comment', '')
        reference = reference.replace('Name', '')
        reference = reference.replace('Website', '')
        reference = reference.replace('*', '')
        reference = reference.lower()
        reference = reference.replace('all rights reserved', '')
        reference = reference.replace(
            'apollo hospitals enterprise limited', '')
        reference = reference.replace('apollo hospitals', '')
        reference = reference.replace(
            'https://www.apolloclinic.com/clinic-locator', '')
        reference = reference.replace('email', '')
        # print(reference)
        return reference

    def get_context(search_result_list1, search_result_list2, search_result_list3):
        url = search_result_list1[0]
        r = requests.get(url[0])
        html = r.text
        context_mayo, context_mayo2 = mayo_info(html)
        url2 = search_result_list2[0]
        r2 = requests.get(url2[0])
        html2 = r2.text
        context_apollo = apollo_info(html2)
        url3 = search_result_list3[0]
        r3 = requests.get(url3[0])
        html3 = r3.text
        context_cdc = cdc_info(html3)
        return context_apollo, context_mayo, context_mayo2, context_cdc

    nlp = spacy.load("en_core_web_sm")

    def cmp_check(wrn_cmp, symspell):
        max_edit_distance_lookup = 2
        sugs = symspell.lookup_compound(wrn_cmp,  max_edit_distance_lookup)
        for sug in sugs:
            return sug.term, sug.distance, sug.count

    def PP_spellings(x):
        path01 = "frequency_dictionary_en_82_765.txt"
        path02 = "frequency_bigramdictionary_en_243_342.txt"
        max_edit_distance_dictionary = 2
        prefix_length = 7
        symspell = SymSpell(max_edit_distance_dictionary, prefix_length)
        symspell.load_dictionary(corpus=path01, term_index=0, count_index=1)
        symspell.load_bigram_dictionary(
            corpus=path02, term_index=0, count_index=2)
        op, _, _ = cmp_check(x, symspell)
        return op

    def replace_acronyms(text, nlp):
        doc = nlp(text)
        altered_tok = [tok.text for tok in doc]
        for abrv in doc._.abbreviations:
            altered_tok[abrv.start] = str(abrv._.long_form)

        return(" ".join(altered_tok))

    def acro_pp(query):
        nlp1 = spacy.load("en_core_sci_sm")
        nlp1.add_pipe("abbreviation_detector")
        # Add the abbreviation pipe to the spacy pipeline.
        output = replace_acronyms(query, nlp1)
        del nlp1
        return output

    def spacy_process(text):
        doc = nlp(text)
        lemma_list = []
        for token in doc:
            lemma_list.append(token.lemma_)
        # print("Tokenize+Lemmatize:")
        # print(lemma_list)

        # removes stop-word like at , i , am ,etc

        # Filter the stopword
        #filtered_sentence =[]
        # for word in lemma_list:
            #lexeme = nlp.vocab[word]
            # if lexeme.is_stop == False:
            # filtered_sentence.append(word)

        # Remove punctuation
        # punctuations="?:!.,;"
        # for word in filtered_sentence:
            # if word in punctuations:
            # filtered_sentence.remove(word)

        counter_string = listToString(lemma_list)
        pp_string = counter_string.replace("-PRON-", "")

        return pp_string

    def spacy_process_gen(text):
        doc = nlp(text)
        lemma_list = []
        for token in doc:
            lemma_list.append(token.lemma_)
        # print("Tokenize+Lemmatize:")
        # print(lemma_list)

        # removes stop-word like at , i , am ,etc

        # Filter the stopword
        filtered_sentence = []
        for word in lemma_list:
            lexeme = nlp.vocab[word]
            if lexeme.is_stop == False:
                filtered_sentence.append(word)

        # Remove punctuation
        punctuations = "?:!.,;"
        for word in filtered_sentence:
            if word in punctuations:
                filtered_sentence.remove(word)

        counter_string = listToString(filtered_sentence)
        pp_string = counter_string.replace("-PRON-", "")

        return pp_string

    def qadiseasebert(question, text):
        tokenizer = AutoTokenizer.from_pretrained(
            "biobert_v1.1_pubmed_squad_v2", model_max_length=512)
        inputs = tokenizer(question, text, add_special_tokens=True,
                           truncation=True, return_tensors="pt")
        input_ids = inputs["input_ids"].tolist()[0]
        model = AutoModelForQuestionAnswering.from_pretrained(
            "biobert_v1.1_pubmed_squad_v2")
        outputs = model(**inputs)
        answer_start_scores = outputs.start_logits
        answer_end_scores = outputs.end_logits
        answer_start = torch.argmax(answer_start_scores)
        answer_end = torch.argmax(answer_end_scores) + 1
        answer = tokenizer.convert_tokens_to_string(
            tokenizer.convert_ids_to_tokens(input_ids[answer_start:answer_end]))
        answer = answer.replace("[CLS]", "")
        answer = answer.replace("[SEP]", "")
        answer = answer.replace(question, "")
        answer = answer.replace(spacy_process(question), "")
        del tokenizer
        del model
        return answer

    def txt_pp(question, text):
        counter = spacy_process(question)
        res = text.split()
        text = counter+" "+text
        return question, text

    def qadisease_c(query):
        list1, list2, list3 = search_web(query)
        context1, context2, context2_1, context3 = get_context(
            list1, list2, list3)
        _, ctxt1 = txt_pp(query, context1)
        # _,ctxt2=txt_pp(query,context2)
        # _,ctxt3=txt_pp(query,context2_1)
        # _,ctxt4=txt_pp(query,context3)
        ans1 = qadiseasebert(query, ctxt1)
        #ans2 = qadiseasebert(query,ctxt2)
        #ans3 = qadiseasebert(query,ctxt3)
        #ans4 = qadiseasebert(query,ctxt4)
        score = []
        ans_t = []
        counter_q = nlp(spacy_process(query))
        score.append(counter_q.similarity(nlp(ans1)))
        # score.append(counter_q.similarity(nlp(ans2)))
        # score.append(counter_q.similarity(nlp(ans3)))
        # score.append(counter_q.similarity(nlp(ans4)))
        ans_t.append(ans1)
        # ans_t.append(ans2)
        # ans_t.append(ans3)
        # ans_t.append(ans4)
        print(score)
        return score, ans_t

    def f_ans(score, ans):
        fallback = "Couldnt understand your query, Please try again"
        maxpos = score.index(max(score))
        if maxpos == 0:
            return ans[0]
        elif maxpos == 1:
            return ans[1]
        elif maxpos == 2:
            return ans[2]
        elif maxpos == 3:
            return ans[3]
        else:
            return fallback

    def reemovNestings(l):
        for i in l:
            if type(i) == list:
                reemovNestings(i)
            else:
                output.append(i)

    def servicecard(lst):
        lst_p = []

        def reemovNestings(l):
            for i in l:
                if type(i) == list:
                    reemovNestings(i)
                else:
                    lst_p.append(i)
        reemovNestings(lst)
        keywords_check = ["sales", "yoga", "excercise", "diet", "nutrition", "nutritional", "nutritionist", "dietician", "consultation", "treatment", "alternative treatment", "weight management", "muscle gain", "pcod", "cardiovascular disease", "renal", "anaemia", "gastrointestinal",
                          "disease", "ayurveda", "ayurvedic", "naturopathy", "homeopathy", "unani", "siddha", "cureya", "doctor", "blood pressure", "heart", "artery", "overweight", "underweight", "pcos", "thyroid", "cardiovascular", "dialysis", "typhoid", "influenza", "malaria", "aids"]
        set_a = set(lst_p)
        set_b = set(keywords_check)
        set_c = set_a & set_b
        output = list(set_c)
        output = listToString(output)
        return output

    counter = 1
    counter2 = 0
    msg_type = 0
    msg_track = []
    url1 = ""
    url2 = ""
    price = ""
    try:
        while(counter == 1):
            query = input_query
            print(query)
            query_c = PP_spellings(query)
            print(query_c)
            query_c = acro_pp(query_c)
            print(query_c)
            output = query
            if(output == "q" or output == "quit" or output == "end" or output == "exit"):
                counter = 0
                print("Thanks")
                break

            check_services = query_c.split()
            keywords_check = ["sales", "doctor", "yoga", "diet", "services", "nutrition", "nutritional", "nutritionist",
                              "dietician", "consultation", "treatment", "alternative treatment", "weight management", "muscle gain"]
            set_a = set(check_services)
            set_b = set(keywords_check)
            set_c = set_a & set_b
            if(len(set_c) > 0):
                msg_type = 2
                answer = "service"
                url1 = "req1"
                url2 = "req2"
                price = 450
                print(answer)
                return msg_type, answer, url1, url2, price

            output = Pipe.predict([spacy_process_i(query_c)])[0]
            print(output)
            msg_track.append(spacy_process_i(query).split(" "))
            if(len(msg_track) < 3):
                if(output == "medical"):
                    msg_type = 1
                    op, at = qadisease_c(query)
                    answer = f_ans(op, at)
                    print(answer)
                    if(len(answer) == 0):
                        answer = "Couldnt understand your query, Please try again in simpler words"
                    # print(msg_type)
                    return msg_type, answer, url1, url2, price
                elif(output == "general"):
                    try:
                        msg_type = 0
                        #query = re.sub(r'[^\w]', ' ', query)
                        print(query)
                        query = spacy_process_gen(query)
                        print(query)
                        answer = evaluate(scripted_searcher, voc, query)
                        print(answer)
                        # print(msg_type)
                        return msg_type, answer, url1, url2, price
                    except:
                        try:
                            msg_type = 0
                            #query = re.sub(r'[^\w]', ' ', query)
                            print(query)
                            query_c2 = PP_spellings(query)
                            print(query_c2)
                            query_c2 = acro_pp(query_c2)
                            print(query_c2)
                            query = spacy_process_gen(query_c2)
                            print(query)
                            answer = evaluate(scripted_searcher, voc, query)
                            print(answer)
                            # print(msg_type)
                            return msg_type, answer, url1, url2, price
                        except:
                            fallback = "Couldnt understand your query, Please try again in simpler words"
                            return 0, fallback, url1, url2, price
                elif(output == "service"):
                    msg_type = 2
                    answer = "service"
                    url1 = "req1"
                    url2 = "req2"
                    price = 450
                    print(answer)
                    return msg_type, answer, url1, url2, price
            else:
                msg_type = 2
                answer = servicecard(msg_track)
                # print(msg_type)
                print("\nCARD-services\n")
                print(answer)
                answer = "service"
                url1 = "req1"
                url2 = "req2"
                price = 450
                return msg_type, answer, url1, url2, price
    except:
        fallback = "Couldnt understand your query, Please try again"
        return 0, fallback, url1, url2, price

    return msg_type, answer, url1, url2, price


class BotSolutions(APIView):

    def post(self, request):
        data = json.loads(request.body.decode("utf-8"))["message"]
        msg_type, ans, url1, url2, price = tt(data)
        ls = []
        ls.append(msg_type)
        ls.append(ans)
        ls.append(url1)
        ls.append(url2)
        ls.append(price)
        #ans = list(ans.split(" "))
        # solutions = manipulated data...
        print(ls)
        return JsonResponse({"solutions": ls}, status=status.HTTP_200_OK)
