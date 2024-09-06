import nltk
import numpy as np
from nltk.translate.meteor_score import single_meteor_score
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from sklearn.feature_extraction.text import TfidfVectorizer
from transformers import BertTokenizer, BertModel
import json
import torch
from tqdm import tqdm

nltk.download('punkt')
nltk.download('wordnet')

def calculate_average_scores(candidates, references):
    # Initialize BERT tokenizer and model
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertModel.from_pretrained('bert-base-uncased')

    # Initialize score accumulators
    total_cider = total_meteor = total_bleu4 = total_bert = 0

    # Process each pair of sentences
    for candidate, reference in tqdm(zip(candidates, references), total=len(candidates)):
        candidate_tokens = nltk.word_tokenize(candidate)
        reference_tokens = nltk.word_tokenize(reference)

        # CIDEr calculation
        corpus = [' '.join(candidate_tokens), ' '.join(reference_tokens)]
        vectorizer = TfidfVectorizer()
        tfidf_matrix = vectorizer.fit_transform(corpus).toarray()
        cider_score = np.dot(tfidf_matrix[0], tfidf_matrix[1]) / (np.linalg.norm(tfidf_matrix[0]) * np.linalg.norm(tfidf_matrix[1]))
        if np.isnan(cider_score):
            cider_score=0
            

        # METEOR calculation
        meteor_score_value = single_meteor_score(reference_tokens, candidate_tokens)

        # BLEU-4 calculation
        smoothing = SmoothingFunction().method1
        bleu4_score = sentence_bleu([reference_tokens], candidate_tokens, smoothing_function=smoothing)


        # BERT similarity calculation
        # Tokenize and encode sentences for BERT
        inputs_candidate = tokenizer(candidate, return_tensors="pt", padding=True, truncation=True)
        inputs_reference = tokenizer(reference, return_tensors="pt", padding=True, truncation=True)

        # Generate embeddings
        with torch.no_grad():
            candidate_embedding = model(**inputs_candidate).last_hidden_state.mean(dim=1)
            reference_embedding = model(**inputs_reference).last_hidden_state.mean(dim=1)

        # Cosine similarity
        bert_similarity = np.dot(candidate_embedding[0].numpy(), reference_embedding[0].numpy()) / (np.linalg.norm(candidate_embedding[0].numpy()) * np.linalg.norm(reference_embedding[0].numpy()))

        # Accumulate scores
        total_cider += cider_score
        total_meteor += meteor_score_value
        total_bleu4 += bleu4_score
        total_bert += bert_similarity
    # Calculate averages
    num_pairs = len(candidates)
    average_cider = total_cider / num_pairs
    average_meteor = total_meteor / num_pairs
    average_bleu4 = total_bleu4 / num_pairs
    average_bert = total_bert / num_pairs

    return average_cider, average_meteor, average_bleu4, average_bert

with open("datasets/SDN_test_conversations.json","r")as f:
    gt_raw=json.load(f)
with open("out/SDN_test_conversations.json","r")as f:
    pred_raw=json.load(f)
gts_seen, gts_unseen=[],[]
preds_seen, preds_unseen=[],[]
count_seen, count_unseen=0,0
total_seen, total_unseen=0,0
for i,pred in enumerate(pred_raw):
    if gt_raw[i]["unseen"]:
        preds_unseen.append(pred[-1].split(":")[-1])
        gts_unseen.append(gt_raw[i][ "conversations"][-1]["value"].split(":")[-1])
        if pred[-1].split("\n")[0].split(":")[1] == gt_raw[i][ "conversations"][-1]["value"].split("\n")[0].split(":")[1]:
            count_unseen+=1
        total_unseen+=1
    else:
        preds_seen.append(pred[-1].split(":")[-1])

        gts_seen.append(gt_raw[i][ "conversations"][-1]["value"].split(":")[-1])
        try:
            if pred[-1].split("\n")[0].split(":")[1] == gt_raw[i][ "conversations"][-1]["value"].split("\n")[0].split(":")[1]:
                count_seen+=1
        except:
            print(pred[-1])
        total_seen+=1
print("unseen:")
print(count_unseen/total_unseen)
# Example usage
average_cider, average_meteor, average_bleu4, average_bert = calculate_average_scores(preds_unseen, gts_unseen)
print("Average CIDEr Score:", average_cider)
print("Average BERT Similarity:", average_bert)
print("Average METEOR Score:", average_meteor)
print("Average BLEU-4 Score:", average_bleu4)
print("seen:")
print(count_seen/total_seen)
# Example usage
average_cider, average_meteor, average_bleu4, average_bert = calculate_average_scores(preds_seen, gts_seen)
print("Average CIDEr Score:", average_cider)
print("Average BERT Similarity:", average_bert)
print("Average METEOR Score:", average_meteor)
print("Average BLEU-4 Score:", average_bleu4)

    