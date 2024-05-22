import itertools
from sklearn.metrics.pairwise import cosine_similarity
from assistant import preprocess_text
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np

class RF: # relevance feedback

    def __init__(self, doc_names_mod=None):
        self.doc_names_mod = None
        if doc_names_mod is not None:
            self.doc_names_mod = doc_names_mod
        pass

    def set_mod(self, mod_name, doc_names_mod):
        self.mod_name = mod_name
        self.doc_names_mod = doc_names_mod


    def getRFSimilarity(self, textReq, rf_docs_names, count_rel_docs=30):
        vectorizer_for_rec = TfidfVectorizer(preprocessor=preprocess_text)

        docs_text = []
        for doc in rf_docs_names:
            docs_text.append(str(doc)[str.find(str(doc), ' ', 9) + 1:])
        textRec = textReq
        docs_text.append(textRec)

        tfifd_matrix = vectorizer_for_rec.fit_transform(docs_text)
        rec_vec_for_actual_doc = tfifd_matrix.toarray()
        rec_keywords_vec = np.zeros(len(vectorizer_for_rec.get_feature_names_out()))
        for i, doc in enumerate(docs_text):
            rec_keywords_vec += rec_vec_for_actual_doc[i]

        docs_vec = vectorizer_for_rec.transform(self.doc_names_mod)
        similarity = cosine_similarity([rec_keywords_vec], docs_vec)

        sorted_relevance = self.sortedRelevance(similarity[0])

        docs_predicted = []
        for rel_coeff, doc in itertools.islice(sorted_relevance.items(), count_rel_docs):
            docs_predicted.append(doc)

        return docs_predicted  # list of N doc names


    def sortedRelevance(self, similarity):
        relevance_map = {}
        for rel, doc in zip(similarity, self.doc_names_mod):
            relevance_map[rel] = doc
        sorted_map = dict(sorted(relevance_map.items(), reverse=True))
        return sorted_map