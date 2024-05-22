# @title EmployeeWord2Vec
from sklearn.metrics.pairwise import cosine_similarity
from assistant import preprocess_text
# from gensim.models import Word2Vec
import numpy as np

class EmployeeWord2Vec:
    def __init__(self, docs, filepath=None, model=None):
        self.model = None
        if filepath is not None:
            self.load_model(filepath)
        elif model is not None:
            self.model = model

        self.vectors = None
        self.docs_names = None

        if docs is not None:
            self._init_vectors(docs)

    def train(self, data):
        # Traini
        pass

    def _init_vectors(self, docs):
        self.docs_names = docs
        self.vectors = []
        for doc in docs:
            self.vectors.append(self.sentence_to_vector(doc))

    def get_word_vector(self, word):
        self.model.wv[word]

    def save_model(self, filepath):
        self.model.save(filepath)


    def load_model(self, filepath):
        self.model = Word2Vec.load(filepath)

    def get_model(self):
        return self.model

    def getRelevanceVector(self, textReq, system_name, moc_name, docsForEval):

        relevance_vector = [0] * len(docsForEval)

        similarityVectorMoc = self.getCosineSimilarityVectorsWord2Vec(textReq= moc_name,
                            docs2 = docsForEval,
                            )

        similarityVectorTextRec = self.getCosineSimilarityVectorsWord2Vec(textReq=textReq,
                            docs2 = docsForEval,
                            )
        similarityVectorSystemName = self.getCosineSimilarityVectorsWord2Vec(textReq=system_name,
                            docs2 = docsForEval,
                            )
        relevance_vector += similarityVectorMoc
        relevance_vector += similarityVectorTextRec
        relevance_vector += similarityVectorSystemName


        return relevance_vector[0]


    def getSimilarytyDoc(self, doc_name, count_rel_docs=15):
        new_vector = self.sentence_to_vector(doc_name)
        cosine_similarities = cosine_similarity([new_vector], self.vectors)

        # Получение наиболее похожих предложений
        most_similar_indices = cosine_similarities.argsort()[0][::-1]


        most_similar_sentences = [self.docs_names[i] for i in most_similar_indices]
        # Получение уникальных элементов, их индексов и количества
        unique_values, indices, counts = np.unique(most_similar_sentences, return_index=True, return_counts=True)

        # Отбор уникальных элементов с сохранением порядка
        unique_arr = unique_values[np.argsort(indices)]
        unique_arr = unique_arr[:count_rel_docs]
        return unique_arr.tolist()# Implement your logic to calculate the similarity between a given document and a list of documents here

    def getCosineSimilarityVectorsWord2Vec(self, textReq, docs2):
        new_vector1 = self.sentence_to_vector(textReq)
        new_vector2 = []
        for doc in docs2:
            new_vector2.append(self.sentence_to_vector(doc))
        cosine_similarities = cosine_similarity([new_vector1], new_vector2)
        return cosine_similarities

    def sentence_to_vector(self, sentence):
        sent_docs = preprocess_text(sentence).split()
        sentence_vec = np.zeros(self.model.vector_size)
        word_count = 0

        for word in sent_docs:
            if word in self.model.wv:
                sentence_vec += self.model.wv[word]
                word_count += 1

        if word_count > 0:
            return sentence_vec / word_count
        else:
            # Возвращать нулевой вектор или другое значение по умолчанию, если предложение не содержит слов из словаря
            return np.zeros(self.model.vector_size)