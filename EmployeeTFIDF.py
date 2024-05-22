import os
from sklearn.metrics.pairwise import cosine_similarity
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
from assistant import preprocess_text


class EmployeeTFIDF:
    def __init__(self, doc_names_mod, mod_name="", filepath=None, filepath_with_text=None):
        self.vectorizer = None
        self.vectorizer_with_text_req = None

        self.mod_name = mod_name

        if filepath is not None:
            self.load_vectorizer(filepath)
        if filepath_with_text is not None:
            self.load_vectorizer_with_text_req(filepath_with_text)
        self.doc_names_mod = doc_names_mod
        self.file_name_mod = f"models/vectors_mod_{mod_name}.pkl"
        if not os.path.isfile(self.file_name_mod):
            self.vectors_mod = self.vectorizer.transform(doc_names_mod)
            self.save_vectorizer(self.vectors_mod, self.file_name_mod)
        else:
            with open(self.file_name_mod, "rb") as file:
                self.vectors_mod = pickle.load(file)

    def set_mod(self, mod_name, doc_names_mod):
        self.doc_names_mod = doc_names_mod
        self.mod_name = mod_name
        self.file_name_mod = f"models/vectors_mod_{mod_name}.pkl"
        self.vectors_mod = self._get_vectors_mod()

    def _get_vectors_mod(self):
        self.file_name_mod = f"models/vectors_mod_{self.mod_name}.pkl"
        if not os.path.isfile(self.file_name_mod):
            vectors_mod = self.vectorizer.transform(self.doc_names_mod)
            self.save_vectorizer(self.vectors_mod, self.file_name_mod)
        else:
            with open(self.file_name_mod, "rb") as file:
                vectors_mod = pickle.load(file)
        return vectors_mod

    def update_vector_mod(self, doc_names_mod):
        self.vectors_mod = self.vectorizer.transform(doc_names_mod)
        self.save_vectorizer(self.vectors_mod)

    def load_vectorizer(self, filepath):
        with open(filepath, 'rb') as f:
            self.vectorizer = pickle.load(f)
    def load_vectorizer_with_text_req(self, filepath):
        with open(filepath, 'rb') as f:
            self.vectorizer_with_text_req = pickle.load(f)

    def save_vectorizer(self, filepath):
        with open(filepath, 'wb') as f:
            pickle.dump(self.vectorizer, f)

    def train(self, documents):
        self.vectorizer = TfidfVectorizer(preprocessor=preprocess_text)
        return self.vectorizer.fit_transform(documents)

    def get_vectorizer(self):
        return self.vectorizer

    def get_top_keywords(self, n):
        pass


    def getRelevanceVector(self, textReq, system_name, moc_name, docsForEval):
        relevance_vector = [0] * len(docsForEval)
        vector_docs = self.vectorizer_with_text_req.transform(docsForEval)

        similarityVectorMoc = self.getCosineSimilarityVectors(textReq = moc_name,
                            vector_docs=vector_docs,
                            )

        similarityVectorTextRec = self.getCosineSimilarityVectors(textReq=textReq,
                            vector_docs=vector_docs
                            )
        print(similarityVectorTextRec)
        similarityVectorSystemName = self.getCosineSimilarityVectors(textReq=system_name,
                            vector_docs=vector_docs
                            )
        relevance_vector += similarityVectorMoc
        relevance_vector += similarityVectorTextRec
        relevance_vector += similarityVectorSystemName

        return relevance_vector[0]

    def getCosineSimilarityVectors(self, textReq, vector_docs):
        new_vector1 = self.vectorizer_with_text_req.transform([textReq])
        cosine_similarities = cosine_similarity(new_vector1, vector_docs)
        return cosine_similarities

    def getSimilarytyDoc(self, doc_name, count_rel_docs=15):
        # использовать векторайзер обученный только на названиях документов. Либо попробовать обученный на всём, но тогда сделать transform
        # только для названий все и сюда закидывать этот vectors.

        new_vector = self.vectorizer.transform([doc_name])
        cosine_similarities = cosine_similarity(new_vector, self.vectors_mod)

        # Получение наиболее похожих предложений
        most_similar_indices = cosine_similarities.argsort()[0][::-1]


        most_similar_sentences = [self.doc_names_mod[i] for i in most_similar_indices]
        # Получение уникальных элементов, их индексов и количества
        unique_values, indices, counts = np.unique(most_similar_sentences, return_index=True, return_counts=True)

        unique_arr = unique_values[np.argsort(indices)]
        unique_arr = unique_arr[:count_rel_docs]
        #   for sentence in unique_arr:
        #       print(sentence)
        #   print(type(unique_arr))
        return unique_arr.tolist()
    
    def save_vectorizer(self, vectorizer, filepath):
        try:
            with open(filepath, 'wb') as file:
                pickle.dump(vectorizer, file)
            print(f"Векторизатор успешно сохранен в файл: {filepath}")
        except Exception as e:
            print(f"Ошибка при сохранении векторизатора: {e}")