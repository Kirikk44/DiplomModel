
from ast import mod
import itertools
import pickle

from EmployeeTFIDF import EmployeeTFIDF
from RF import RF
from assistant import preprocess_text


class Predictor():
    def __init__(self, dataset_handler, mod_name = "modsh", filepath_tfidf="models/tfidf_vectorizer_pystemmer.pkl",
                 filepath_tf_idf_with_rec="models/tfidf_vectorizer_with_text_rec_pystemmer.pkl",
                 filepath_w2v="models/w2v_model2.model"):
        self.dataset_handler = dataset_handler
        self.data_actual = dataset_handler.getActualDictSystemForModByName(mod_name)
        self.rec_dict = dataset_handler.rec_dict

        self.mod_name = mod_name
        self.mod_names = ["modsh", "mod1", "mod2"] # разрешённые названия модификаций

        self.methods = ["TFIDF-TDFIDF", "TFIDF-Word2Vec", "Word2Vec-TFIDF", "Word2Vec-Word2Vec", "RF-TFIDF", "RF-Word2Vec"]

        self.mod = dataset_handler.getModByName(mod_name)
        self.doc_names_mod = self.dataset_handler.getDocNamesInMods(mod_name)
        self.employeeRf = RF(doc_names_mod=self.doc_names_mod)

        self.employeeTFIDF = EmployeeTFIDF(doc_names_mod = self.doc_names_mod, mod_name=mod_name, filepath=filepath_tfidf, filepath_with_text=filepath_tf_idf_with_rec)
        self.moc2text = {
            "1": "Анализ Конструкции АТ, КД и ЭД",
            "2": "расчёт",
            "3": "Анализ отказобезопастности",
            "4": "Стендовые испытания",
            "5": "Наземные испытания",
            "6": "Лётные испытания",
            "7": "Моделирование(моделирующие стенды)",
            "8": "Одобрение комплектующих изделий",
            "9": "Обобщение опыта эксплуатации",
        }

    def set_mod(self, mod_name):
        if mod_name == self.mod_name:
            return
        elif mod_name not in self.mod_names:
            return
        self.data_actual = self.dataset_handler.getActualDictSystemForModByName(mod_name)
        self.mod = self.dataset_handler.getModByName(mod_name)
        self.doc_names_mod = self.dataset_handler.getDocNamesInMods(mod_name)
        self.employeeTFIDF.set_mod(mod_name=mod_name, doc_names_mod=self.doc_names_mod)
        self.employeeRf.set_mod(mod_name=mod_name, doc_names_mod=self.doc_names_mod)

    def predict(self, uid):
        return ["sdfsdf", "sdfsdfsdfff", "sdfsdfsdfsdf"]
    
    def predict(self, uidAWBReq, moc):
        
        feedBackData = self.getFeedbackFromSystem(uidAWBReq, moc)

        if len(feedBackData) == 0:
            return ["пока что не могу обработать"]

        feedBackNames = self.getNamesFromUids(feedBackData)

        uids = self.getRelevanceDocs(uidAWBReq, moc, feedBackNames)
        
        return uids
    
    def getRelevanceDocs(self, uidAWBReq, moc, feedBackNames, count_rel_docs=20): 
        systemName = self.dataset_handler.getSystemName(uidAWBReq)
        if not systemName:
            systemName = ""

        similarity_docs_tfidf = self.getSimilarytyDocsForFeedbackListTdIdf(feedBackNames)
        ranked_documents = self.getRelevanceVectorTFIDF(uidAWBReq, moc, systemName, similarity_docs_tfidf)
        ranked_documents_dict = self.getSortedRankedDocs(ranked_documents, similarity_docs_tfidf)
        uids = []
        for key1, value1 in itertools.islice(ranked_documents_dict.items(), count_rel_docs):
            uids += self.dataset_handler.getUidsByDocName(value1, self.mod)
        return uids
    
    def getNamesFromUids(self, uids):
        names = []
        for uid in uids:
            name = self.dataset_handler.getDocName(uid)
            name = name[str.find(name, ' ', 9) + 1:]
            names.append(name)
        return names
    
    def getFeedbackFromSystem(self, reqUID, moc, mod=None):
        if reqUID + ":" + moc in self.rec_dict:
            return self.rec_dict[reqUID + ":" + moc]
        else:
            return []
        
    def getSimilarytyDocsForFeedbackListTdIdf(self, feedback):
        similarity_docs = []
        count_similar_element = 15
        for doc in feedback:
            similarity_docs.append(self.employeeTFIDF.getSimilarytyDoc(doc_name=doc, count_rel_docs=count_similar_element))


        res_semilaryty_docs = []
        for i in range(count_similar_element):
            for docs in similarity_docs:
                if i < len(docs) :
                    res_semilaryty_docs.append(docs[i])
        
        return list(dict.fromkeys(res_semilaryty_docs))

    def getRelevanceVectorTFIDF(self, uidReq, moc, system_name, docsForEval):
        return self.employeeTFIDF.getRelevanceVector(textReq=self.dataset_handler.getReqText(reqUid = uidReq), system_name=system_name, moc_name=self.getMocName(moc), docsForEval=docsForEval)
        # return self.employeeTFIDF.getRelevanceVector(textReq="квазилинейность", system_name=system_name, moc_name=self.getMocName(moc), docsForEval=docsForEval)
    
    def getMocName(self, moc_num):
        if moc_num in self.moc2text:
            return self.moc2text[moc_num]
        else:
            return ""
        
    def getSortedRankedDocs(self, ranked_documents, similarity_docs):
        step = 5
        coeff = 1.1
        step_coeff = 0.02
        relevance_map = {}
        i = 0
        for rel, doc in zip(ranked_documents, similarity_docs):
            i+=1
            if (i % step == 0):
                coeff -= step_coeff
            relevance_map[rel*coeff] = doc
        sorted_map = dict(sorted(relevance_map.items(), reverse=True))
        return sorted_map
    
    def getUidsByDocName(self, docName, mod):
        uids = []
        for key, value in mod.items():
            if docName in str(value):
                uids.append(key)
        return uids