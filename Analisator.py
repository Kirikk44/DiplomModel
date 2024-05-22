class Analizator:
    def __init__(self, data_presolver, dataset_handler, predictor):
        self.data_presolver = data_presolver
        self.predictor = predictor

    def evaluate(self, actual_dict, start_index=0, end_index=-1, count_rel_docs=20):
        if end_index == -1:
            end_index = len(self.data_actual)

        for reqUidMoc, actual_docs_for_line in tqdm(itertools.islice(actual_dict.items(), start_index, end_index)):
            uidAWBReq = reqUidMoc[:-2]
            moc = reqUidMoc[-1]

            systemName = self.predictor.dataset_handler.getSystemName(uidAWBReq)
            if not len(systemName):
                continue

            feedBackData = self.predictor.getFeedbackFromSystem(uidAWBReq, moc)

            if len(feedBackData) == 0:
                continue

            feedBackNames = self.predictor.getNamesFromUids(feedBackData)

            similarity_docs_tfidf = self.predictor.getSimilarytyDocsForFeedbackListTdIdf(feedBackNames)
            similarity_docs_word2vec = self.predictor.getSimilarytyDocsForFeedbackListWord2Vec(feedBackNames)
            similarity_docs_RF = self.predictor.getRFForFeedbackList(uidAWBReq, moc, feedBackNames)

            for method in self.methods:
                if method == "TFIDF-TDFIDF":
                    ranked_documents = self.predictor.getRelevanceVectorTFIDF(uidAWBReq, moc, systemName, similarity_docs_tfidf)
                    ranked_documents_dict = self.predictor.getSortedRankedDocs(ranked_documents, similarity_docs_tfidf)
                elif method == "TFIDF-Word2Vec":
                    ranked_documents = self.predictor.getRelevanceVectorWord2Vec(uidAWBReq, moc, systemName, similarity_docs_tfidf)
                    ranked_documents_dict = self.predictor.getSortedRankedDocs(ranked_documents, similarity_docs_tfidf)
                elif method == "Word2Vec-TFIDF":
                    ranked_documents = self.predictor.getRelevanceVectorTFIDF(uidAWBReq, moc, systemName, similarity_docs_word2vec)
                    ranked_documents_dict = self.predictor.getSortedRankedDocs(ranked_documents, similarity_docs_word2vec)
                elif method == "Word2Vec-Word2Vec":
                    ranked_documents = self.predictor.getRelevanceVectorWord2Vec(uidAWBReq, moc, systemName, similarity_docs_word2vec)
                    ranked_documents_dict = self.predictor.getSortedRankedDocs(ranked_documents, similarity_docs_word2vec)
                elif method == "RF-TFIDF":
                    ranked_documents = self.predictor.getRelevanceVectorTFIDF(uidAWBReq, moc, systemName, similarity_docs_RF)
                    ranked_documents_dict = self.predictor.getSortedRankedDocs(ranked_documents, similarity_docs_RF)
                elif method == "RF-Word2Vec":
                    ranked_documents = self.predictor.getRelevanceVectorWord2Vec(uidAWBReq, moc, systemName, similarity_docs_RF)
                    ranked_documents_dict = self.predictor.getSortedRankedDocs(ranked_documents, similarity_docs_RF)

                uids = []
                for key1, value1 in itertools.islice(ranked_documents_dict.items(), count_rel_docs):
                    uids += self.dataset_handler.getUidsByDocName(value1, self.mod)

                if reqUidMoc not in self.data_presolver.method_classes_reqs[method]:
                    self.data_presolver.method_classes_reqs[method][reqUidMoc] = uids
                else:
                    print("Error: key already exists in self.method_classes_reqs[method]")

                if reqUidMoc not in self.data_presolver.method_classes_metrics[method]:
                    self.data_presolver.method_classes_metrics[method][reqUidMoc] = [TestMetrics.recall(predicted_relevant=uids, really_relevant=actual_docs_for_line),
                                                                 TestMetrics.precision(predicted_relevant=uids, really_relevant=actual_docs_for_line),
                                                                 TestMetrics.f1(predicted_relevant=uids, really_relevant=actual_docs_for_line)]

        for method in self.methods:
            for reqUidMoc, actual_docs_for_line in self.data_presolver.method_classes_metrics[method].items():
                self.data_presolver.average_metrics[method][0] += actual_docs_for_line[0]
                self.data_presolver.average_metrics[method][1] += actual_docs_for_line[1]
                self.data_presolver.average_metrics[method][2] += actual_docs_for_line[2]
        for method in self.methods:
            self.data_presolver.average_metrics[method][0] /= len(self.data_presolver.method_classes_metrics[method])
            self.data_presolver.average_metrics[method][1] /= len(self.data_presolver.method_classes_metrics[method])
            self.data_presolver.average_metrics[method][2] /= len(self.data_presolver.method_classes_metrics[method])
