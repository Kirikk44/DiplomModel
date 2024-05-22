import os
import numpy as np
import pandas as pd

class DatasetHandler:
    def __init__(self):
        self.df_rec = None
        self.df_actual = None

        self.datasetReqToDocsWithName = None
        self.uidAwbRecToSystemName = None

        self.systems_dataset_mod1 = None
        self.systems_dataset_mod2 = None
        self.systems_dataset_modsh = None

        self.init_df_rec_actual()
        self.init_mods_uid_to_doc()
        self.init_datasetReqToDocsWithName()
        self.init_mods_uiditem_to_doc()
        self.init_systems_dataset()
        self.init_df_files_text_reqs()

    def init_df_files_text_reqs(self):
        df_text_reqs = pd.read_csv("datasets/dataset.csv", delimiter=";", on_bad_lines="skip")

        self.text_reqs_dict = df_text_reqs.set_index('req_name')['text'].to_dict()
        print("test init_df_files_text_reqs")

    def init_df_rec_actual(self):
        self.df_rec = pd.read_csv("datasets/datasetbaserec.csv", delimiter=";" , names=range(54), on_bad_lines="skip")
        self.df_actual = pd.read_csv("datasets/datasetbaseactual.csv", delimiter=";" , names=range(54), on_bad_lines="skip")

        self.rec_dict = {}

        for index, row in self.df_rec.iterrows():
            self.rec_dict[f"{row.iloc[0]}:{row.iloc[1]}"] = row.dropna().tolist()[2:]

    def init_mods_uid_to_doc(self):
        df_mod1 = pd.read_csv("datasets/mod1.csv", delimiter=";" , names=range(8), on_bad_lines="skip")
        df_mod2 = pd.read_csv("datasets/mod2.csv", delimiter=";" , names=range(8), on_bad_lines="skip")
        self.df_modsh = pd.read_csv("datasets/sh.csv", delimiter=";" , names=range(8), on_bad_lines="skip")

        self.mod1 = df_mod1.set_index(0)[1].to_dict()
        self.mod2 = df_mod2.set_index(0)[1].to_dict()
        self.modsh = self.df_modsh.set_index(0)[1].to_dict()

        self.mod1_docs_name = self.get_column_values(df_mod1)
        self.mod2_docs_name = self.get_column_values(df_mod2)
        self.modsh_docs_name = self.get_column_values(self.df_modsh)

        self.mods_names = {
            "mod1": self.mod1,
            "mod2": self.mod2,
            "modsh": self.modsh
        }

        self._mods_revision = [self.mod1, self.mod2, self.modsh]

    def getModByName(self, mod_name):
        return self.mods_names[mod_name]

    def init_datasetReqToDocsWithName(self):
        self.datasetReqToDocsWithName = pd.read_csv("datasets/datasetWithUidAWBReq.csv", delimiter=";" , names=range(40), on_bad_lines="skip")
        self.uidAwbRecToSystemName = self.datasetReqToDocsWithName.fillna("").set_index(0)[[1, 4]].apply(list, axis=1).to_dict() # подумать как преобразовать системык строке

    def init_mods_uiditem_to_doc(self):
        df_mod1_item = pd.read_csv("datasets/mod1_item.csv", delimiter=";" , names=range(8), on_bad_lines="skip")
        df_mod2_item = pd.read_csv("datasets/mod2_item.csv", delimiter=";" , names=range(8), on_bad_lines="skip")
        df_sh_item = pd.read_csv("datasets/sh_item.csv", delimiter=";" , names=range(8), on_bad_lines="skip")

        mod1_item = df_mod1_item.set_index(0)[1].to_dict()
        mod2_item = df_mod2_item.set_index(0)[1].to_dict()
        modsh_item = df_sh_item.set_index(0)[1].to_dict()

        self._mods = [mod1_item, mod2_item, modsh_item]

    def getDocName(self, uid):
        for mod in self._mods:
            if uid in mod.keys():
                return mod[uid]
        return "non"

    def getDocNamesInMods(self, mod_name):
        if mod_name == "mod1":
            return self.mod1_docs_name
        if mod_name == "mod2":
            return self.mod2_docs_name
        if mod_name == "modsh":
            return self.modsh_docs_name


    def getDocNameRevision(self, uid):
        for mod in self._mods_revision:
            if uid in mod.keys():
                return mod[uid]
        return "non"


    def init_systems_dataset(self):
        self.systems_dataset_mod1 = pd.read_csv("datasets/systemsdataset_mod1.csv", delimiter=";" , names=range(40), on_bad_lines="skip")
        self.systems_dataset_mod2 = pd.read_csv("datasets/systemsdataset_mod2.csv", delimiter=";" , names=range(40), on_bad_lines="skip")
        self.systems_dataset_modsh = pd.read_csv("datasets/systemsdataset_sh.csv", delimiter=";" , names=range(40), on_bad_lines="skip")

        uids_docs_mod1 = self.mod1.keys()
        uids_docs_mod2 = self.mod2.keys()
        uids_docs_modsh = self.modsh.keys()

        print("test")
        actual_dict_temp = {}
        self.actual_dict_mod1 = {}
        for index, row in self.systems_dataset_mod1.iterrows():
            actual_list = row.dropna().tolist()[3:]
            filtered_list = [x for x in actual_list if x in uids_docs_mod1]
            actual_dict_temp[f"{row.iloc[0]}:{row.iloc[1]}"] = filtered_list

        for key, value in actual_dict_temp.items():
            if not len(value):
                continue
            self.actual_dict_mod1[key] = value

        actual_dict_temp = {}
        for index, row in self.systems_dataset_mod2.iterrows():
            actual_list = row.dropna().tolist()[3:]
            filtered_list = [x for x in actual_list if x in uids_docs_mod2]
            actual_dict_temp[f"{row.iloc[0]}:{row.iloc[1]}"] = filtered_list

        self.actual_dict_mod2 = {}
        for key, value in actual_dict_temp.items():
            if not len(value):
                continue
            self.actual_dict_mod2[key] = value
        print("test2")
        actual_dict_temp = {}

        self.systems_dataset_modsh.iloc[:, 1] = self.systems_dataset_modsh.iloc[:, 1].astype(str)
        for index, row in self.systems_dataset_modsh.iterrows():
            actual_list = row.dropna().tolist()[3:]
            filtered_list = [x for x in actual_list if x in uids_docs_modsh]
            actual_dict_temp[f"{row.iloc[0]}:{row.iloc[1]}"] = filtered_list
        self.actual_dict_sh = {}
        for key, value in actual_dict_temp.items():
            if not len(value):
                continue
            self.actual_dict_sh[key] = value
        print("test3")

    def getActualDictSystemForModByName(self, mod_name):
        if mod_name == "mod1":
            return self.actual_dict_mod1
        if mod_name == "mod2":
            return self.actual_dict_mod2
        if mod_name == "modsh":
            return self.actual_dict_sh


    def getSystemName(self, uidAWBReq):
        if uidAWBReq not in self.uidAwbRecToSystemName.keys():
            return ""
        return self.uidAwbRecToSystemName[uidAWBReq][1]

    def getReqName(self, uidAwbRec):
        if uidAwbRec not in self.uidAwbRecToSystemName.keys():
            return ""
        return self.uidAwbRecToSystemName[uidAwbRec][0]

    def getReqText(self, reqName=None, reqUid=None):
        if reqName is not None:
            if reqName in self.text_reqs_dict:
                return self.text_reqs_dict[reqName]
            else:
                return ""
        elif reqUid is not None:
            reqName = self.getReqName(reqUid)
            if reqName in self.text_reqs_dict:
                return self.text_reqs_dict[reqName]
            else:
                return ""
        else:
            return ""

    def get_column_values(self, df):
        column_values = df.iloc[:, 1:].fillna('').values.flatten()

        column_values = column_values[column_values!='']

        column_values = np.array([s[str.find(s, ' ', 9) + 1:] for s in column_values])
        return column_values

    def getUidsByDocName(self, docName, mod):
        uids = []
        for key, value in mod.items():
            if docName in str(value):
                uids.append(key)
        return uids
