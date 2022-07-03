import spacy
import re
import warnings
import copy
import ginza
import numpy as np
from collections import defaultdict
from collections import deque
nlp = spacy.load('ja_ginza')

class Node:
    def __init__(self, id, token, parent):
        # id: 文中の語のid
        # word: 単語(lemma)
        # parent: 親ノードを表現．(id, <relation>)の．例えば(0, case)
        # isRoot: 根ノードか
        self.id = id
        self.token = token
        self.parent = parent
        self.child = []
        self.isRoot = (parent is None)

    def add_child(self, id, relation):
        # 子ノード 情報の追加．表記方法はparentと同じ．
        self.child.append((id, relation))

class SentenceTree: # 一つの文のみを処理する木
    def __init__(self, sentence, speaker, talknumber=0):

        #前処理
        processed = self.preprocessing(sentence)
        self.processed_sentence = processed["sentence"]
        self.mention = processed["mention"]

        doc = nlp(self.processed_sentence)
        self.id_lemma_dict = dict() # idとlemma
        self.node_list = [] # ノードを格納するリスト
        self.speaker = speaker # 発言者
        self.talk_number = talknumber

        # ノードの登録
        for token in doc:
            self.id_lemma_dict[token.i] = token.lemma_
            idx = token.i
            lemma = token.lemma_

            if token.head.i == token.i: # ROOT
                parent = None
            else:
                parent = (token.head.i, token.dep_)
            self.node_list.append(Node(idx, lemma, parent))
        
        # 子ノード情報の登録
        for i in range(len(self.node_list)):
            if self.node_list[i].isRoot: # ROOTのインデックスを登録
                self.root_idx = i
            else: # ROOTでなければ親にアクセス
                (parent_id, relation) = self.node_list[i].parent
                self.node_list[parent_id].add_child(i, relation) # 子の追加
        # トークン数
        self.N = len(self.node_list)

        # 結果の確認
        """
        for i in range(self.N):
            print(self.node_list[i].id, self.node_list[i].token, self.node_list[i].child)             
        """

    def preprocessing(self, sentence):
        # メンションを検知し、取り除いてmentionにid(0～4)を格納
        mention = None
        if ">>Agent" in sentence:
            mention = int(sentence[9]) - 1
            sentence = sentence[12:]

        # Agent[01]を一郎に置換する
        # KNPで1単語として認識されないと色々面倒なので
        sentence = sentence.replace("Agent[01]", "一郎")
        sentence = sentence.replace("Agent[02]", "二郎")
        sentence = sentence.replace("Agent[03]", "三郎")
        sentence = sentence.replace("Agent[04]", "四郎")
        sentence = sentence.replace("Agent[05]", "五郎")

        return {"sentence": sentence, "mention": mention}

    def bfs(self, token_list, start_idx=None, include_root=True, ng_relation_list=[]):
        """
        input: token(捜索対象候補) start_idx(部分木の根，省略時はROOT) include_root(根も探索の対象か？)
        output: 存在すればid，しなければNone
        """
        if start_idx is None:
            start_idx = self.root_idx
        d = np.full(self.N, -1)
        q = deque()
        q.append(start_idx)
        d[start_idx] = 0

        if include_root and self.node_list[start_idx].token in token_list:
            return start_idx

        while len(q) != 0:
            n = q.popleft()
            for i in range(len(self.node_list[n].child)):
                child_id, child_relation =  self.node_list[n].child[i]
                if d[child_id] == -1:
                    if child_relation not in ng_relation_list:
                        q.append(child_id)
                    d[child_id] = d[n] + 1
                    if self.node_list[child_id].token in token_list and child_relation not in ng_relation_list:
                        return child_id
        return None

    def bfs_count(self, token_list, start_idx=None, include_root=True, ng_word_list=[], ng_relation_list=[]):
        """
        input: token(捜索対象候補) start_idx(部分木の根，省略時はROOT) include_root(根も探索の対象か？)
        output: 条件を満たす物の数を返す

        ng_word_list: これを見つけたら，その先は探索しない
        """ 
        ret = 0 
        if start_idx is None:
            start_idx = self.root_idx
        d = np.full(self.N, -1)
        q = deque()
        q.append(start_idx)
        d[start_idx] = 0

        if include_root and self.node_list[start_idx].token in token_list:
            ret += 1

        while len(q) != 0:
            n = q.popleft()
            for i in range(len(self.node_list[n].child)):
                child_id, relation =  self.node_list[n].child[i]
                if d[child_id] == -1:
                    if self.id_lemma_dict[child_id] not in ng_word_list and relation not in ng_relation_list:
                        q.append(child_id)
                    d[child_id] = d[n] + 1
                    if self.node_list[child_id].token in token_list and relation not in ng_relation_list:
                        ret += 1
        return ret

    def search_child(self, parent_idx, relation, token_list=[]):
        """
        特定の親ノードから，あるrelationで結ばれた子ノードの番号を探す
        token_listに何らかの指定があるときは，その中から探す．見つけ次第終了
        """
        for i in range(len(self.node_list[parent_idx].child)):
            child = self.node_list[parent_idx].child[i]
            if child[1] == relation:
                if (len(token_list)>0 and self.id_lemma_dict[child[0]] in token_list) or len(token_list)==0:
                    return child[0]
        return None

    def isCO(self):
        """
        a. 「n1はn2だ」型の文(e.g. 一郎は人狼だ，僕は占い師だ)
        b. 「n1はn2だと思う」型の文(e.g. 二郎は占い師だと思う)
        c. 「n1がn2だというのはXだ」型の文(e.g. 二郎が占い師だというのは嘘でない，一郎が人狼だというのは本当だ)
        d. 「n1がn2だというのはXだと思う」型の文(bとcの複合)
        """
    
        estimate_verb_list = ["思う", "考える", "推測"]
        false_word_list = ["誤り", "間違い", "嘘", "違い"]
        role_seer_list = ["占い師"]
        role_villager_list = ["村人", "むらびと", "一般人"]
        role_wolf_list = ["人狼", "狼", "黒"]
        role_possessed_list = ["狂人"]
        name_list = ["一郎", "二郎", "三郎", "四郎", "五郎"]
        other_name_list = name_list.copy()
        other_name_list.remove(self.speaker)
        first_person = ["私", "僕", "俺"]+[self.speaker]
        negative_aux_list = ["ぬ", "ない"]

        role_list = role_seer_list+role_villager_list+role_wolf_list+role_possessed_list
        person_list = name_list+first_person
        person_and_role = role_list+person_list              

        root_lemma = self.id_lemma_dict[self.root_idx]
        positive = 1

        # 「~だと思う」型の文では一人称を無視
        if root_lemma in estimate_verb_list:
            n2 = self.bfs(role_list+other_name_list)
            if n2 is not None:
                n1 = self.bfs(start_idx=n2,token_list=role_list+other_name_list, include_root=False, ng_relation_list=["acl"])
                if n1 is not None:
                    positive *= (-1)**self.bfs_count(negative_aux_list+false_word_list, ng_word_list=first_person, ng_relation_list=["acl", "nmod"])

        # 「~だと思う」を含まない場合
        else:
            n2 = self.bfs(person_and_role)
            if n2 is not None:
                n1 = self.bfs(start_idx=n2,token_list=person_and_role, include_root=False, ng_relation_list=["acl"])
                if n1 is not None:
                    positive *= (-1)**self.bfs_count(negative_aux_list+false_word_list, ng_relation_list=["acl", "nmod"])
        # if n2 is None:
        #     return not_target+": n2発見できず"
        # elif n1 is None:
        #     return not_target+": n1発見できず"
        if n2 is None or n1 is None:
            return None

        n1_token = self.id_lemma_dict[n1]
        n2_token = self.id_lemma_dict[n2]

        if (n1_token in role_list and n2_token in role_list) or            (n1_token in person_list and n2_token in person_list):
            # return not_target+": 無意味な文" # "私は一郎だ"，"占い師は白だ"などの無意味な文
            return None
    
        else:
            if n1_token in role_list:
                role_token = n1_token
                person_token = n2_token
            else:
                role_token = n2_token
                person_token = n1_token

            if role_token in role_seer_list:
                role = "SEER"
            elif role_token in role_villager_list:
                role = "VILLAGER"
            elif role_token in role_possessed_list:
                role = "POSSESSED"
            else:
                role = "WEREWOLF"

            if person_token in first_person:
                person = self.speaker
                conversation_type = "COMINGOUT"
            else:
                person = person_token
                conversation_type = "ESTIMATE"

            if positive == 1:
                positive = "POSITIVE"
            else:
                positive = "NEGATIVE"

            return(self.speaker, conversation_type, person, role, positive)
        
    def isVOTE(self):
        positive = 1

        vote_word_list = ["投票", "吊る", "入れる", "する"]
        negative_aux_list = ["ぬ", "ない"]
        name_list = ["一郎", "二郎", "三郎", "四郎", "五郎"]
        
        root = self.node_list[self.root_idx].token
        root_node = next(filter(lambda x: x.isRoot, self.node_list), None)
        
        child_list = []
        for i in range(len(root_node.child)):
            idx, relation = root_node.child[i]
            child_list.append(self.node_list[idx].token)
        
        child_vote = False
        for i in range(len(child_list)):
            if child_list[i] in vote_word_list:
                child_vote = True
        
        if root in vote_word_list or child_vote:
            if ("する" in self.id_lemma_dict.values()):
                if("投票" in self.id_lemma_dict.values()) == False:
                    #return not_target + ": 投票を表す文章ではない"
                    return None
                
            if root == "吊る":
                if self.root_idx == self.N-2:
                    #return not_target + ": 投票を表す文章ではない"
                    return None
                elif self.node_list[self.root_idx+1].token != "たい":
                    #return not_target + ": 投票を表す文章ではない"
                    return None
                    
            #人を探して、それを投票の対象と見なす
            target_id = self.bfs(name_list)
            if target_id is None:
                #return not_target + ": target発見できず"
                return None
            else:
                target = self.node_list[target_id].token

            for i in range(len(root_node.child)):
                idx, relation = root_node.child[i]
                if self.node_list[idx].token in negative_aux_list:
                    positive = 0
                    
            if root in negative_aux_list:
                positive = 0

            if positive:
                return(self.speaker, "VOTE", target, "POSITIVE")
            else:
                return(self.speaker, "VOTE", target, "NEGATIVE")
        
        else:
            #return not_target + ": 投票を表す文章ではない"
            return None
    
    def isDIVINATION(self):
        divine_word_list = ["占う", "占い", "調べる"]
        name_list = ["一郎", "二郎", "三郎", "四郎", "五郎"]

        # rootノード
        root_node = next(filter(lambda x: x.isRoot, self.node_list), None)

        if root_node.token in divine_word_list:
            # 占い対象を探す
            # objが優先だが、なければnsubj（三郎が人狼か占う など）
            divine_target = None
            divine_target = self.search_child(root_node.id, "obj", name_list)
            if divine_target is None:
                divine_target = self.search_child(root_node.id, "nsubj", name_list)

            if divine_target is not None:
                divine_target_name = self.node_list[divine_target].token
            else:
                divine_target_name = "ANY"
            
            return (self.speaker, "DIVINATION", divine_target_name)
        
        return None

    def isDIVINED(self):
        positive = 1

        result_human_list = ["村人", "むらびと", "一般人", "人", "人間", "白", "白い", "市民"]
        result_werewolf_list = ["人狼", "狼", "黒", "黒い"]
        name_list = ["一郎", "二郎", "三郎", "四郎", "五郎"]
        negative_aux_list = ["ぬ", "ない"]
        assertion_list = ["結果", "出る", "なる", "分かる"]

        result_list = result_human_list + result_werewolf_list

        # 発話者が占いCOしていないなら、占い文ではない
        # 例えば「Agent[03]は人狼です。」という発言は、占い師の発言なら占い結果として、そうでない人の発言ならESTIMATEとして扱うのが妥当だろう
        
        # CO_list的なものを実装してから反映させる
        # if !(CO_list includes (self.speaker, SEER)　的な何か) return not_target + ": 発話者が占い師でない"

        # rootノード
        root_node = next(filter(lambda x: x.isRoot, self.node_list), None)

        # 一般的な占い報告文は、rootが結果である
        if root_node.token in result_list or root_node.token in assertion_list:
            if root_node.token in result_human_list:
                species = "HUMAN"
            elif root_node.token in result_werewolf_list:
                species = "WEREWOLF"
            #rootが「結果」「出る」など断定の意味の場合（黒と出ました など）、そのchildからresult_listのワード（「黒」などを検出する）
            elif root_node.token in assertion_list:
                result_node = self.node_list[self.bfs(result_list)]
                if result_node.token in result_human_list:
                    species = "HUMAN"
                elif result_node.token in result_werewolf_list:
                    species = "WEREWOLF"

            #人を探して、それを占いの対象と見なす
            target_id = self.bfs(name_list)
            if target_id is None:
                #return not_target + ": target発見できず"
                return None
            else:
                target = self.node_list[target_id].token
            
            #否定語探し
            positive = 1
            for i in range(len(root_node.child)):
                idx, relation = root_node.child[i]
                if relation == "cop":
                    positive *= (-1)**self.bfs_count(negative_aux_list, start_idx=idx) # negative auxの探索
            
            #否定を反映
            if positive == -1:
                if species == "HUMAN":
                    species = "WEREWOLF"
                elif species == "WEREWOLF":
                    species = "HUMAN"

            return(self.speaker, "DIVINED", target, species)
        
        else:
            #return not_target + ": rootが占い結果を表す語でない"
            return None
    
    def isAGREE(self):
        
        positive = 1

        tri_agree = ["思う", "同意", "正しい", "賛成", "信ずる","信じる", "通り", "賛同"]     #加えて、caseか→fixedも
        tri_disagree = ["間違い", "間違う", "違う", "反対", "矛盾", "おかしい"]
        tri_all = tri_agree + tri_disagree
        obj_that = ["それ", "そう", "その"]
        obj_man = ["一郎","二郎", "三郎", "四郎", "五郎"]
        obj_all = obj_that + obj_man

        negative_aux = ["ない", "ぬ"]

        #objを発見する関数
        def searchObj(id,token):
            objcase = None
            obj1 = None
            if token in obj_that: #それ、とか
                objcase = "talk"
                obj1 = "SENTENCE" + str(self.talk_number-1)
            elif token in obj_man: #二郎、とか
                objcase = "player"
                obj1 = self.id_lemma_dict[id]
            elif token.isdecimal(): #talk_number
                objcase = "talk"
                obj1 = "SENTENCE" + str(token)
            else:
                for i in range(len(self.node_list[id].child)):
                    idx, relation = self.node_list[id].child[i]
                    idxlemma = self.id_lemma_dict[idx]
                    if relation == "nmod" or relation == "det":  #二郎の考え、の二郎を発見したい
                        objcase, obj1 = searchObj(idx, idxlemma)
                        
            return objcase, obj1

        #まず、トリガーを探す
        trigger1 = self.bfs(tri_all)
        if trigger1 is None:
            #return not_target + "：トリガーなし"
            return None
        else:
            tri_temp = trigger1
            #ccompによる補文を探す
            for i in range(len(self.node_list[trigger1].child)):
                idx, relation = self.node_list[trigger1].child[i]
                if relation == "ccomp" and self.id_lemma_dict[idx] in tri_all: #補文
                    trigger0 = trigger1
                    tri_temp = idx
                if relation == "aux" and self.id_lemma_dict[idx] in negative_aux: #くっついてる否定を検出
                    positive *= -1
            if tri_temp!=trigger1:
                trigger1 = tri_temp
                for i in range(len(self.node_list[trigger1].child)):
                    idx, relation = self.node_list[trigger1].child[i]
                    if relation == "ccomp" and self.id_lemma_dict[idx] in tri_all: #補文
                        trigger0 = trigger1
                        tri_temp = idx
                    if relation == "aux" and self.id_lemma_dict[idx] in negative_aux: #くっついてる否定を検出
                        positive *= -1
            
            #この時点で、trigger1に同意対象がくっついていると思われる
            tri1_token = self.id_lemma_dict[trigger1]

            objcase = None
            obj1 = None

            if tri1_token=="賛成" or tri1_token=="同意" or tri1_token=="賛同":
                for i in range(len(self.node_list[trigger1].child)):
                    idx, relation = self.node_list[trigger1].child[i]
                    idxlemma = self.id_lemma_dict[idx]
                    if relation == "obl":  #目的語発見
                        objcase, obj1 = searchObj(idx, idxlemma)
            elif tri1_token=="反対":
                positive *= -1
                for i in range(len(self.node_list[trigger1].child)):
                    idx, relation = self.node_list[trigger1].child[i]
                    idxlemma = self.id_lemma_dict[idx]
                    if relation == "obl":  #目的語発見
                        objcase, obj1 = searchObj(idx, idxlemma)
            elif tri1_token=="思う" or tri1_token=="信ずる" or tri1_token=="信じる":
                for i in range(len(self.node_list[trigger1].child)):
                    idx, relation = self.node_list[trigger1].child[i]
                    idxlemma = self.id_lemma_dict[idx]
                    if relation == "advmod" or relation == "obj":  #目的語発見
                        objcase, obj1 = searchObj(idx, idxlemma)
                    # if relation == "nsubj":  #目的語発見
                    #     objcase, obj1 = searchObj(idx, idxlemma)  #オプション、二郎の考えを信じる、とか。ただし「俺は」が検出されちゃうかも
            elif tri1_token=="正しい":  
                for i in range(len(self.node_list[trigger1].child)):
                    idx, relation = self.node_list[trigger1].child[i]
                    idxlemma = self.id_lemma_dict[idx]
                    if relation == "nsubj":  #目的語発見
                        objcase, obj1 = searchObj(idx, idxlemma)
            elif tri1_token=="間違い" or tri1_token == "間違う" or tri1_token == "違う" or tri1_token == "矛盾" or tri1_token == "おかしい":  
                positive *= -1
                for i in range(len(self.node_list[trigger1].child)):
                    idx, relation = self.node_list[trigger1].child[i]
                    idxlemma = self.id_lemma_dict[idx]
                    if relation == "nsubj":  #目的語発見
                        objcase, obj1 = searchObj(idx, idxlemma)
            elif tri1_token=="通り":  
                for i in range(len(self.node_list[trigger1].child)):
                    idx, relation = self.node_list[trigger1].child[i]
                    idxlemma = self.id_lemma_dict[idx]
                    if relation == "det" or relation == "acl":  #目的語発見
                        objcase, obj1 = searchObj(idx, idxlemma)
            

            if objcase is None or obj1 is None:
                #return not_target+"：目的語発見できず"
                return None
            elif positive==1:
                return (self.speaker, "AGREE", obj1)
            else:
                return (self.speaker, "DISAGREE", obj1)

    def isREQUEST(self):

        action_list = ["占う", "調べる", "入れる", "吊る"]
        name_list = ["一郎", "二郎", "三郎", "四郎", "五郎"]
        job_list = {"占い師": "SEER", "占い": "SEER", "狂人": "POSESSED"}
        name_and_job_list = name_list + list(job_list.keys())

        # これに加え、「占って」のような「て」で終わっているのもrequestとなる
        request_aux_list = ["くれる", "もらう", "ほしい", "くださる"]

        # 「て」を見つけ、それが文末である（後ろに「。」があるかもなのでidは末尾or末尾-1）か、そのchildがrequest_aux_listの補助動詞なら、「て」の親をmain_verbとする
        main_verb_id = None
        te_id = self.bfs(["て"])
        if te_id is not None:
            if self.search_child(te_id, "fixed", request_aux_list) or (len(self.node_list) == te_id+1 or len(self.node_list) == te_id+2):
                main_verb_id = self.node_list[te_id].parent[0]
        if main_verb_id is not None:

            name_request_to = ""

            copied_sentencetree = copy.deepcopy(self)

            if self.mention is None:
                request_to = None
                request_to = self.search_child(main_verb_id, 'obl', name_and_job_list)
                if request_to is None:
                    request_to = self.search_child(main_verb_id, 'nsubj', name_and_job_list)
                if request_to is not None:
                    # 検出した主語（request_to）以下の部分と、mark（「て」）以下の部分を消去する
                    # main_verb[id].childからrequest_toとmarkを除外する
                    # rootとmain_verbが異なるなら、元root以降は消去する。main_verbをrootに。
                    copied_sentencetree.node_list[main_verb_id].child = list(filter(lambda x: x[0] != request_to and x[0] != te_id, self.node_list[main_verb_id].child))
                    if copied_sentencetree.node_list[main_verb_id].parent is not None:
                        old_root_node = next(filter(lambda x: x.isRoot, self.node_list), None)
                        copied_sentencetree.node_list = copied_sentencetree.node_list[:old_root_node.id] #old_root_node以降を削除
                        copied_sentencetree.node_list[main_verb_id].parent = None
                        copied_sentencetree.node_list[main_verb_id].isRoot = True
                        copied_sentencetree.root_idx = main_verb_id
                    name_request_to = self.node_list[request_to].token
                else:
                    # request_toがないなら、全員に呼びかけていると判断　mark(「て」)以下の部分を消去する 
                    copied_sentencetree.node_list[main_verb_id].child = list(filter(lambda x: x[0] != te_id, self.node_list[main_verb_id].child))
                    if copied_sentencetree.node_list[main_verb_id].parent is not None:
                        old_root_node = next(filter(lambda x: x.isRoot, self.node_list), None)
                        copied_sentencetree.node_list = copied_sentencetree.node_list[:old_root_node.id] #old_root_node以降を削除
                        copied_sentencetree.node_list[main_verb_id].parent = None
                        copied_sentencetree.node_list[main_verb_id].isRoot = True
                        copied_sentencetree.root_idx = main_verb_id
                    name_request_to = "ANY"
            else:
                # mentionがあるなら、「て」以下のみ消去する
                copied_sentencetree.node_list[main_verb_id].child = list(filter(lambda x: x[0] != te_id, self.node_list[main_verb_id].child))
                if copied_sentencetree.node_list[main_verb_id].parent is not None:
                    old_root_node = next(filter(lambda x: x.isRoot, self.node_list), None)
                    copied_sentencetree.node_list = copied_sentencetree.node_list[:old_root_node.id] #old_root_node以降を削除
                    copied_sentencetree.node_list[main_verb_id].parent = None
                    copied_sentencetree.node_list[main_verb_id].isRoot = True
                    copied_sentencetree.root_idx = main_verb_id
                name_request_to = name_list[self.mention]

            # name_request_to　が「占い師」なら、英語のSEERに変換する
            if name_request_to in job_list.keys():
                name_request_to = job_list[name_request_to]
            
            # TODO:: name_request_toには、「一郎」「二郎」などのname_listに加え、「占い師」などのjob_listの語も含まれる。
            # 占いCOした人のリストなどを参照し、job_listが「占い師」ならばそれをCOした人の名前に変換する

            request_contents = copied_sentencetree.analyze_sentence()
            if request_contents:
                return (self.speaker, "REQUEST", name_request_to, name_request_to) + request_contents[1:]
            else:
                return None

        return None

        # #「動詞」+「て」+「特定の補助動詞（なし も含む）」を検出する
        # te_id = self.search_child(root_node.id, "mark") #「て」のid
        # if te_id and self.node_list[te_id].token == "て":
        #     # request_aux_listに該当するchildがある または 「て」で終わっている
        #     if self.search_child(te_id, "fixed", request_aux_list) or (len(self.node_list[te_id].child) == 0):
        #         #mentionがなければ、request先を検出する
        #         #nsubj（（人）は） or obl（（人）には）を検出
        #         #最終的にname_request_toに、"一郎"～"五郎"のどれかを格納する
        #         name_request_to = ""

        #         copied_sentencetree = copy.deepcopy(self)

        #         if self.mention is None:
        #             request_to = None
        #             request_to = self.search_child(root_node.id, 'obl', name_and_job_list)
        #             if request_to is None:
        #                 request_to = self.search_child(root_node.id, 'nsubj', name_and_job_list)
        #             if request_to is not None:
        #                 # 検出した主語（request_to）以下の部分と、mark（「て」）以下の部分を消去する
        #                 # root_node.childからrequest_toとmarkを除外する
        #                 copied_sentencetree.node_list[root_node.id].child = list(filter(lambda x: x[0] != request_to and x[0] != te_id, root_node.child))
        #                 name_request_to = self.node_list[request_to].token
        #             else:
        #                 # request_toがないなら、全員に呼びかけていると判断　mark(「て」)以下の部分を消去する 
        #                 copied_sentencetree.node_list[root_node.id].child = list(filter(lambda x: x[0] != te_id, root_node.child))
        #                 name_request_to = "ANY"
        #         else:
        #             # mentionがあるなら、「て」以下のみ消去する
        #             copied_sentencetree.node_list[root_node.id].child = list(filter(lambda x: x[0] != te_id, root_node.child))
        #             name_request_to = name_list[self.mention -1] #1-5を0-4に
                
        #         # TODO:: name_request_toには、「一郎」「二郎」などのname_listに加え、「占い師」などのjob_listの語も含まれる。
        #         # 占いCOした人のリストなどを参照し、job_listが「占い師」ならばそれをCOした人の名前に変換する

        #         # 残った木に対して分類を実行し、speaker REQUEST request_to （analyazeの返り値）の形で返す
                
        #         request_contents = copied_sentencetree.analyze_sentence()
        #         if request_contents:
        #             return (self.speaker, "REQUEST", name_request_to, name_request_to) + request_contents[1:]
        #         else:
        #             return None

    def isINQUIRE(self):

        #ToDoList: talk_numberを統合に組み込む、processed_sentenceをselfに、mentionなしは誰向けと判断？、「誰を怪しいと思ってる？」、REQUEST候補あり、疑問詞二つは非対応、
        # 「○○はなんだと思ってる？」がうまくいかない、「誰を占いましたか？」はDIVINE？
        #「誰に投票したらいいと思う？」がVOTEDに、わからないときは特定のエラー吐いた方がよい？
        #クラス内で自身のオブジェクト作っていいの...？自身について破壊的操作？
        #Because対応はする？、狂人追加
        inquire_trigger = ["何","なに","どう","なぜ","なん","だれ","誰","何故"]
        question_mark = ["?","？"]
        think_exp = ["と思う", "と思わない", "と思いませんか"]
        black_term = ["怪しい", "あやしい", "人狼", "狼", "黒い"]
        white_term = ["白い", "人間", "村"]
        role_term = ["人狼","占い師","村人"]
        player_term = ["一郎", "二郎", "三郎", "四郎", "五郎"]
        self_term = ["俺","僕","ぼく","私","わたし","おれ","あたくし"]
        you_term = ["お前","君","きみ","あなた"]
        
        interro_idx = self.bfs(inquire_trigger)
        question_idx = self.bfs(question_mark)

        #疑問詞があればANYタイプのINQUIRE
        if interro_idx is not None:
            interrogative = self.id_lemma_dict[interro_idx]
            
            #「占いはどうだった？」
            if interrogative=="何" or interrogative=="なに" or interrogative=="どう" or interrogative=="なん":
                obj = self.bfs(["占い","占う"], start_idx=interro_idx)
                if obj is not None:
                    return (self.speaker, "INQUIRE", player_term[self.mention] or "ANY", "DIVINED", "ANY")

            #「誰が怪しいと思う？」
            if interrogative=="誰" or interrogative=="だれ":
                who_parent_idx = self.node_list[interro_idx].parent[0]
                if who_parent_idx is not None:
                    if self.id_lemma_dict[who_parent_idx] in black_term:
                        return (self.speaker, "INQUIRE", player_term[self.mention] or "ANY", "ESTIMATE", "ANY", "WEREWOLF")

            #「二郎の役職はなんだと思う？」「あなたの役職はなんですか？」
            if interrogative=="何" or interrogative=="なに" or interrogative=="なん":
                for i in range(len(self.node_list[interro_idx].child)):
                    idx, relation = self.node_list[interro_idx].child[i]
                    idx_lemma = self.id_lemma_dict[idx]
                    if idx_lemma=="役職":
                        for i in range(len(self.node_list[idx].child)):
                            idx_yaku, relation_yaku = self.node_list[idx].child[i]
                            yaku_lemma = self.id_lemma_dict[idx_yaku]
                            if yaku_lemma in player_term:
                                return (self.speaker, "INQUIRE", player_term[self.mention] or "ANY", "ESTIMATE", yaku_lemma, "ANY")
                            if yaku_lemma in self_term:
                                return (self.speaker, "INQUIRE", player_term[self.mention] or "ANY", "ESTIMATE", self.speaker, "ANY")
                        return (self.speaker, "INQUIRE", player_term[self.mention] or "ANY", "COMINGOUT")             #ここREQUESTにする？

            #「君はどう思う？」
            if interrogative=="どう":
                how_parent_idx = self.node_list[interro_idx].parent[0]
                if how_parent_idx is not None:
                    if self.id_lemma_dict[how_parent_idx] == "思う":
                        #○○についてどう思う？はめんどくさいので、それ以外を検出
                        if self.search_child(how_parent_idx, relation="obl") is None:
                            return (self.speaker, "INQUIRE", player_term[self.mention] or "ANY", "THINK", "ANY")

            #「誰を占うといいと思う？」
            if interrogative=="誰" or interrogative=="だれ":
                see_parent_idx, see_relation = self.node_list[interro_idx].parent
                see_parent_lemma = self.id_lemma_dict[see_parent_idx]
                if see_relation=="obj" and see_parent_lemma=="占う":
                    return (self.speaker, "INQUIRE", player_term[self.mention] or "ANY", "THINK", "DIVINE", "ANY")

            #「誰に投票するつもり？」「昨日はだれに投票したの？」
            if interrogative=="誰" or interrogative=="だれ":
                vote_parent_idx, vote_relation = self.node_list[interro_idx].parent
                vote_parent_lemma = self.id_lemma_dict[vote_parent_idx]
                if vote_relation=="obl" and (vote_parent_lemma=="投票" or vote_parent_lemma=="入れる" or vote_parent_lemma=="いれる"):
                    #過去形を検出
                    for i in range(len(self.node_list[vote_parent_idx].child)):
                        idx, relation = self.node_list[vote_parent_idx].child[i]
                        if self.id_lemma_dict[idx]=="た":
                            return (self.speaker, "INQUIRE", player_term[self.mention] or "ANY", "VOTED", "ANY")
                    return (self.speaker, "INQUIRE", player_term[self.mention] or "ANY", "VOTE", "ANY")

            return None

        #?があればYes/NoタイプのINQUIRE
        elif question_idx is not None:

            #「～と思う？」「と思わない？」タイプは、それ以前を抜粋して新たに分析
            inside_sentence = self.processed_sentence
            send_again = 0
            for think in think_exp:
                if think in self.processed_sentence:
                    inside_sentence = inside_sentence.replace('?', '')
                    inside_sentence = inside_sentence.replace('？', '')
                    inside_sentence = inside_sentence.replace(think, '')
                    send_again = 1
                    break
            #クラス内で自身のオブジェクト作っていいの...？自身について破壊的操作？
                
            #「～と思う」以外で役職確認、「二郎は人狼に見える？」
            role_object = self.bfs(role_term+black_term+white_term)
            if role_object is not None:
                role_lemma = self.id_lemma_dict[role_object]
                for i in range(len(self.node_list[role_object].child)):
                    subj_idx, subj_relation = self.node_list[role_object].child[i]
                    subj_lemma = self.id_lemma_dict[subj_idx]
                    if subj_lemma in self_term and subj_relation=="nsubj":
                        if role_lemma=="人狼" or role_lemma in black_term:
                            return (self.speaker, "INQUIRE", player_term[self.mention] or "ANY", "ESTIMATE", self.speaker, "WEREWOLF")
                        elif role_lemma=="占い師":
                            return (self.speaker, "INQUIRE", player_term[self.mention] or "ANY", "ESTIMATE", self.speaker, "SEER")
                        elif role_lemma=="村人":
                            return (self.speaker, "INQUIRE", player_term[self.mention] or "ANY", "ESTIMATE", self.speaker, "VILLAGER")
                        elif role_lemma in white_term:
                            return (self.speaker, "INQUIRE", player_term[self.mention] or "ANY", "ESTIMATE", self.speaker, "HUMAN")
                    elif subj_lemma in you_term and subj_relation=="nsubj":
                        if role_lemma=="人狼" or role_lemma in black_term:
                            return (self.speaker, "INQUIRE", player_term[self.mention] or "ANY", "ESTIMATE", player_term[self.mention], "WEREWOLF")
                        elif role_lemma=="占い師":
                            return (self.speaker, "INQUIRE", player_term[self.mention] or "ANY", "ESTIMATE", player_term[self.mention], "SEER")
                        elif role_lemma=="村人":
                            return (self.speaker, "INQUIRE", player_term[self.mention] or "ANY", "ESTIMATE", player_term[self.mention], "VILLAGER")
                    elif subj_lemma in player_term and subj_relation=="nsubj":
                        if role_lemma=="人狼" or role_lemma in black_term:
                            return (self.speaker, "INQUIRE", player_term[self.mention] or "ANY", "ESTIMATE", subj_lemma, "WEREWOLF")
                        elif role_lemma=="占い師":
                            return (self.speaker, "INQUIRE", player_term[self.mention] or "ANY", "ESTIMATE", subj_lemma, "SEER")
                        elif role_lemma=="村人":
                            return (self.speaker, "INQUIRE", player_term[self.mention] or "ANY", "ESTIMATE", subj_lemma, "VILLAGER")
                        elif role_lemma in white_term:
                            return (self.speaker, "INQUIRE", player_term[self.mention] or "ANY", "ESTIMATE", subj_lemma, "HUMAN")

            return None
        
        #疑問形ではない
        else:
            return None

    def analyze_sentence(self):
        result_isDIVINED = self.isDIVINED()
        result_isDIVINATION = self.isDIVINATION()
        result_isCO = self.isCO()
        result_isVOTE = self.isVOTE()
        result_isAGREE = self.isAGREE()
        result_isREQUEST = self.isREQUEST()
        result_isINQUIRE = self.isINQUIRE()

        if result_isINQUIRE:
            return self.postprocessing(result_isINQUIRE)
        if result_isREQUEST:
            return self.postprocessing(result_isREQUEST)
        if result_isDIVINED:
            return self.postprocessing(result_isDIVINED)
        if result_isDIVINATION:
            return self.postprocessing(result_isDIVINATION)
        if result_isCO:
            return self.postprocessing(result_isCO)
        if result_isVOTE:
            return self.postprocessing(result_isVOTE)
        if result_isAGREE:
            return self.postprocessing(result_isAGREE)
    
        return None
    
    def postprocessing(self, analyzed_result):
        # 一郎、二郎…をAgent[01]～Agent[05]に戻す関数
        # ("一郎", "DIVINED", "三郎", "WEREWOLF") -> (0, "DIVINNED", 2, "WEREWOLF")
        processed = []

        old_name_list = ["一郎", "二郎", "三郎", "四郎", "五郎"]
        new_name_list = ["Agent[01]", "Agent[02]", "Agent[03]", "Agent[04]", "Agent[05]"]
        for word in analyzed_result:
            if word in old_name_list:
                processed.append(new_name_list[old_name_list.index(word)])
            else:
                processed.append(word)
        return tuple(processed)

if __name__ == "__main__":
    sentence_list = [
        "Agent[03]を占った結果、白でした。",
        "Agent[03]は人間です。",
        "今日の占い結果は、Agent[03]が黒です。",
        "Agent[03]の占い結果は人狼です。",
        "Agent[03]は人狼ではありませんでした。",
        "Agent[03]が昨日怪しかったけど、村人でした。",
        "占ったところAgent[03]は人狼なんかじゃなかった。",
        "Agent[03]は黒という結果でした",
        "Agent[03]を占ったところ、人間と出ました",
        "Agent[03]が人狼だと思います。",
        "個人的にはAgent[03]は村人だと信じてる。",
        "今夜はAgent[03]を占いたい。",
        "占い師にはAgent[03]が村人か調べてほしい。",
        "Agent[03]が人狼か占ってほしい。",
        "僕は占い師です",
        "僕は人狼ではありません",
        "二郎が占い師だというのは嘘だ",
        "二郎が占い師だというのは嘘で、僕が占い師です",
        "二郎は狂人だと思う",
        "僕は三郎が人狼だと思う",
        "一郎は村人です",
        "僕は一郎です",
        "二郎が村人だというのは確か",
        "僕の役職は占い師です",
        "僕の役職は狂人ではなく村人です",
        "二郎ではなく三郎が村人です",
        "三郎の役職は狂人だと僕は思う",
        "俺は三郎が村人だって信じてるよ", # 失敗
        "三郎が村人だと僕は信じているよ",
        "僕が思うに二郎は占い師ではない",
        "二郎の発言には矛盾があるから、二郎は狂人に違いない",
        "僕は村人だって信じてくれよ", # 失敗
        "占い師じゃない僕は二郎が占い師だというのは嘘だと思うよ", # なぜか失敗
        "占い師じゃない僕は二郎が占い師でないというのは嘘だと思うよ",
        "占い師じゃない僕は二郎が占い師でないというのは嘘でないと思うよ",
        "占い師じゃない僕は二郎が占い師でないというのは嘘でないと思わないよ",
        "二郎は占い師ではなく人狼だ",
        "Agent[03]に投票します。",
        "今日はAgent[03]に入れます。",
        "Agent[03]を吊りたいです。",
        "投票先はAgent[03]にします。",
        "Agent[03]には投票していないです。",
        "Agent[03]は吊りたくない。",
        "Agent[03]を吊ろうと思っている。",
        "Agent[03]の言う通りだと思う。",
        "Agent[03]に賛成です。",
        "Agent[03]の発言は嘘だと思う。",
        "Agent[03]の言っていることは正しい。",
        "Agent[03]は嘘をついている。",
        "僕はAgent[03]を信じます。",
        "Agent[03]の考えに同意。",
        "確かにAgent[03]の発言は矛盾しているね。",
        "今日はAgent[04]を占ってみたい。",
        "Agent[03]には今夜はAgent[04]を占ってほしい。",
        "占い師はAgent[04]を占って。",
        ">>Agent[03] Agent[04]を占ってくれ。",
        ">>Agent[03] Agent[04]を調べてほしいと思ってる。",
        ">>Agent[03] Agent[04]を占ってほしい気持ちです。",
        ">>Agent[03] 占いはどうだった？",
        ">>Agent[03] 占いの結果は何？",
        ">>Agent[03] 誰が人狼だと思う？",
        ">>Agent[03] 誰が占い師だと思ってる？",
        ">>Agent[03] お前は誰が怪しい？",
        ">>Agent[03] あなたの役職は何ですか。",
        ">>Agent[03] 四郎の役職は何だと思う？",
        ">>Agent[03] 俺は怪しく見える？",
        ">>Agent[03] 君はどう思う？",
        ">>Agent[03] 昨日は誰を占ったの？",
        ">>Agent[03] 昨日は誰に投票した？",
        ">>Agent[03] 今日は誰に入れればいいですか",
        ">>Agent[03] なぜ",
        ">>Agent[03] どうしてそう思うの？",
        ">>Agent[03] 根拠は？",
        ">>Agent[03] 君もそう思わない？",
        ">>Agent[03] お前が人狼か？",
        ">>Agent[03] 三郎が占い師ってコト!？",
        "完全に同意",
        "その通りだね。"
    ]

    for sentence in sentence_list:
        tree = SentenceTree(sentence, speaker="一郎", talknumber=100)
        print(sentence)
        print(tree.analyze_sentence(), "\n")
        # for node in tree.node_list:
        #     print(str(node.id) + " " + node.token +  " " + str(node.parent) + " " + str(node.child))
            