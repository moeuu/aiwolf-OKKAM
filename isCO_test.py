from turtle import pos
import spacy
import re
import warnings
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
    def __init__(self, sentence, speaker):
        doc = nlp(sentence)
        self.id_lemma_dict = dict() # idとlemma
        self.node_list = [] # ノードを格納するリスト
        self.speaker = speaker # 発言者

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



    def bfs(self, token_list, start_idx=None, include_root=True):
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
                child_id, _ =  self.node_list[n].child[i]
                if d[child_id] == -1:
                    q.append(child_id)
                    d[child_id] = d[n] + 1
                    if self.node_list[child_id].token in token_list:
                        return child_id
        return None

    def bfs_count(self, token_list, start_idx=None, include_root=True):
        """
        input: token(捜索対象候補) start_idx(部分木の根，省略時はROOT) include_root(根も探索の対象か？)
        output: 条件を満たす物の数を返す
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
                child_id, _ =  self.node_list[n].child[i]
                if d[child_id] == -1:
                    q.append(child_id)
                    d[child_id] = d[n] + 1
                    if self.node_list[child_id].token in token_list:
                        ret += 1
        return ret

    def isDIVINE(self):
        positive = 1
        not_target = "not DIVINE"

        divine_word_list = ["占う", "占い", "調べる"]
        result_human_list = ["村人", "むらびと", "一般人", "人", "人間", "白", "白い", "市民"]
        result_werewolf_list = ["人狼", "狼", "黒", "黒い"]
        negative_aux_list = ["ぬ", "ない"]

        result_list = result_human_list + result_werewolf_list

        # 発話者が占いCOしていないなら、占い文ではない
        # 例えば「Agent[03]は人狼です。」という発言は、占い師の発言なら占い結果として、そうでない人の発言ならESTIMATEとして扱うのが妥当だろう
        
        # CO_list的なものを実装してから反映させる
        # if !(CO_list includes (self.speaker, SEER)　的な何か) return not_target + ": 発話者が占い師でない"


        # rootノード
        root_node = next(filter(lambda x: x.isRoot, self.node_list), None)

        # 一般的な占い報告文は、rootが結果である
        # 未解決：「Agent[03]は黒という結果でした」等は「結果」がrootになり失敗する。どうするか
        # rootが推量系ならESTIMATE、断定系や「結果」ならDIVINEとする？
        if root_node.token in result_list:

            if root_node.token in result_human_list:
                species = "HUMAN"
            elif root_node.token in result_werewolf_list:
                species = "WEREWOLF"

            #人を探して、それを占いの対象と見なす
            target_id = self.bfs(["agent"])
            if target_id is None:
                return not_target + ": target発見できず"
            else:
                #"agent" "[" "03" "]"のように分かれて取得されるので、つなげて"Agent03"とする
                target = self.node_list[target_id].token + self.node_list[target_id+2].token

            
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
            return not_target + ": rootが占い結果を表す語でない"

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
    "Agent[03]が人狼か占ってほしい。"
]

for i in range(len(sentence_list)):
    tree = SentenceTree(sentence_list[i], speaker="Agent[01]")
    print(sentence_list[i])
    print(tree.isDIVINE(), "\n")
    # for node in tree.node_list:
    #     print(str(node.id) + " " + node.token +  " " + str(node.parent) + " " + str(node.child))