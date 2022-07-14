#from asyncio.windows_events import None
#from distutils.log import error
from fileinput import filename
#from datetime import date
from gc import get_freeze_count
from hashlib import scrypt
from multiprocessing.spawn import old_main_modules
from reprlib import recursive_repr
import shutil
from tkinter import BooleanVar
from tracemalloc import start
from types import NoneType
from weakref import WeakKeyDictionary
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.by import By
from selenium.webdriver.common.action_chains import ActionChains
from selenium.webdriver.support.select import Select

from selenium.webdriver.chrome import service as fs
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
#%matplotlib inline
import japanize_matplotlib
from operator import itemgetter

from collections import defaultdict

from janome.tokenizer import Tokenizer
from pyknp import Juman

from bs4 import BeautifulSoup
import time
import datetime as dtime
import sys
import joblib
import pickle
import os
import io
from PIL import Image
import uuid
import re
import pandas as pd
import numpy as np
from urllib import request
import requests
import json
import glob
import random
import atexit
import copy
import csv
import gensim
import MeCab

import General as g

from dataclasses import dataclass


debug = False

re_hiragana = "\u3041-\u309F"
re_katakana1 = "\u30A1-\u30FF"
re_katakana2 = "\uFF66-\uFF9F"
re_kanji = "\u2E80-\u2FDF\u3005-\u3007\u3400-\u4DBF\u4E00-\u9FFF\uF900-\uFAFF\U00020000-\U0002EBEF"
re_kigou = "\u0020-\u002F\u003A-\u0040\u005B-\u0060\u007B-\u007E"
re_num1 = "0-9"
re_num2 = "０-９"
re_alfa = "a-zA-Zａ-ｚＡ-Ｚ"
re_http = "https?://[\w!\?/\+\-_~=;\.,\*&@#\$%\(\)'\[\]]+"
re_kigou_a = "`|,|.|_|^|\＾|ﾉ|;|；|/|／|:|：|ゝ|*|ヾ|\"|Ｏ|\\\|+|＋|⁺|◟|·̫| |ᵒ̴̶̷̥́| | |.|｡ﾟ|+|.|-|`|、|◜|ω|◝| ᷇|࿀| ᷆| |*|´|◒|`|*|◉|‿|◉|٩|ˊ|ᗜ|ˋ|*|و|ง| |•̀|_|•́|ง|▽|°|:|3|ง| |•̀|_|•́|ง|▿|⁰|*|๑|╹|ω|∩|´|∀|｀|•́|︿|•̀|｡|³|ω|³|＠|・|o"
re_kigou_b = "!|-|?|"


CHROMEDRIVERPATH = "/usr/bin/chromedriver"


# === 構造体 ===
# ツイッターにログインするためのデータ
@dataclass
class FTwitterLoginData:
    id: str
    pw: str

# === クラス ===

# 自然言語処理
class NLProcessing:
    tokenizer = Tokenizer()

    def __init__(self):
        pass

    # ネガポジ辞書データの取得
    def GetNPDict(self):
        # すでにある場合はそれを取得する
        np_dic = g.LoadData("damp", "np_dic")
        if np_dic:
            return np_dic

        # ない場合は作成して保存する
        np_dic = {}
        #
        fp = open("./../pn.csv", "rt", encoding="utf-8")
        #
        reader = csv.reader(fp, delimiter='\t')
        #
        for i, row in enumerate(reader):
            name = row[0]
            result = row[1]
            #print(name)
            #print(result)
            np_dic[name] = result
            #if i % 1000 == 0: print(i)
        #
        g.SaveData(np_dic, "damp", "np_dic")
        print("ok")

        return np_dic

    # 文章に対してネガポジ分析を行う
    def GetEasyNPData(self, input_str):
        pos_count = 0
        neg_count = 0
        word_count = 0
        tokens = self.tokenizer.tokenize(input_str)

        np_dic = self.GetNPDict()
        for token in tokens:
            base_form = token.base_form # 原型 / 基本形
            # ネガポジ辞書に存在するか確認して対応する方を１増やす
            if base_form in np_dic:
                # 単語を辞書のキーとして、そのバリューがpかnか確認する
                if np_dic[base_form] == "p":
                    pos_coutn += 1
                    # どんな言葉がポジ版tネイされているか確認用（あとでコメントアウト）
                    print("POS:" + base_form)
                if np_dic[base_form] == "n":
                    neg_count += 1
                    print("NEG:" + base_form)
            # 存在しようがしまいが、単語数を１増やす
            word_count += 1

        return pos_count, neg_count, word_count

    # chiVeによる単語解析
    def GetNLModel_chiVe(self):
        # モデルの読み込み
        MODEL_PATH = r"\home\hikachof\デスクトップ\LangModels\chive-1.2-mc5_gensim\chive-1.2-mc5_gensim\chive-1.2-mc5.kv"
        return gensim.models.KeyedVectors.load(MODEL_PATH)
    def GetNLModel_fastText(self):
        # モデルの読み込み
        MODEL_PATH = r"\home\hikachof\デスクトップ\LangModels\cc.ja.300.vec"
        return gensim.models.KeyedVectors.load_word2vec_format(MODEL_PATH, binary=False)

    def GetWordView_chiVe(self, word):
        wv = self.GetNLModel_chiVe()
        
        # 類似度上位10件を取得
        match = wv.most_similar(word, topn=10)
        
        # 見やすい形式で表示
        print(match)

    # fastTextによる単語解析
    def GetWordView_fastText(self, word):
        wv = self.GetNLModel_fastText()
        
        # 類似度上位10件を取得
        match = wv.most_similar(word, topn=10)
        
        # 見やすい形式で表示
        print(match)

    # 形態素解析１
    def MakeMorphologicalAnalysis_Juman(self, txts):
        jumanpp = Juman("jumanpp", None, 32000, 30)
        #jumanpp.timeout = 3000
        # 要素の出現頻度を計算する
        word_freq = defaultdict(int)
        for t in txts:
            print(t)
            t = t.replace("\n", "")
            t = t.replace("\\", "")
            result = jumanpp.analysis(t)
            ress = []
            for mrph in result.mrph_list(): # 各形態素にアクセス
                #print("見出し:%s, 読み:%s, 原形:%s, 品詞:%s, 品詞細分類:%s, 活用型:%s, 活用形:%s, 意味情報:%s, 代表表記:%s" \
                #        % (mrph.midasi, mrph.yomi, mrph.genkei, mrph.hinsi, mrph.bunrui, mrph.katuyou1, mrph.katuyou2, mrph.imis, mrph.repname))

                surface = mrph.midasi
                base = mrph.genkei
                pos = mrph.hinsi
                pos1 = mrph.bunrui
                ress.append(dict(表層形=surface, 基本形=base, 品詞=pos, 品詞1=pos1))

            #
            for res in ress:
                if res["品詞"] == '名詞':
                    word_freq[res['基本形']] += 1

        
        sort_words = sorted(word_freq.items(), key=lambda x:x[1], reverse=True)
        #for sw in sort_words:
        #    print(sw)
        return sort_words

    def MakeMorphologicalAnalysis_MeCab(self, txt):
        option0 = ''
        option1 = '-Ochasen'
        option2 = '-Owakati'
        tagger = MeCab.Tagger()
        # 簡単な形態素解析
        parsed_txt = tagger.parse(txt)

        #
        ress = []
        elements = parsed_txt.split("\n")[:-2]
        for element in elements:
            #print(element)
            #print(element)
            parts = element.split(",")
            surface_pos = parts[0].split("\t")
            try:
                surface = surface_pos[0]
                pos = surface_pos[1]
            except:
                pass
            pos1 = parts[1]
            base = parts[-3]
            #
            if base != "*":
                ress.append(dict(表層形=surface, 基本形=base, 品詞=pos, 品詞1=pos1))

        #print(ress)
        # 要素の出現頻度を計算する
        word_freq = defaultdict(int)
        word_kigou = defaultdict(int)

        #
        for res in ress:
            #print(res)
            if res["品詞"] == '名詞':
                word_freq[res['基本形']] += 1
            if res["品詞"] == '記号':
                word_kigou[res['基本形']] += 1

        #print(word_freq)
        
        sort_freqs = sorted(word_freq.items(), key=lambda x:x[1], reverse=True)
        sort_kigous = sorted(word_kigou.items(), key=lambda x:x[1], reverse=True)
        #for sw in sort_words:
        #    print(sw)
        return sort_freqs, sort_kigous
        
    


# Seleniumなどを用いたスクレイピングのクラス
class ScraypinIn:
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 6.1; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/83.0.4103.116 Safari/537.36'}
    driver = None

    counttimes = {}

    def __init__(self):
        # ブラウザーを起動
        chrome_service = fs.Service(executable_path=CHROMEDRIVERPATH)

        #　ヘッドレスモードでブラウザを起動
        options = Options()
        options.add_argument(f'service={chrome_service}')
        #options.headless = True
        #options.add_argument('--headless')
        options.add_experimental_option('excludeSwitches', ['enable-logging'])

        self.driver = webdriver.Chrome(options=options)

        self.DoTimeCounter("GT_ALL")

    def __del__(self):
        self.DoTimeCounter("GT_ALL")

    # 実行時間の計測
    def DoTimeCounter(self, CountName = "Default"):
        nowtime = time.perf_counter()
        try:
            t = self.counttimes[CountName]
            print("実行時間 : " + CountName + " : " + str(nowtime - t))
        except:
            pass
        #    
        self.counttimes[CountName] = nowtime

    # ブラウザを閉じて終了する
    def Quit(self):
        time.sleep(3)
        self.driver.quit()


    # 指定した名前で検索してすべての配信情報を得る
    def SearchUserSite(self, username):
        driver = self.driver

        # Twitter
        url = "https://twitter.com/search?q=" + username + "&src=typed_query&f=user"
        #driver.execute_script("window.open(\"" + url + "\");")
        driver.get(url)
        #time.sleep(0.5)
        # Twitch
        url = "https://www.twitch.tv/search?term=" + username
        driver.execute_script("window.open(\"" + url + "\");")
        #time.sleep(0.5)
        # Youtube
        url = "https://www.youtube.com/results?search_query=" + username + "&sp=EgJAAQ%253D%253D"
        driver.execute_script("window.open(\"" + url + "\");")
        #time.sleep(0.5)
        # Mildom
        url = "https://www.mildom.com/search/" + username + "?tab=live"
        driver.execute_script("window.open(\"" + url + "\");")
        #time.sleep(0.5)
        # NicoNico
        url = "https://live.nicovideo.jp/search?keyword=" + username + "&status=onair&sortOrder=recentDesc&providerTypes=community"
        driver.execute_script("window.open(\"" + url + "\");")
        #time.sleep(0.5)

        # タブを閉じる
        # driver.close()
        #self.Quit()

    # ツイッターなどから取得できる簡略化された文字数字を数値に変換する
    def FixStrNumber(self, stnum):
        stnum = stnum.replace(",", "")
        if "万" in stnum:
            stnum = stnum.replace("万", "")
            stnum = float(stnum) * 10000

        return int(stnum)

# Seleniumを用いたTwitterからの情報収集
class ScrayTwitter(ScraypinIn):
    # パラメータ
    myID = ""
    
    # -- Global Params ---
    globalngwords = ["東方神起", "ゲロ", "クソ"]
    # ページのスクロール回数
    max_scroll_count = 100
    scroll_wait_time = 0.1
    # ReTweetは取得するのに時間がかかる
    # 制限を設けることによって面倒をなくす
    maxretweetpage = 8
    maxretweet = 100
    # ツイートの最大取得数の設定
    maxgettweetcount = 0
    # 含まれてほしくないワード
    ngwords = []
    # いいね数の範囲
    mincountlike = 0
    maxcountlike = 0
    # リツイート数の範囲
    mincountretweet = 0
    maxcountretweet = 0
    # リプライ数の範囲
    mincountreply = 0
    maxcountreply = 0
    # -- LOCAL VALUES --
    # スクロール時に最後に読み取ったデータ
    LastElemData = None
    #LastElemDataCount = 0
    # 実行内容はここに保存される
    tweet_list = []
    id_list = []
    #
    bLogin = False

    # 情報収集などに用いるアカウント
    loginData_Dummy = FTwitterLoginData("Dummy___Plug", "KAWAZOE4367")
    # 通常の使用されるアカウント
    loginData_Type01 = FTwitterLoginData("lepumoshion", "KAWAZOE4367")
    loginData_Type02 = FTwitterLoginData("Aun114514", "KAWAZOE4367")
    loginData_Type03 = FTwitterLoginData("hikarutanden", "KAWAZOE4367")
    loginData_Type04 = FTwitterLoginData("punnchipunn", "KAWAZOE4367")
    loginData_Type05 = FTwitterLoginData("tandemotarou", "KAWAZOE4367")
    
    #
    debugnum = 0

    
    def __init__(self, myid = "Dummy___Plug"):
        super().__init__()
        if myid:
            self.myID = myid

        self.LoginTwitter()

    def Reset(self):
        self.tweet_list.clear()
        self.id_list.clear()

    # ツイートの内容をチェックする
    def CheckInfo(self, info):
        # NGワード
        for nw in self.ngwords:
            if re.search(nw, info["tweet"]):
                return False

        # 検索時に範囲指定をするのでいい
        """
        # いいね数の範囲
        if self.mincountlike != 0:
            if self.mincountlike > info["likecount"]:
                return False
        if self.maxcountlike != 0:
            if info["likecount"] > self.maxcountlike:
                return False

        # リツイート数の範囲
        if self.mincountretweet != 0:
            if self.mincountretweet > info["retweetcount"]:
                return False
        if self.maxcountretweet != 0:
            if info["retweetcount"] > self.maxcountretweet:
                return False

        # リプライ数の範囲
        if self.mincountreply != 0:
            if self.mincountretweet > info["replycount"]:
                return False
        if self.maxcountreply != 0:
            if info["replycount"] > self.maxcountreply:
                return False
        #
        """
        return True

    # ログインする
    def LoginTwitter(self):
        loginData = self.GetLoginData()

        if not loginData:
            return

        driver = self.driver

        id = loginData.id
        pw = loginData.pw

        url = "https://twitter.com/i/flow/login"
        driver.get(url)
        
        time.sleep(4)
        input_mail = driver.find_element(By.TAG_NAME, "input")
        input_mail.clear()
        input_mail.send_keys(id)
        btn_enter = driver.find_element(By.XPATH, "/html/body/div/div/div/div[1]/div/div/div/div/div/div/div[2]/div[2]/div/div/div[2]/div[2]/div/div/div/div[6]")
        btn_enter.click()
        time.sleep(3)
        # あなたはぼっとですかっていう簡単なチェック、出てこないこともあるので対応する
        try:
            input_username = driver.find_element(By.TAG_NAME, "input")
            input_username.clear()
            input_username.send_keys(id)
            btn_enter = driver.find_element(By.XPATH, "/html/body/div[1]/div/div/div[1]/div/div/div/div/div/div/div[2]/div[2]/div/div/div[2]/div[2]/div[2]/div/div/div/div")
            btn_enter.click()
            time.sleep(3)
        except:
            pass
        #
        input_pass = driver.find_elements(By.TAG_NAME, "input")[1]
        input_pass.clear()
        input_pass.send_keys(pw)
        time.sleep(3)
        btn_enter = driver.find_element(By.XPATH, "/html/body/div[1]/div/div/div[1]/div/div/div/div/div/div/div[2]/div[2]/div/div/div[2]/div[2]/div[2]/div/div[1]/div/div")
        btn_enter.click()
        #
        time.sleep(3)

        self.bLogin = True

    # ログインしているか
    def IsLogin(self):
        return self.bLogin

    # 自らのログインデータの中からIDから必要なPASSを得る
    def GetLoginData(self):
        if self.loginData_Dummy.id == self.myID:
            return self.loginData_Dummy
        if self.loginData_Type01.id == self.myID:
            return self.loginData_Type01
        elif self.loginData_Type02.id == self.myID:
            return self.loginData_Type02
        elif self.loginData_Type03.id == self.myID:
            return self.loginData_Type03
        elif self.loginData_Type04.id == self.myID:
            return self.loginData_Type04
        elif self.loginData_Type05.id == self.myID:
            return self.loginData_Type05

    # 日本語のみを対象とした正規表現のパターンを得る
    def GetReCompile_JP(self):
        p = re.compile("[^" + re_hiragana + re_katakana1 + re_katakana2 + re_kanji + re_num1 + re_num2 + re_kigou + re_alfa + "]+")
        return p

    # 指定したキーワードに関しての最近のツイートを得る
    # そのキーワードに関してあまりアクティブでない場合は無効なツイートを返す
    def GetTargetWordTweet(self, word):
        # %23 == #
        activeword = word.replace("#", "%23")

        self.mincountlike = 5
        self.maxcountlike = 80
        self.maxgettweetcount = 80

        tweetdata = self.GetTweet(activeword, 3, 1, 0, False, True, True, False)
        if tweetdata:
            basetweets = []
            # 必要数確保できなかった場合
            if self.maxgettweetcount > len(tweetdata):
                return
            # ツイートの含まれるハッシュタグを抽出して削除する
            takehashs = {}
            takehashs[word.replace("#", "")] = 999
            for tw in tweetdata:
                #print(tw)
                #print(tw["tweet"])
                # 抽出
                for nh in re.findall(r"#(\w+)", tw["tweet"]):
                    if nh in takehashs:
                        takehashs[nh] += 1
                    else:
                        takehashs[nh] = 1
                # 削除
                # すべて消すか、#のみ消すか
                if False:
                    basetweets.append(re.sub(r"#(\w+)", "", tw["tweet"]))
                else:
                    basetweets.append(tw["tweet"].replace("#",""))
            # ハッシュ出現数が多い順にする
            #print(takehashs)
            takehashs = sorted(takehashs.items(), key=lambda x:x[1], reverse=True)
            # オリジナルのツイートの作成.>
            nexttweets = self.GetOriginalTweetFromAI(word, basetweets, 10)
            # 最初に問題のある文字列が含まれていた場合はそれを削除する
            if nexttweets:
                tweets = []
                for nexttweet in nexttweets:
                    nexttweet = re.sub(r"^[^" + re_hiragana + re_katakana1 + re_katakana2 + re_kanji + re_alfa + "]+", "", nexttweet)
                    if len(nexttweet) > 8:
                        for v in takehashs:
                            # 40ツイート中に10以上含まれているハッシュタグのみを追加する
                            if v[1] > 10:
                                nexttweet += "\n#" + v[0]
                        #
                        tweets.append(nexttweet)
                return tweets

    # 指定した個数の現在のトレンドか興味のあるワードのツイートを作成して得る
    def MakeActiveTweet(self, gettrend, getnum):
        tws = []
        nghashs = []
        # 過去のNGHashを呼び出す
        loadhashs = g.LoadData("damps", "damp_nghashs")
        if loadhashs:
            for has in loadhashs:
                nghashs.append(has["0"])
        print("NG:")
        print(nghashs)
        count = 0
        word = None
        for n in range(getnum):
            if gettrend:
                nword = self.GetMostActiveHash(nghashs)
                # 3回同じワード許す処理
                if word != nword:
                    count = 0
                    word = nword
                # NGの追加
                count += 1
                print("count: " + str(count))
                if count >= 3:
                    count = 0
                    print("NGワード追加: " + nword)
                    nghashs.append(nword)
                    
            else:
                word = self.GetBaseTweetFileName()
            # アクティブなハッシュからツイートを取得して、そのツイートをもとにツイートの作成を行う
            #print(originalactivehash)
            if word:
                nexttweets = self.GetTargetWordTweet(word)
                if nexttweets:
                    for nexttweet in nexttweets:
                        #print(nexttweet)
                        tws.append(nexttweet)
        #
        if word and not (word in nghashs):
            print("NGワード追加: " + word)
            nghashs.append(word)


        # すでに行ったハッシュは保存しておく
        #print("NG:")
        #print(nghashs)
        g.SaveData(nghashs, "damps", "damp_nghashs")

        return tws

    # AIのべりすとのツイートファイルの名前をランダムで返す
    def GetBaseTweetFileName(self):
        filepaths = glob.glob("C:\\Users\\hikac\\Documents\\VSCode\\Pythons\\TwitterProject\\AIのべりすと\\*")
        filepath = random.choice(filepaths)
        return filepath.split("\\")[-1].split("_")[0]

    # 参考ツイートを指定した中からランダムで指定数取得する
    # これはAIのべりすとにおいて、実際にツイートして使用する文章の導として使用する
    def GetBaseTweet(self, fileType, getnum):
        try:
            f = open(f"AIのべりすと\\{fileType}_twitter.txt", 'r', encoding='UTF-8')
            ss = f.readlines()
            f.close()
            return random.sample(ss, getnum)
        except:
            pass

    # AIのべりすとからツイート文を作成して取得する
    # basetextにより作成文章の方向性を示すことができる
    # extractnumでbasetextをいくつ抽出するか設定する
    def GetOriginalTweetFromAI(self, word, originalbasetexts, extractnum):
        driver = self.driver

        # 古いタブのハンドル
        oldurl = driver.current_url
        #
        url = "https://ai-novel.com/login.php"
        driver.get(url)

        mostngwords = ["ゲロ", "クソ", "最悪", "つまらない", "最低"]
        
        try:
            time.sleep(2)
            # AIのべりすとにログインする
            try:
                elem_ed_mail = driver.find_element(By.XPATH, "//input[contains(@name, 'userid')]")
                elem_ed_password = driver.find_element(By.XPATH, "//input[contains(@name, 'password')]")
                elem_btn_sinin = driver.find_element(By.XPATH, "//input[contains(@id, 'login')]")
                # 
                elem_ed_mail.send_keys("hikachofaw@gmail.com")
                elem_ed_password.send_keys("konnsome")
                elem_btn_sinin.click()
                time.sleep(1)

                # 下準備
                url = "https://ai-novel.com/mods.php"
                driver.get(url)
                time.sleep(1)
                elem_sel_mod = driver.find_element(By.ID, "mods-options")
                elem_sel_mod.click()
                elem_sel_mod.send_keys(Keys.ARROW_DOWN)
                elem_sel_mod.send_keys(Keys.ENTER)
                time.sleep(0.5)
            except:
                pass

            #"""
            # 初期設定を行う
            driver.get("https://ai-novel.com/index.php")
            time.sleep(1)
            elem_btn_start = driver.find_element(By.ID, "newdocument").find_element(By.ID, "continuebutton")
            elem_btn_start.click()

            #
            time.sleep(3)
            elem_ed_authorsnote = driver.find_element(By.XPATH, "//textarea[contains(@id, 'authorsnote')]")
            elem_ed_badwords = driver.find_element(By.XPATH, "//textarea[contains(@id, 'badwords')]")
            elem_btn_02 = driver.find_element(By.ID, "balloon_options_pin")
            elem_btn_02.click()

            # オーサーズ・ノートや禁止ワードの設定
            time.sleep(0.5)
            elem_ed_authorsnote.send_keys("[ジャンル：ツイッター]\n性格が明るい\nポジティブ\n明るい\n" + word + "が大好き")
            # 今回のワードにNGワードが含まれている場合はそれを削除する
            ngws = "雑談\nDM\n最新\n\"゚\n特典\nSwitch\n飽き\npsvita\nPSVITA\nPSVita\n嫌い\n悪い\n興味ない\n☆\nzz\nZZ\noo\noO\nRT\n予約\n0\n1\n2\n3\n4\n5\n6\n7\n8\n9\n:\n.\n日\n月\n。。\nゲロ\n最近\n今期\nゴミ\nツイキャス\nwww\n本日\n出演\n参加\n明日\n昨日\n今日\n優勝\n予選\n突破\n大会\n賞金\n募集\nクソ\ncom\nhttps\nhttp\n・\n（注）\n『\n』\n【\n】\n(\n)\n（\n）\n#\nネガティブ\n@\n：\n:\n※\n小説\n作文\n息子\n娘\n子供\nおっぱい\nTwitter\nアカウント\nツイート\n?\nー\n—\n今年\n来年\nコメント\nおはよう\nおやすみ\n配信"
            ngws = ngws.split("\n")
            s = ""
            for nw in ngws:
                if nw in word:
                    pass
                else:
                    s += nw + "\n"
            #print(s)
            # ガチNG
            ngws += "東方神起\n"
            elem_ed_badwords.send_keys(s)
            
            # ツイートをゲットする
            # 適切な文ができるまで適当に回す
            maxloop = 5
            for i in range(maxloop):
                elem_btn_nara = driver.find_element(By.XPATH, "//label[contains(@for, 'writingmode-3')]")
                elem_btn_nara.click()

                basetexts = random.sample(originalbasetexts, extractnum)

                elem_ed_text = driver.find_element(By.XPATH, "//div[contains(@id, 'data_edit')]")
                elem_btn_next = driver.find_element(By.XPATH, "//input[contains(@id, 'getcontinuation')]")
                #
                elem_ed_text.click()
                elem_ed_text.clear()
                time.sleep(0.5)
                #
                textnum = len(basetexts)
                for r,s in enumerate(basetexts):
                    s = s.replace("\n", "")
                    elem_ed_text.send_keys(s)
                    #print(str(r) + "/" + str(textnum))
                    if (r+1) == textnum and not re.search("[。|!|！]", s[-1]):
                        elem_ed_text.send_keys("。")
                    elem_ed_text.send_keys("\n")
                time.sleep(1)
                # 最大ループ回数
                # AIのべりすとによるツイートの作成
                oldtext = ""
                elem_btn_next.click()
                # ループによって抜けた場合は最後にチェックする
                loopout = True
                for i in range(5):
                    try:
                        # 異常なトラフィックが連続して検出されました。アクセスを一時停止しています
                        #print("loop: " + str(i))
                        elem_btn_next.click()
                        # すでにある文章とは別に新たに１行文章が追加された場合に終了する
                        elem_ed_text = driver.find_element(By.XPATH, "//div[contains(@id, 'data_edit')]")
                        t = elem_ed_text.text
                        ts = t.split("\n")
                        # ツイートできそうな文章ができたので取得する

                        def CheckTweet(checktweet):
                            # 区切りがある場合は複数のツイート候補とする
                            nexttweets = []
                            # 半角の空白で区切りたいが、アルファベットの場合は辞める
                            while True:
                                sd = re.search(f"[^{re_alfa}] [^{re_alfa}]", checktweet)
                                if sd:
                                    sep = sd.span()[0]
                                    nexttweets.append(checktweet[:sep+1])
                                    checktweet = checktweet[sep+2:]
                                else:
                                    break
                            #
                            nexttweets.append(checktweet)
                            # 条件に合うツイートのみを追加する
                            tweets = []
                            for nexttweet in nexttweets:
                                ero = False
                                if len(nexttweet) > 120 or len(nexttweet) < 10:
                                    print("A")
                                    ero = True                                    
                                if  not re.search(f"[{re_hiragana}]", nexttweet):
                                    print("B")
                                    ero = True
                                if nexttweet[-1] == "、":
                                    print("C")
                                    ero = True
                                if not (word in nexttweet):
                                    print("D")
                                    ero = True
                                # 絶対に入れてほしくないNGワード
                                # これは上の方で指定してもなんか入ってしまうことがあるので２重チェックとして行っている
                                for mng in mostngwords:
                                    if mng in nexttweet:
                                        print("NG:" + mng)
                                        ero = True
                                        break
                                
                                if ero:
                                    continue
                          
                                tweets.append(nexttweet)

                            # 1つでもあればそれらを渡す
                            if len(tweets) == 0:
                                return None
                                
                            driver.get(oldurl)
                            time.sleep(3)
                            
                            return tweets

                        if (textnum+1) < len(ts):
                            # 指定数以下の文字数か、ひらがなが含まれていない場合に再度検索する
                            nexttweet = ts[textnum]
                            tweets = CheckTweet(nexttweet)
                            if tweets:
                                return tweets
                            else:
                                loopout = False
                                break
                            
                        # 新たに作成された文章が全然だった場合には初めから
                        if (len(ts)-1) == textnum:
                            newtext = ts[textnum]
                            #print("new: " + newtext + "\nold: " + oldtext)
                            sa = len(newtext) - len(oldtext)
                            if sa < 4:
                                # 最後のチェック
                                tweets = CheckTweet(nexttweet)
                                if tweets:
                                    return tweets
                                else:
                                    print("E: " + str(sa))
                                    loopout = False
                                    break
                            else:
                                oldtext = newtext
                        #
                        time.sleep(15)
                    except:
                        time.sleep(20)
                # ループによって抜けた場合は最後の文字列をチェックする
                if loopout and len(ts) > textnum:
                    try:
                        nexttweet = ts[-1]
                        tweets = CheckTweet(nexttweet)
                        if tweets:
                            return tweets
                    except:
                        pass

                driver.refresh()
                time.sleep(3)
        except:
            pass

        
        driver.get(oldurl)
        time.sleep(3)


    # 読み取った文字列を純粋な文章形式にする
    def GetFixText(self, t):
        tokusyu = "、、＠"
        # URLの削除
        t = re.sub(re_http, "、", t)
        # 顔文字関連の削除
        #t = re.sub(r"\([^あ-ん\u30A1-\u30F4\u2E80-\u2FDF\u3005-\u3007\u3400-\u4DBF\u4E00-\u9FFF\uF900-\uFAFF\U00020000-\U0002EBEF]+?\)", ",", t)
        # そこから漏れてしまった不要な文字の削除
        t = t.replace("#", "")
        #t = t.replace(chr(0xa), "")
        #t = t.replace("pino", "")
        #t = re.sub(r"[.|。ﾟ|\+|٩|و|‧|⁺|◟|*|:|≡|＾|Ｏ|…|∠|／|･|◯|！|!|。|｡|＼|\\| |/|！|◁|▷|⌒|\xa0|\n|♡|;|；|：|゜|'|_|^|-|＿|ʕ|•|ʔ|'|▶︎|．|?|？|‥|◆|●|★|]", tokusyu, t)
        
        p = self.GetReCompile_JP()
        t = p.sub(tokusyu, t)
        
        #kigou = "!|,|.|?|_|^|＾|ﾉ|;|；|/|／|:|：|ゝ|*|ヾ|\"|Ｏ|\\\|+|＋|⁺|◟|ᵒ̴̶̷̥́ |·̫| |ᵒ̴̶̷̣̥̀| |.｡ﾟ+.)"
        t = re.sub("[　| ]", "", t)
        t = re.sub("[" + re_kigou_b + re_kigou_a + "]", tokusyu, t)
        # 顔文字とかの装飾文字列をカウントして、出現回数によって文章の最後にビックリマークをつけるかを行う
        if t.count(tokusyu) >= 3:
            t += "！"
        else:
            t += "。"
        t = t.replace(tokusyu, "、")
        t = re.sub("、+", "、", t)
        t = t.replace("、！", "！")
        t = t.replace("、。", "。")
        # ()の中のゴミ掃除
        t = re.sub("[(|（|\[][" + re_kigou_a + re_kigou_b + "、]+[)|）|\]]", "", t)
        t = t.replace("、！", "！")
        t = t.replace("、。", "。")
        #print(t)
        #t = t.replace("*\\", "")
        # 最後の文字もそうだが、最初の文字も気をつけなければいけない
        if len(t) > 0:
            t0 = t[0]
            # 最初の文字が 、 だと変な部分で改行される
            if t0 == "、":
                t = t[1:]
            # 最初の文字が半角英数字だと良くない
            #newt = mojimoji.han_to_zen(t[0])#, kana=True, digit=True, ascii=True)
            #t = newt + t[1:]
            t0 = t[0]
            if re.match("[" + re_alfa + re_num1 + re_katakana2 + "]", t0):
                t = "まぁ" + t
        t = t.replace("、", "")

        return t

    # 受け取った文字列から感情を読み取る処理を行ってその結果を書き出す
    # 呼び出す際にお金が発生するので気をつけること
    def CreateSentiment(self, targetID):
        key = "AIzaSyA-ALaHEs7lFOv9nLF936-qzgIRbI21YyM"

        # ツイートデータの取得
        tweets = []
        datas = g.LoadData("Users\\" + targetID, "Tweet")
        for d in datas:
            tweets.append(d["tweet"])
        

        # 指定したTargetIDのTweetデータを読み込んで、
        if len(tweets) > 0:
            text = ""
            # 送った回数が料金に直結する、複数のツイート内容を１つの文章のようにまとめて送ることによって回数を節約する
            # Googleの処理においてここで文が区切らているという解釈がされる文字などがある、それを重視して意識的に文を書き換えている(とても重要)
            # これを行わないとツイートごとの感情分析がもらえない
            count = 0
            for t in tweets:
                t = self.GetFixText(t)
                if len(t) > 1:
                    #print(t)
                    text += t
                    text += "\n"
                    count += 1
                    print(str(count) + ":" + t)

            # 1000000Byteを超えるとGoogle側が対応できないのでそれ以上は切り捨てる
            # でも、1000000Byteだと、1000000だめですよってエラー出る、それ以下になってるはずなのに、なので半分にする
            #print(text)
            maxSize = 500000
            dsize = sys.getsizeof(text)
            if dsize > maxSize:
                h = maxSize / dsize
                s = int(len(text) * h) - 10
                #print(len(text))
                text = text[:s]
                s = text.rsplit("\n", 1)
                text = s[0]
                #print(sys.getsizeof(text))
            print(dsize)
            print(sys.getsizeof(text))

            # Google側にデータを送って結果をもらう
            # 料金が発生するので注意が必要である
            url = f"https://language.googleapis.com/v1beta2/documents:analyzeSentiment?key={key}"
            header = {"Content-Type": "application/json"}
            body = {
                "document": {
                    "type": "PLAIN_TEXT",
                    "language": "JA",
                    "content": text
                }
            }
            res = requests.post(url, headers=header, json=body)
            res = res.json()

            g.SaveArticle(res, f"{targetID}_SentimentData")

    # 指定したツイートの内容の感情データを描画する
    # 事前にcreateSentimentを呼び出さないと使えない
    def ViewSentiment(self, targetID):
        dumpname = f"{targetID}_SentimentData"
        tw_res = g.LoadData("Users//" + targetID, "Tweet")
        sen_res = g.LoadArticle(dumpname)
        
        g_datas = []
        sens = sen_res["sentences"]
        print(str(len(tw_res)) + ":" + str(len(sens)))
        
        hoge = {}
        # 感情分析データから必要なデータを取得して関連したツイートの日付などのデータと合わせて１つのデータとする
        for i,sen in enumerate(sens):
            text = sen["text"]["content"]

            # 文字が正しく処理されているかチェック
            if debug:
                if len(sens) > (i+1):
                    nexttext = sens[i+1]["text"]["content"]
                    try:
                        hoge[text[-1]][nexttext[0]] = i
                    except:
                        hoge[text[-1]] = {nexttext[0]: i}
                # 改行が２つ以上ある文を抽出
                c = text.count(chr(0xa))
                if False and c > 0:
                    print(str(i) + ":count:" + str(c) + ":" + text)
                #continue
                if True:
                    print("Begin:" + str(i))
                    print(text)
                    tx = tw_res[i]["tweet"]
                    print(tx)
                    tx = self.GetFixText(tx)
                    print(tx)
            # score, mag, datetime等のデータを纏める
            score = sen["sentiment"]["score"]
            mag = sen["sentiment"]["magnitude"]
            datetime = tw_res[i]["datetime"]
            datetime = datetime.split("_")[:3]
            truedatetime = int(datetime[0]) * 365 + int(datetime[1]) * 30 + int(datetime[2])
            # 2022-02-10の形式にする
            datetime = dtime(int(datetime[0]), int(datetime[1]), int(datetime[2]))
            # すべてのデータをまとめる
            g = {}
            g["text"] = text
            g["score"] = score
            g["mag"] = mag
            g["datetime"] = datetime
            g["truedatetime"] = truedatetime
            g_datas.append(g)
        # 
        if debug:
            for h in hoge:
                print("BEGIN:" + h)
                print(hoge[h])
            return
        maxNum = len(g_datas)
        maxDay = 7
        nowscore = 0
        nowday = 0
        startday = 0
        #
        plot_score = []
        plot_day = []
        #
        g_datas.reverse()
        #
        startdatetime = 0
        enddatetime = 0
        for i,g in enumerate(g_datas):
            nowscore += g["score"] * g["mag"]
            day = g["truedatetime"]
            if startday == 0:
                startday = day
                startdatetime = g["datetime"]
            if nowday == 0:
                nowday = day
            # 指定した日数経過した時点でそれまでのスコアを書き出す
            # 次のデータが別に日にちになっていた場合
            #pday = nowday - startday
            #print(day)
            if maxNum > (i+1):
                if g_datas[i + 1]["truedatetime"] > day:
                    # 前回の記録した日にちから経過した日にち
                    progday = day - nowday
                    #print(progday)
                    if progday > maxDay:
                        # データの書き出し
                        plot_score.append(nowscore)
                        plot_day.append(g["datetime"])
                        nowday = 0
                        nowscore = 0
            else:
                # 残っているデータの書き出し
                plot_score.append(nowscore)
                plot_day.append(g["datetime"])
                enddatetime = g["datetime"]
        
        #print(len(plot_score))
        # データをグラフで見る
        print(str(startdatetime) + " : " + str(enddatetime))
        # データの最初と最後の経過日数を週にしたもの
        # これを使用してデータの横サイズを設定する
        xsize = int((enddatetime - startdatetime).days/7)/5
        #
        x = plot_day
        y = plot_score
        df = pd.DataFrame({"x": x, "y": y})
        #
        x = df["x"]
        y = df["y"]
        #
        #print(df["x"])
        plt.figure(figsize=(xsize,20))
        ax = plt.axes()
        ax.set_facecolor("black")
        print(id)
        l1 = "@" + id
        l2 = l1 + "sma05"
        l3 = l1 + "sma25"
        span05 = 10
        span25 = 20
        df["sma05"] = y.rolling(window=span05).mean()
        df["sma25"] = y.rolling(window=span25).mean()
        
        plt.plot(x, y, color="red", linewidth=3, linestyle="solid", marker="o", label=l1)
        plt.plot(x, df["sma05"], color="#87ceeb", linewidth=4, linestyle="solid", marker="o", label=l2)
        plt.plot(x, df["sma25"], color="yellow", linewidth=4, linestyle="solid", marker="o", label=l3)
        plt.xticks(fontsize=xsize)
        plt.yticks(fontsize=50)
        plt.xlabel("Day", fontsize=xsize)
        plt.ylabel("Score", fontsize=70)
        plt.grid(axis="x")
        #plt.annotate("annotation", xy=(x[2],y[2]), xytext=(20,20), fontsize=15, color="b", arrowprops=dict(color="gray"))
        plt.title("メンタルスコア", fontsize=40)
        # plotのLabelを有効にするために必要
        plt.legend(fontsize=xsize)
        plt.ylim(-20, 20)
        #plt.show()
        plt.savefig(dumpname + ".png")

    # 今のトレンドの中から自分が興味あるジャンルの中で最もアクティブなハッシュタグを得る
    def GetMostActiveHash(self, nghashs = None):
        # 興味のあるジャンル
        genres = ["ゲーム", "アニメ", "テクノロジー"]
        driver = self.driver

        driver.get("https://twitter.com/explore/tabs/trending")
        time.sleep(3)
        tags = []
        lastelem = None

        # すべての有効なハッシュを取得する
        while True:
            elems_tag = driver.find_elements(By.XPATH, "//div[contains(@data-testid, 'cellInnerDiv')]")
            for tag in elems_tag:
                try:
                    #circleが存在していれば有効とする
                    tag.find_element(By.TAG_NAME, "circle")
                    # 重複した要素が存在していたか
                    if tag in tags:
                        continue
                    tags.append(tag)
                except:
                    pass
            # 一番下の要素までスクロール
            # 最後の要素が同じだった場合はそこで終了する
            if lastelem == elems_tag[-1]:
                break
            lastelem = elems_tag[-1]
            #
            actions = ActionChains(self.driver);
            actions.move_to_element(lastelem);
            actions.perform();
        
        
        # ハッシュタグの中で最もアクティブなものを取得する
        mostactivenum = 0
        mostactivehash = ""
        for tag in tags:
            #print(tag)
            try:
                elems_top = tag.find_elements(By.XPATH, "./div/div/div/div/div")
                activetext = elems_top[-2].text
                #print(activetext)
                if "件のツイート" in activetext:
                    activenum = activetext.split("件")[0]
                    activenum = int(activenum.replace(",",""))
                    #print(activenum)
                    genretext = elems_top[0].text
                    # ジャンルが適切か
                    hashtag = elems_top[1].text
                    # 無効なハッシュの場合はキャンセル
                    if nghashs:
                        for nghash in nghashs:
                            if nghash in hashtag:
                                raise ValueError("")
                    
                    #if hashtag[0] == "#":
                    for genre in genres:
                        if genre in genretext:
                            if mostactivenum < activenum:
                                #print(genretext)
                                mostactivenum = activenum
                                mostactivehash = hashtag
                #print(tag.text)
            except:
                pass

        #
        #print(mostactivehash)
        #print(mostactivenum)

        return mostactivehash
                

    # 指定したアカウントをフォローするシステム
    def DoFollow(self, targetid):
        if self.IsLogin():
            driver = self.driver
            #
            url = "https://twitter.com/" + targetid
            driver.get(url)
            WebDriverWait(driver, 3).until(EC.visibility_of_element_located((By.TAG_NAME, 'article')))
            #
            elem_follow = driver.find_element(By.XPATH, "//div[contains(@aria-label, 'フォロー')]")
            elem_follow.click()


    # 指定した文をツイートするシステム
    def DoTweet(self, tw):
        driver = self.driver
        driver.get("https://twitter.com/home")
        time.sleep(3)
        #elem_tw = driver.find_element(By.CLASS_NAME, "DraftEditor-root")
        elem_tw = driver.find_element(By.XPATH, "//div[@aria-multiline='true']")
        elem_tw.send_keys(tw)
        time.sleep(1)
        try:
            elem_btn_tw = driver.find_element(By.XPATH, "//div[contains(@data-testid, 'tweetButtonInline')]")
            elem_btn_tw.click()
        except:
            elem_tw.send_keys(Keys.ESCAPE)
            time.sleep(0.3)
            elem_btn_tw = driver.find_element(By.XPATH, "//div[contains(@data-testid, 'tweetButtonInline')]")
            elem_btn_tw.click()


    # 指定したツイートをいいねしているアカウントを取得する
    def GetAcountForLike(self, targetID, tweetID):
        driver = self.driver

        likesurl = "https://twitter.com/" + targetID + "/status/" + tweetID + "/likes"

        driver.get(likesurl)
        time.sleep(1.2)
        #
        elem_likes = driver.find_element(By.XPATH, "//div[contains(@aria-label, 'タイムライン: いいねしたユーザー')]")
        ids = []
        # ループさせながらスクロールして取得していく
        while True:
            elems_like = elem_likes.find_elements(By.XPATH, ".//div[@dir='ltr']")
            for elem_like in elems_like:
                getid = elem_like.text
                if not (getid in ids):
                    ids.append(getid)
            
            if self.scroll_to_elem(elem_likes, By.XPATH, "./div/div[@data-testid='cellInnerDiv']"):
                break
            
            # 待つ（サイトに負荷を与えないと同時にコンテンツの読み込み待ち）
            time.sleep(self.scroll_wait_time)
              
        return ids

    # 指定したIDのリツイートの比率を取得する
    def GetAcountForLike(self, targetID, tweetID):
        driver = self.driver

        url = "https://twitter.com/" + targetID

        driver.get(url)
        time.sleep(1.2)
        #
        elem_tweettop = driver.find_element(By.XPATH, "//div[contains(@aria-label, 'タイムライン:')]")
        ids = []
        # ループさせながらスクロールして取得していく
        while True:
            elems_like = elem_likes.find_elements(By.XPATH, ".//div[@dir='ltr']")
            for elem_like in elems_like:
                getid = elem_like.text
                if not (getid in ids):
                    ids.append(getid)
            
            if self.scroll_to_elem(elem_likes, By.XPATH, "./div/div[@data-testid='cellInnerDiv']"):
                break
            
            # 待つ（サイトに負荷を与えないと同時にコンテンツの読み込み待ち）
            time.sleep(self.scroll_wait_time)
              
        return ids

    # 指定したアカウントのフォローしているアカウントを取得する
    def GetAcountForFollowing(self, targetID):
        driver = self.driver

        file_path = "Users/" + targetID
        file_name = "Following"
        ids = g.LoadData(file_path, file_name)
        if not isinstance(ids, NoneType):
            return ids

        url = "https://twitter.com/" + targetID + "/following"

        driver.get(url)
        time.sleep(1.2)
        #
        elem_followings = driver.find_element(By.XPATH, "//div[contains(@aria-label, 'タイムライン: フォロー中')]")
        ids = []
        # ループさせながらスクロールして取得していく
        while True:
            elems_like = elem_followings.find_elements(By.XPATH, ".//div[@dir='ltr']")
            for elem_like in elems_like:
                getid = elem_like.text
                if not (getid in ids):
                    ids.append(getid)
            
            if self.scroll_to_elem(elem_followings, By.XPATH, "./div/div[@data-testid='cellInnerDiv']"):
                break
            
            # 待つ（サイトに負荷を与えないと同時にコンテンツの読み込み待ち）
            time.sleep(self.scroll_wait_time)

        file_path = "Users/" + targetID
        file_name = "Following"
        self.SaveData(ids, file_path, file_name)
              
        return ids

    # 指定したツイートを開く
    def DoOpenTweet(self, targetID, tweetID):
        driver = self.driver

        url = "https://twitter.com/" + targetID + "/status/" + tweetID
        driver.get(url)
        time.sleep(1)

    # 最近の自分のツイートをいいねしているアカウントをすべて取得する
    def GetAcountsForMyLikeTweet(self):
        self.mincountlike = 1
        twdata = self.GetTweet(self.myID, 30, 1, 0, True, True, False, False)
        
        # いいねしているユーザーIDを取得していく
        if twdata:
            ids = []
            for t in twdata:
                oneids = self.GetAcountForLike(self.myID, t["link"])
                for id in oneids:
                    if not (id in ids):
                        ids.append(id)

            return ids


    # 自分の興味のある今話題のワードで最も近く話題のツイートをリツイートする
    def DoReTweetForTrendWord(self):
        # 今話題のワードを取得する
        nghashs = []
        # 過去にすでにリツイートを行ったワードは行わない
        loadhashs = g.LoadData("damps", self.myID + "_damp_nghashs")
        #print(loadhashs)
        if loadhashs:
            for has in loadhashs:
                nghashs.append(has["0"])
        activeword = self.GetMostActiveHash(nghashs)

        # 話題のワードで人気のあるツイートを取得する
        print(activeword)
        if activeword:
            nghashs.append(activeword)
            self.maxgettweetcount = 30
            self.mincountlike = 2000
            twdata = self.GetTweet(activeword, 3, 1, 0, False, True, False, False)

            # 取得したツイートの中で最もフォロワー数が少ないアカウントのツイートをリツイートする
            if twdata:
                mostminacount = None
                mostmintweetid = None
                mostminfollower = None
                acountdatas = []
                for t in twdata:
                    acount = t["id"]
                    tweetid = t["link"]
                    
                    # すでにデータが有る場合は追加しない
                    bIn = False
                    for ad in acountdatas:
                        if ad["acount"] == acount:
                            followernum = ad["fornum"]
                            bIn = True
                    # 素手の調べたものを再度調べると面倒なので行わないようにする
                    if not bIn:
                        # HomeDataを得る
                        homedata = self.GetTwitterHome(acount)
                        followernum = homedata["follower"]
                        acountdatas.append({"acount": acount, "fornum": followernum})
                    
                    
                    # フォロワーが少ない場合はそちらを主とする
                    if not mostminacount or mostminfollower > followernum:
                        mostminacount = acount
                        mostmintweetid = tweetid
                        mostminfollower = followernum

                # 最も適切なツイートに対してリツイートを行う
                if mostminacount:
                    self.DoTweet_ReTweet(mostminacount, mostmintweetid)

        # そのアカウントですでに取り上げたワードはもう行わない
        g.SaveData(nghashs, "damps", self.myID + "_damp_nghashs")

    # 対象のアカウントの直近のツイートの中で適当なものをいいねする
    def DoLikeTweetForAcount(self, targetAcount):
        # 対象の最近のツイートを取得する
        tws = self.GetTweet(targetAcount, 3, 1, 0, True, True, False, False)
        # 最近のツイートから適当なものを選択する
        if tws:
            # とりあえず適当に
            tw = random.choice(tws)
            link = tw["link"]
            # 選んだツイートをいいネする
            self.DoTweet_Like(targetAcount, link)

    # いいねがされているもの意外のツイートをすべて削除する
    def DoRemoveTweetOutLike(self):
        self.maxcountlike = 1
        twdata = self.GetTweet(self.myID, 10, 1, 0, True, True, False, False)

        if twdata:
            driver = self.driver
            for t in twdata:
                self.DoOpenTweet(self.myID, t["link"])
                elem_btn_svg = driver.find_element(By.XPATH, "//div[contains(@aria-label, 'もっと見る')]")
                elem_btn_svg.click()
                time.sleep(0.2)
                elem_menu = driver.find_element(By.XPATH, "//div[contains(@role, 'menu')]")
                elem_btn_rm = elem_menu.find_element(By.XPATH, ".//div[contains(@role, 'menuitem')]")
                elem_btn_rm.click()
                time.sleep(0.2)
                elem_menu = driver.find_element(By.XPATH, "//div[contains(@data-testid, 'confirmationSheetDialog')]")
                elem_btn_ok = elem_menu.find_element(By.XPATH, ".//div[contains(@role, 'button')]")
                elem_btn_ok.click()
                time.sleep(0.2)
        
    

    # 指定したツイートに対していいねを行う
    def DoTweet_Like(self, targetID, tweetID, NotLike = False):
        self.DoOpenTweet(targetID, tweetID)
        driver = self.driver
        # すでにいいねを行っていた場合には変わるのでそれに対しての処理
        try:
            if NotLike:
                elem_likebutton = driver.find_element(By.XPATH, "//div[@aria-label='いいねを取り消す'][@role='button']")
            else:
                elem_likebutton = driver.find_element(By.XPATH, "//div[@aria-label='いいね'][@role='button']")
        except:
            return
        elem_likebutton.click()
        time.sleep(0.3)

    # 指定したツイートに対してリツイートを行う
    def DoTweet_ReTweet(self, targetID, tweetID, NotReTweet = False):
        self.DoOpenTweet(targetID, tweetID)
        driver = self.driver
        # すでにリツイートを行っていた場合には変わるのでそれに対しての処理
        try:
            if NotReTweet:
                elem_likebutton = driver.find_element(By.XPATH, "//div[@aria-label='リツイートを取り消す'][@role='button']")
            else:
                elem_likebutton = driver.find_element(By.XPATH, "//div[@aria-label='リツイート'][@role='button']")
        except:
            return
        elem_likebutton.click()
        time.sleep(1)
        elem_menu = driver.find_element(By.XPATH, "//div[@role='group']")
        elem_retweet = elem_menu.find_element(By.XPATH, "//div[@role='menuitem']")
        elem_retweet.click()
        time.sleep(0.3)
        
        
    # すべてのTwitter情報を取得する
    def AllGetTwitters(self, targetID):
        self.LoginTwitter(self.loginData_Type01)
        self.GetTwitterHome(targetID)
        self.GetTweet(targetID, 30, 60, 0, True, True, False, False, targetID)
        self.getReTweet(targetID)
        self.getLikeTweet(targetID)
        self.Reset()
        self.Quit()

    # ツイッタアカウントの基本情報の取得
    def GetTwitterHome(self, targetID):
        driver = self.driver
        #atexit.register(self.Reset)

        # すでに存在している場合は処理しない
        file_path = "Users/" + targetID
        file_name = "TwitterHome"
        dummy = g.LoadData(file_path, file_name)
        if dummy:
            return dummy

        try:
            url = "https://twitter.com/" + targetID
            driver.get(url)

            homeData = {}

            time.sleep(5)
            #
            try:
                elem_banner = driver.find_element(By.XPATH, "//img[contains(@src, 'profile_banners')]")
                file_name = targetID + "_Banner"
                g.SaveImage(elem_banner.get_attribute("src"), file_path, file_name)
            except:
                pass
            try:
                elem_profile = driver.find_element(By.XPATH, "//img[@alt='プロフィール画像を開きます'][contains(@src, 'profile_images')]")
                file_name = targetID + "_Icon"
                g.SaveImage(elem_profile.get_attribute("src"), file_path, file_name)
            except:
                pass
            elem_username = driver.find_element(By.XPATH, "//div[@data-testid='UserName']")
            elems_username_span = elem_username.find_elements(By.TAG_NAME, "span")
            #for i,s in enumerate(elems_username_span):
            #    print(str(i) + ":" + s.text)
            homeData["name"] = elems_username_span[1].text
            homeData["id"] = elems_username_span[3].text
            
            # divの個数がBannerによって変化するので対応している
            elem_home = driver.find_element(By.XPATH, "//nav[@aria-label='プロフィールタイムライン'][@role='navigation']")
            elem_home = elem_home.find_element(By.XPATH, "./../div")
            #print(elem_home)
            elems_home_div = elem_home.find_elements(By.XPATH, "./div/div")
            if len(elems_home_div) == 0:
                elems_home_div = elem_home.find_elements(By.XPATH, "./div[1]/div")
            # 自己紹介がなかった場合は個数が１つ減るのでそれをチェックする
            bNoOverView = len(elems_home_div) < 5
            startNum = 1 if bNoOverView else 2
            #for i,e in enumerate(elems_home_div):
            #    print(str(i) + ":" + e.text)
            homeData["overview"] = elems_home_div[startNum].text
            elems_profileheader = elems_home_div[startNum + 1].find_elements(By.XPATH, "./div/*")
            for pf in elems_profileheader:
                s = pf.get_attribute("data-testid")
                if s:
                    homeData[s] = pf.text
                else:
                    homeData["opendate"] = pf.text
            elems_follow = elems_home_div[-2].find_elements(By.XPATH, "./div")
            homeData["follow"] = self.FixStrNumber(elems_follow[0].text.split()[0])
            homeData["follower"] = self.FixStrNumber(elems_follow[1].text.split()[0])
            
            #
            #print(homeData)
            #
            file_name = "TwitterHome"
            g.SaveData(homeData, file_path, file_name)
            
            savelist = copy.deepcopy(homeData)

            self.Reset()

            return savelist
        except:
            errorchecks = driver.find_elements(By.XPATH, "//div[@role='button']")
            for ec in errorchecks:
                if ec.text == "やりなおす":
                    print("過度なアクセスになっている")
                    time.sleep(600)
                    self.GetTwitterHome(targetID)
                    return
            # 通常のエラー処理
            print("HomeError: " + targetID)
            file_name = "TwitterHome"
            g.SaveData("empty", file_path, file_name)
            pass
        self.Reset()

    # キーワードを指定してそれに関するツイートを取得する
    # wordにIDを指定して、foracountをTrueとすれば、そのIDの発言を取得できる
    # baseFileNameが存在しない場合はファイルの作成は行わない
    def GetTweet(self, word, backday, backcount, offsetday, foracount = True, notreply = True, notlink = True, nothash = True, baseFileName = None, ngwords = None, Overlayed = False):
        driver = self.driver
        #atexit.register(self.Reset)

        # すでに存在している場合は処理しない
        if not Overlayed:
            if baseFileName:
                file_path = "Users/" + baseFileName
                file_name = "Tweet"
                tws = g.LoadData(file_path, file_name)
                if tws:
                    return tws


        if ngwords:
            ngwords += self.globalngwords
        else:
            ngwords = self.globalngwords

        UntilTime = dtime.datetime.now()
        until_y, until_m, until_d = g.GetDateTime(UntilTime)
        # 指定した日にち前
        if offsetday > 0:
            #since_y,since_m,since_d = self.getDateTime(since_y, since_m, since_d, offsetmonth)
            UntilTime = UntilTime - dtime.timedelta(days=offsetday)
            until_y, until_m, until_d = g.GetDateTime(UntilTime)
            
        # 指定した月遡ってデータを得る
        # １月単位でデータを得ることによって読み込み速度を上げる
        SinceTime = UntilTime
        for i in range(backcount):
            UntilTime = SinceTime
            SinceTime = UntilTime - dtime.timedelta(days=backday)
            since_y, since_m, since_d = g.GetDateTime(SinceTime)
            until_y, until_m, until_d = g.GetDateTime(UntilTime)
    
            since = str(since_y) + "-" + str(since_m).zfill(2) + "-" + str(since_d).zfill(2)
            until = str(until_y) + "-" + str(until_m).zfill(2) + "-" + str(until_d).zfill(2)
            print(word + ":" + str(i) + ":" + since + ":" + until)
            #
            url = f"https://twitter.com/search?q="
            # ワードがアカウントIDだった場合にはこちらを使用すればそのIDの発言を取得できる
            if foracount:
                url += "from%3A"
            url += f"{word}"
            # NGワード追加
            for ngword in ngwords:
                url += f"%20-\"{ngword}\""
            # 時間の範囲
            url += f"%20since%3A{since}%20until%3A{until}"
            # リプライを含まない
            if notreply:
                url += "%20-filter%3Areplies"
            # リンクのあるものを含まない、画像も含まない
            if notlink:
                url += "%20-filter%3Alinks"
            # ハッシュタグを含まない
            if nothash:
                url += "%20exclude%3Ahashtags"
            # リツイート・いいね・リプライの範囲を指定する
            if self.mincountretweet > 0:
                url += f"%20min_retweets%3A{self.mincountretweet}"
            if self.maxcountretweet > 0:
                url += f"%20-min_retweets%3A{self.maxcountretweet}"
            if self.mincountlike > 0:
                url += f"%20min_faves%3A{self.mincountlike}"
            if self.maxcountlike > 0:
                url += f"%20-min_faves%3A{self.maxcountlike}"
            if self.mincountreply > 0:
                url += f"%20min_replies%3A{self.mincountreply}"
            if self.maxcountreply > 0:
                url += f"%20-min_replies%3A{self.maxcountreply}"
            url += "%20lang%3Aja&src=typed_query&f=live"
            #
            # url = f"https://twitter.com/search?q={word}%20since%3A{since}%20until%3A{until}%20-filter%3Areplies&src=typed_query&f=live"
            driver.get(url)
            #print(url)
        
            # articleタグが読み込まれるまで待機（最大15秒）
            try:
                self.debugnum = "A"
                WebDriverWait(driver, 3.0).until(EC.visibility_of_element_located((By.TAG_NAME, 'article')))
                listnum = len(self.tweet_list)
                self.getTweetData(baseFileName)
                self.debugnum = "B"
                # 最大取得ツイート数に達していた場合は抜ける
                if self.maxgettweetcount != 0 and self.maxgettweetcount < len(self.tweet_list):
                    break
                # 取得ツイート数によって取得日数を変動させる
                nowlistnum = len(self.tweet_list)
                newlistnum = nowlistnum - listnum
                if newlistnum == 0:
                    backday *= 3
                self.debugnum = "C"
                backday = int((100 / newlistnum) * backday)
                self.debugnum = "D"
                if backday < 0:
                    print(self.myID + " : " + word + ":" + str(i) + ":" + since + ":" + until)
                    print("BACKDAY:" + str(backday) + " : " + str(newlistnum) + " : " + str(listnum) + " : " + str(nowlistnum))
                    backday = abs(backday)

            except:
                print("ERROR:" + self.debugnum)
                pass
            # 
            if backday > 500:
                break

        if len(self.tweet_list) > 0:
            if baseFileName:
                file_path = "Users/" + baseFileName
                file_name = "Tweet"
                g.SaveData(self.tweet_list, file_path, file_name)

            savelist = copy.deepcopy(self.tweet_list)
            
            self.Reset()

            return savelist
        else:
            if baseFileName:
                self.SaveData("empty", file_path, file_name)

        self.Reset()

    # すでに開かれた対象者のTwitterからスクロールしながらデータを取得する
    def getTweetData(self, baseFileName):
        # 指定回数スクロール
        # スクロールして情報を更新しながらそれらの情報を収集していく
        #print(self.tweet_list)
        for i in range(self.max_scroll_count):
            #
            self.debugnum = "AA"
            self.getTweetData_Page(baseFileName)
            self.debugnum = "AB"
            if self.maxgettweetcount != 0 and self.maxgettweetcount < len(self.tweet_list):
                break
            # スクロール＝ページ移動
            if self.scroll_to_elem(self.driver):
                break
            self.debugnum = "AC"
            
            # 待つ（サイトに負荷を与えないと同時にコンテンツの読み込み待ち）
            time.sleep(self.scroll_wait_time)

    def getTweetData_Page(self, baseFileName):
        driver = self.driver
        elems_article = driver.find_elements(By.TAG_NAME, 'article')

        self.debugnum = "BA"

        for elem_article in elems_article:
            if elem_article:
                self.debugnum = "START"
                try:
                    elems_a = elem_article.find_elements(By.TAG_NAME, "a")
                except:
                    self.debugnum = "BBA"
                    continue                
                # ツイートへのURLによってすでにそれが読み込まれているかをチェックする
                self.debugnum = "HOP"
                try:
                    id = elems_a[3].get_attribute("href")
                except:
                    for o,ea in enumerate(elems_a):
                        print(str(o) + " : " + ea.text)
                    self.debugnum = "HREF BAG"
                    continue
                self.debugnum = "BBB"
                id = id.split("status/")[-1]
                self.debugnum = "BBC"
                if id in self.id_list:
                    #print("重複")
                    pass
                else:
                    self.debugnum = "BB"
                    self.id_list.append(id)
                    tweet = {}
                    elems_tw = elem_article.find_element(By.XPATH, ".//div[@lang=\"ja\"]")
                    #
                    tweet["name"] = elems_a[1].text
                    tweet["id"] = elems_a[2].text
                    try:
                        tweet["link"] = id
                    except:
                        print("ERROR:")
                        for i,e in enumerate(elems_a):
                            print(str(i) + ":" + e.text)
                    ttime = elems_a[3].find_element(By.TAG_NAME, "time").get_attribute("datetime")
                    self.debugnum = "BC"
                    tweet["datetime"] = ttime
                    self.debugnum = "BD"
                    tweet["tweet"] = elems_tw.text
                    self.debugnum = "BE"
                    try:
                        tweet["likecount"] = elem_article.find_element(By.XPATH, ".//div[contains(@aria-label, 'いいねする')]").text
                    except:
                        tweet["likecount"] = elem_article.find_element(By.XPATH, ".//div[contains(@aria-label, 'いいねしました')]").text
                    self.debugnum = "BF"
                    try:
                        tweet["retweetcount"] = elem_article.find_element(By.XPATH, ".//div[contains(@aria-label, 'リツイートする')]").text
                    except:
                        tweet["retweetcount"] = elem_article.find_element(By.XPATH, ".//div[contains(@aria-label, 'リツイートしました')]").text
                    self.debugnum = "BG"
                    tweet["replycount"] = elem_article.find_element(By.XPATH, ".//div[contains(@aria-label, '返信する')]").text
                    self.debugnum = "BH"

                    # イメージの保存
                    if baseFileName:
                        try:
                            imgcount = 0
                            #
                            ttime = ttime.replace("-", "_")
                            ttime = ttime.replace(":", "_")
                            ttime = ttime.replace("T", "_")
                            ttime = ttime.split(".")[0]
                            file_path = "Users/" + baseFileName + "/Images"
                            file_name = baseFileName + "_" + ttime + "_" + tweet["id"] + "_" + id
                            elems_img = elem_article.find_elements(By.XPATH, ".//img[contains(@alt, '画像')]")
                            for elem_img in elems_img:
                                photourl = elem_img.get_attribute("src")
                                if "media" in photourl:
                                    imgcount += 1
                                    g.SaveImage(photourl, file_path, file_name + "_" + str(imgcount))
                        except:
                            print("ERROR: IMG " + tweet["name"] + " : " + tweet["id"])
                    

                    self.debugnum = "BI"
                    self.tweet_list.append(tweet)


    # いいねしたものを取得する
    def getLikeTweet(self, targetID):
        driver = self.driver
        
        url = "https://twitter.com/" + targetID + "/likes"
        driver.get(url)
        
        self.getLikeTweetData(url)

        #
        file_path = "Users/" + targetID
        file_name = "TwitterLike"
        g.SaveData(self.tweet_list, file_path, file_name)
        self.Reset()

    # 
    def getLikeTweetData(self, url):
        # 指定回数スクロール
        # スクロールして情報を更新しながらそれらの情報を収集していく
        for i in range(self.max_scroll_count):
            self.getLikeTweetData_Page()
            # スクロール＝ページ移動
            self.scroll_to_elem(self.driver)
            
            # 待つ（サイトに負荷を与えないと同時にコンテンツの読み込み待ち）
            time.sleep(self.scroll_wait_time)

    #
    def getLikeTweetData_Page(self):
        driver = self.driver

        elems_article = driver.find_elements(By.TAG_NAME, 'article')

        for i,elem_article in enumerate(elems_article):
            if elem_article:
                try:
                    elems_a = elem_article.find_elements(By.TAG_NAME, "a")
                    # ツイートへのURLによってすでにそれが読み込まれているかをチェックする
                    id = elems_a[3].get_attribute("href")
                    if id in self.id_list:
                        #print("重複")
                        pass
                    else:
                        self.id_list.append(id)
                        tweet = {}
                        elems_tw = elem_article.find_element(By.XPATH, ".//div[@lang=\"ja\"]")
                        #
                        tweet["name"] = elems_a[1].text
                        tweet["id"] = elems_a[2].text
                        tweet["time"] = elems_a[3].find_element(By.TAG_NAME, "time").get_attribute("datetime")
                        tweet["tweet"] = elems_tw.text
                        tweet["likecount"] = elem_article.find_element(By.XPATH, ".//div[contains(@aria-label, 'いいねする')]").text

                        #print(tweet)
                        self.tweet_list.append(tweet)
                except:
                    pass

    # 特殊なサイトからリツイートのみを取得する
    # これは公式にリツイートを取得しようとすると無理？だからだ
    # 最初の方しか取得できなかったり、そもそも表示されなかったりとか、なんかおかしい
    def getReTweet(self, targetID):
        url = "https://retweetlog.herokuapp.com/"

        driver = self.driver
        driver.get(url)
        #
        WebDriverWait(driver, 15).until(EC.visibility_of_element_located((By.TAG_NAME, 'input')))
        #
        input_id = driver.find_element(By.NAME, "screenName")
        input_id.clear()
        input_id.send_keys(targetID)
        #
        btn_enter = driver.find_element(By.TAG_NAME, "button")
        btn_enter.click()
        time.sleep(5)
        #driver.set_script_timeout(20)
        artnum = 0

        i = 0
        while i < self.maxretweetpage:
            btn_enters = driver.find_elements(By.TAG_NAME, "button")
            btn_enters[1].click()
            # なんか別のタブを開く時があるのでそれは消す``
            if len(driver.window_handles) != 1:
                driver.switch_to.window(driver.window_handles[-1])
                driver.close()
                driver.switch_to.window(driver.window_handles[0])
                time.sleep(1)
            else:
                time.sleep(3)
                i += 1
                elems_article = driver.find_elements(By.TAG_NAME, "iframe")
                nownum = len(elems_article)
                if self.maxretweet < nownum:
                    break
                if nownum != artnum:
                    artnum = nownum
                else:
                    break
        
        #
        user_id = []
        user_name = []
        retweet = []
        #elems_article = driver.find_elements(By.TAG_NAME, "iframe")
        elems_iframe = driver.find_elements(By.TAG_NAME, "iframe")
        
        date = ""
        
        for i,e in enumerate(elems_iframe):
            #print(str(i) + str(e))
            # あまりやると時間かかりすぎる
            if self.maxretweet < i:
                print("ReTweet Open Limit")
                break

            driver.switch_to.frame(e)
            try:
                print(str(i) + "/" + str(len(elems_iframe)))
                date = ""
                # TweetのIDを取得する
                elem_id = driver.find_element(By.XPATH, "//article/a")
                tweet_id = elem_id.get_attribute("href").split("status/")[1].split("?")[0]
                
                # article直下、子の要素でdivをすべて取得する
                elems_div = driver.find_elements(By.XPATH, "//article/div")
                if elems_div:
                    bTalkLink = False
                    
                    # 引用ツイートか
                    try:
                        QuoteTweet = driver.find_element(By.XPATH, "//article/div/article")
                    except:
                        pass
                    #print("div:" + str(len(elems_div)))
                    # 会話へのリンクが存在しているか
                    for i,div in enumerate(elems_div):
                        #print("div:" + str(i))
                        elems_span = div.find_elements(By.TAG_NAME, "span")

                        # 日にちを抜き出す
                        if len(date) == 0:
                            date = re.findall("午.\d+:\d+ · .*日", div.text)

                        for r,span in enumerate(elems_span):
                            if span.text == "Twitterで会話をすべて読む":
                                bTalkLink = True
                    # 画像がある場合はそれらを取得する
                    elems_image = []

                    if bTalkLink:
                        elems_image = elems_div[1].find_elements(By.XPATH, ".//img[@alt=\"画像\"]")
                    else:
                        try:
                            elems_div[2].find_element(By.TAG_NAME, "article")
                        except:
                            elems_image = elems_div[2].find_elements(By.XPATH, ".//img[@alt=\"画像\"]")

                    # 必要な要素を取得していく
                    elems_user = elems_div[0].find_elements(By.TAG_NAME, "span")
                    elems_tweet = elems_div[1].find_elements(By.TAG_NAME, "span")

                    #
                    user_name.append(elems_user[0].text)
                    t_id = elems_user[-1].text
                    user_id.append(t_id)

                    if bTalkLink:
                        retweet.append(self.getTrueTweet(elems_tweet[:-1]))
                    else:
                        retweet.append(self.getTrueTweet(elems_tweet))
                    # 画像の保存
                    file_path = "Users/" + targetID + "/Images"
                    for num,img in enumerate(elems_image):
                        s = ""
                        for d in date:
                            s += d
                        #print(s)
                        date = g.getFixDateTime(s)
                        file_name = targetID + "_" + "RT_" + t_id + "_" + date + "_" + tweet_id + "_" + str(num)
                        imgurl = img.get_attribute("src")
                        g.SaveImage(imgurl, file_path, file_name)
                
            except:
                pass

            driver.switch_to.default_content()

        retweetdata = []
        for i,t in enumerate(retweet):
            #print(user_id[i])
            #print(user_name[i])
            #print(retweet[i])
            tw = {}
            tw["name"] = user_name[i]
            tw["id"] = user_id[i]
            tw["tweet"] = retweet[i]
            #print(tw)
            retweetdata.append(tw)
        file_path = "Users/" + targetID
        file_name = "TwietterReTweet"

        g.SaveData(retweetdata, file_path, file_name)
        self.Reset()

    # ツイート内容の中にリンクが有る場合はうまくそれを取得しないといけない
    # それを含んだツイート全体の文字列を取得する
    def getTrueTweet(self, elems_tweet):
        tweet = ""
        for t in elems_tweet:
            tt = t.text
            # httpsにおいて、URLを取得するための処理
            if tt == "https://":
                elem_url = t.find_element(By.XPATH, "..")
                tweet += elem_url.text
            else:
                tweet += tt
            # 改行して欲しいタイミングで改行を行う
            if tt != " " and not "\n" in tt and not "#" in tt:
                tweet += "\n"
        return tweet
    #
    def scroll_to_elem(self, startElem, targetBy = By.TAG_NAME, targetElem = 'article'):
        # 最後の要素の一つ前までスクロール
        time.sleep(0.2)
        elems_article = startElem.find_elements(targetBy, targetElem)
        last_elem = elems_article[-2]
        
        actions = ActionChains(self.driver);
        actions.move_to_element(last_elem);
        actions.perform();

        if self.LastElemData == last_elem:
            # 前回とデータが変わってないことを知らせる
            return True
        else:
            self.LastElemData = last_elem
            return False


# 対象のIDのワードをチェックして要素ごとに得点と受けて保存する
def CheckWordsTargetID(id, debug = False):
    NL = NLProcessing()

    keywords_P_hobby = {"ゲーム": 3, "スプラ": 3, "スマブラ": 3, "Apex": 6, "プレステ": 2, "スイッチ": 2, "Switch": 1, "格ゲー": 4, "アニメ": 5, "今期": 3, "リズムゲー": 2, "オタク": 2}
    keywords_N_hobby = {"BTS": 5}
    keywords_P_sex = {"私": 3}
    keywords_N_sex = {"俺": 20, "僕": 10, "男": 5}
    keywords_P_job = {"バイト": 3, "コス": 4, "コスプレ": 4, "配信": 2, "投稿": 2, "風俗": 5, "アイドル": 6, "地下": 5, "モデル": 4, "大学": 5, "学生": 3, "学校": 3, "教室": 3, "部活": 3, "試験": 3, "徹夜": 2, "進路": 2, "親友": 2, "友達": 2, "大会": 2, "下校": 3, "放課後": 4, "看護": 2, "ナース": 2, "サークル": 4, "成人式": 2, "ゲーマー": 6, "高校": 3}
    keywords_N_job = {"会社": 1, "営業": 1, "通勤": 3, "残業": 3, "公式": 30, "プレゼント": 30}
    keywords_P_loneli = {"イベント": 4, "出会い": 4, "夏コミ": 6, "サークル": 5, "コミケ": 6, "別れ": 3, "募": 8}
    keywords_N_loneli = {"恋人": 1, "彼氏": 1, "子供": 4, "婚活": 8, "息子": 20, "パートナー": 10, "婚": 10, "旦那": 30}
    keywords_P_home = {"名古屋": 8, "愛知": 5, "新宿": 3, "原宿": 3}
    keywords_N_home = {"九州": 5}
    keywords_P_mental = {"鬱": 3, "孤独": 4, "一人": 4, "誰か": 4, "死": 3, "氏": 3, "どうにか": 3}
    keywords_N_mental = {"パーティ": 6}
    NGWords = ["セフレ", "セクフレ", "糞", "結婚式"]
    HomeNGWords = ["育児", "イラスト", "創設者", "原作者", "Illustrator", "児", "ママ", "代表", "子育て", "母", "運営", "作家", "社長", "カメコ", "Team", "子供", "息子", "夫", "結婚", "漫画家", "イラストレーター", "絵描き", "競馬", "公式アカウント", "阪神", "野球", "公式", "パチンコ", "パチスロ"]
    # 指定したワード軍の中にPワードとNワードがいくつあるかチェックしてポイントをつけていく
    # mulによって倍率を操作できる
    def checkPN(ws, pws, nws, targetkey, mul = 1):
        # 一人称によって男女の判別を行う
        PositiveWords = pws
        NegativeWords = nws
        for w in words:
            for k, v in PositiveWords.items():
                if k == w[0]:
                    lc = w[1] * v * mul
                    #
                    checkkey = targetkey + "_" + k
                    if checkkey in tid:
                        tid[checkkey] += lc
                    else:
                        tid[checkkey] = lc
            for k, v in NegativeWords.items():
                if k == w[0]:
                    lc = w[1] * v * mul
                    #
                    checkkey = targetkey + "_" + k
                    if checkkey in tid:
                        tid[checkkey] += -lc
                    else:
                        tid[checkkey] = -lc

    tid = {}
    id = g.FixTwitterID(id)
    tid["ID"] = id
    
    # ホーム情報の内容をチェックしてく
    hometws = g.LoadData(r"Users/" + id, "TwitterHome")

    if isinstance(hometws, list):
        homestr = ""
        for k, v in hometws[0].items():
            homestr += v
            homestr += "\n"
        
        words, kigous = NL.MakeMorphologicalAnalysis_MeCab(homestr)
        if debug:
            print("Home :" + str(words))
            print(homestr)

        # 絶対的なNGワードがある場合はその時点で排除する
        for w in words:
            for ngw in HomeNGWords:
                if ngw in w:
                    return
        
        # ワードに得点とつけていく
        checkPN(words, keywords_P_sex, keywords_N_sex, "sex", 30)
        checkPN(words, keywords_P_job, keywords_N_job, "job", 30)
        checkPN(words, keywords_P_hobby, keywords_N_hobby, "hobby", 30)
        checkPN(words, keywords_P_loneli, keywords_N_loneli, "loneli", 30)
        checkPN(words, keywords_P_home, keywords_N_home, "home", 30)
        checkPN(words, keywords_P_mental, keywords_N_mental, "mental", 30)

        #print(kigous)

    # ツイートの内容をチェックしていく
    tws = g.LoadData(r"Users/" + id, "Tweet")
    if isinstance(tws, list):
        tweetstr = ""
        twnum = len(tws)
        # ツイート数が少ないものも削除
        if twnum < 10:
            return
        for tw in tws:
            tweetstr += tw["tweet"]
            tweetstr += "\n"

        words, kigous = NL.MakeMorphologicalAnalysis_MeCab(tweetstr)
        #if debug:
            #print("Tweets :" + str(words))

        # ワードに得点とつけていく
        checkPN(words, keywords_P_sex, keywords_N_sex, "sex")
        checkPN(words, keywords_P_job, keywords_N_job, "job")
        checkPN(words, keywords_P_hobby, keywords_N_hobby, "hobby")
        checkPN(words, keywords_P_loneli, keywords_N_loneli, "loneli")
        checkPN(words, keywords_P_home, keywords_N_home, "home")
        checkPN(words, keywords_P_mental, keywords_N_mental, "mental")

        #print(kigous)
    
    # １つ１つファイル化しないとデータ量のせいか知らないが無効なデータ生まれる
    g.SaveData(tid, "damp", "WordPointData_" + id)

if __name__ == '__main__':
    #NL = NLProcessing()
    #s = "⏰本日深夜0:59〜 放送！『熱闘！Mリーグ』に出演します！【麻雀初心者】の私に、Mリーグの魅力や楽しみ方をプレゼンしていただきました✨麻雀あんまり分からないよ〜っていう人も、見ればハマっちゃうかも…！"
    #words, kigous = NL.MakeMorphologicalAnalysis_MeCab(s)
    #print(words)
    #print(kigous)
    if False:
        NL = NLProcessing()
        s = "Illustrator"
        words = NL.MakeMorphologicalAnalysis_MeCab(s)
        print(words)

    if False:
        files = glob.glob("/home/hikachof/デスクトップ/datas/Users/@*")
        fnum = len(files)
        for i, fi in enumerate(files):
            id = fi.split("/")[-1]
            print("CheckWords: " + id + " : " + str(i) + "/" + str(fnum))
            CheckWordsTargetID(id)

    #CheckWordsTargetID("kyame", False)

    if False:
        GT = ScrayTwitter()
        files = glob.glob("/home/hikachof/デスクトップ/datas/damp/WordPointData*")
        #targetid = g.LoadData("damp", "saveids")
        okcount = 0
        saveids = []
        
        for fi in files:
            #
            fname = fi.split("/")[-1]
            fname = fname.split(".")[0]

            data = g.LoadData(r"damp", fname)

            if data:
                data = data[0]
                #print(data)

                print("========================================================================")
                print("========================================================================")
                print(fname)
                    
                # 対象のキーに関連したカウントの合計を得る
                def checkkeycount(targetkey, border = 15):
                    count = 0
                    for k, v in data.items():
                        if targetkey in k:
                            count += v
                            print(k + " : " + str(v) + " : " + str(count))
                    return (count - border) > 0

                if checkkeycount("sex", 15) and (checkkeycount("job", 10) or checkkeycount("hobby", 10)):
                    GT.driver.get(f"https://twitter.com/{data['ID']}")
                    okcount += 1
                    print("OK: " + str(okcount))
                    try:
                        val = input()
                        if len(val) > 0:
                            #print("Save: " + id)
                            saveids.append(id)
                    except:
                        pass
                    #time.sleep(3)
        #g.SaveData(saveids, "damp", "saveids")
        GT.Quit()


    #tws = g.LoadData("Users\@enako_cos", "Tweet")
    #eds = []
    #for tw in tws:
    #    eds.append(g.GetSimplificationDateTime(tw["datetime"])) 

    #
    #g.MakeGraph_date(eds, "year")
    #g.MakeGraph_date(eds, "month")
    #g.MakeGraph_date(eds, "week", "2019-2021")
    #g.MakeGraph_date(eds, "week", "2019-2021", None, None, "22-5")
    #g.MakeGraph_date(eds, "hour")