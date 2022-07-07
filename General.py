import io
import os
import sys
import pickle
import re
import datetime
import json
import numpy as np
import pandas as pd
from PIL import Image
from urllib import request

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
#%matplotlib inline
import japanize_matplotlib
from operator import itemgetter


#########################################################################################################

def utc_to_jst(timestamp_utc):
    datetime_utc = datetime.datetime.strptime(timestamp_utc + "+0000", "%Y-%m-%d %H:%M:%S.%f%z")
    datetime_jst = datetime_utc.astimezone(datetime.timezone(datetime.timedelta(hours=+9)))
    timestamp_jst = datetime.datetime.strftime(datetime_jst, '%Y-%m-%d %H:%M:%S')
    return timestamp_jst

def shape_data(data):
    for i, d in enumerate(data):
        # URLの削除
        data[i]['text'] = re.sub('[ 　]https://t\.co/[a-zA-Z0-9]+', '', d['text'])
        # ユーザー名の削除
        data[i]['text'] = re.sub('[ 　]?@[a-zA-Z0-9_]+[ 　]', '', d['text'])
        # 絵文字の除去
        data[i]['text'] = d['text'].encode('cp932',errors='ignore').decode('cp932')
#         # ハッシュタグの削除
#         data[i]['text'] = re.sub('#.+ ', '', d['text'])
        # 全角スペース、タブ、改行を削除
        data[i]['text'] = re.sub(r"[\u3000\t\n]", "", d['text'])
        # 日付時刻の変換（UTCからJST）
        data[i]['created_at'] = utc_to_jst(d['created_at'].replace('T', ' ')[:-1])
    return data


# 適切なファイルパスとファイル名を作成する
basefolder = "C:\\Users\\hikac\\Desktop\\datas\\"
def MakeFilePath(filepath, filename, fileEx):
    filepath = basefolder + filepath
    # そこまでのフォルダーの作成を行う
    try:
        os.makedirs(filepath)
    except:
        pass
    return filepath + "/" + filename + "." + fileEx

# 保存したいデータを渡すことによって書き出しを行う
def SaveData(savedata, file_path, file_name):
    filepath = MakeFilePath(file_path, file_name, "json")
    # 内部にdictが存在しているかが重要
    typecheck = savedata
    while True:
        if isinstance(typecheck, list):
            typecheck = typecheck[0]
        else:
            break
    # jsonとして保存
    if isinstance(typecheck, dict):
        try:
            df = pd.DataFrame(savedata, index=["i",])
        except:
            df = pd.DataFrame(savedata)
        df.to_json(filepath, orient="records", force_ascii=False)
    else:
        filepath = filepath.replace(".json", "")
        np.save(filepath, savedata)
        
    
    return filepath

# 単純にデータを読み込む
def LoadData(file_path, file_name):
    filepath = MakeFilePath(file_path, file_name, "json")
    try:
        jo = open(filepath, "r", encoding="utf-8")
        return json.load(jo)
    except:
        try:
            filepath = filepath.replace("json", "npy")
            return np.load(filepath)
        except:
            pass
    #
    return None

# 渡したデータをダンプ形式で保存する
def SaveArticle(savedata, filename):
    sys.setrecursionlimit(100000)#エラー回避
    filepath = MakeFilePath("temp", "temp_" + filename, "txt")
    if False:
        # こっちの場合は特定の部分でエラーが出る、理由は不明
        joblib.dump(savedata, filename, compress=3)
    else:
        with open(filepath, "wb") as f:
            pickle.dump(savedata, f)

def LoadArticle(filename):
    filepath = MakeFilePath("temp", "temp_" + filename, "txt")
    with open(filepath, "rb") as f:
        return pickle.load(f)

# URLを渡すことによってそのイメージを保存する
# ユニーク名前を付けるためにグローバルなカウントを使用する
def SaveImage(url, file_path, file_name):
    f = io.BytesIO(request.urlopen(url).read())
    img = Image.open(f)

    #
    filepath = MakeFilePath(file_path, file_name, "jpg")
    img.save(filepath)
    img.close()

# 指定したDateTimeデータから年、月、日を得る
def GetDateTime(dt):
    return dt.year, dt.month, dt.day


# グラフの作成を行う
# xs,ysには関連した値を配列として渡すと複数のグラフを同時に設定できる
def MakeGraph(xs, ys, labels, title, xlabel, ylabel, size, filename):
    colors = ["red", "#87ceeb", "yellow", "blue", "white", "black"]

    # それぞれの配列数は一致していなければいけないのでチェックする
    xsnum = len(xs)
    ysnum = len(ys)
    lsnum = len(labels)
    if xsnum != ysnum or ysnum != lsnum:
        print("ERROR Match Graph")
        return
    #
    plt.figure(figsize=(size, size))
    for i,ls in enumerate(labels):
        x = xs[i]
        y = ys[i]
        c = colors[i]
        plt.plot(x, y, color=c, linewidth=3, linestyle="solid", marker="o", label=ls)
    #
    plt.xticks(fontsize=size)
    plt.yticks(fontsize=size)
    plt.xlabel(xlabel, fontsize=size)
    plt.ylabel(ylabel, fontsize=size)
    plt.grid(axis="x")
    plt.title(title, fontsize=size)
    plt.legend(fontsize=size)
    plt.ylim(-20, 20)
    #
    filepath = MakeFilePath("Graphs", filename, "png")
    plt.saveifg(filepath)


#########################################################################################################