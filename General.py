from decimal import Clamped
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
from collections import defaultdict
#import androidx.core.math.MathUtils


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
basefolder = "/home/hikachof/デスクトップ/datas"
def MakeFilePath(filepath, filename, fileEx):
    filepath = basefolder + "/" + filepath
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
    #plt.ylim(-20, 20)
    #
    filepath = MakeFilePath("Graphs", filename, "png")
    plt.savefig(filepath)

# keydate : year / month / week / hour
# targetによって指定範囲に絞ってデータを収集することが可能、例えばtargetweekを0とすれば、月曜日飲みを対象とすることができる
# 他にもtargethourを "3-10" とすれば、3時から10時までの時間を対象としてデータを収集することができる。"22-3"というようにまたぐことも可能、また "3,10,22"というように複数指定することもできる
def MakeGraph_date(dates, keydate, targetyear = None, targetmonth = None, targetweek = None, targethour = None):

    # 数値文字によってターゲットとなっている時間を返す
    def targetrange(target, minnum = None, maxnum = None):
        targetnums = []
        if target:
            # , によって複数の数値の指定を行えて "-"によって範囲の指定が可能となっている両方ある場合は無効なデータとする
            if "," in target and "-" in target:
                return None
            #
            if "," in target:
                ts = target.split(",")
                for t in ts:
                    targetnums.append(int(t))
                return targetnums
            # A-Bの範囲のターゲットをすべて配列に入れる、シンプルに以降の処理を統一できるのでこうしている
            elif "-" in target:
                ts = target.split("-")
                # ２つしか有効でない
                tA = int(ts[0])
                tB = int(ts[1])
                # 通常の数値が小大である場合
                if tA < tB:
                    sa = tB - tA
                    for i in range(sa+1):
                        targetnums.append(tA + i)
                elif not maxnum is None and not minnum is None:
                    tA = min(maxnum, max(tA, minnum))
                    tB = min(maxnum, max(tB, minnum))
                    sa = (maxnum - tA) + (tB - minnum) + 1
                    #print(sa)
                    for i in range(sa+1):
                        num = tA + i
                        if maxnum < num:
                            num = minnum + (num - maxnum) - 1
                        #print(num)
                        targetnums.append(num)
            # 普通の数値
            else:
                targetnums.append(int(target))
            return targetnums
    # 
    targets = {}
    targets["year"] = targetrange(targetyear)
    targets["month"] = targetrange(targetmonth, 1, 12)
    targets["week"] = targetrange(targetweek, 0, 6)
    targets["hour"] = targetrange(targethour, 0, 23)

    
    print(targets)

    datecount = defaultdict(int)
    for ed in dates:
        # 対象の時間をチェックする
        def checktarget(targetkey):
            target = targets[targetkey]
            ednum = int(ed[targetkey])
            if target:
                for t in target:
                    if t == ednum:
                        return False
                return True
            return False

        # 複数のターゲットを指定してその範囲？においての情報のみに絞れる
        if checktarget("year"):
            continue
        if checktarget("month"):
            continue
        if checktarget("week"):
            continue
        if checktarget("hour"):
            continue
        # 
        datecount[str(ed[keydate])] += 1

    # Keyの数値の順に変更する
    datecount = sorted(datecount.items(), key=lambda x:int(x[0]), reverse=False)
    #print(datecount)

    xs = []
    ys = []

    # Dateによってのツイートの偏りをグラフデータにする
    if keydate == "week":
        weekstr = ["月曜", "火曜", "水曜", "木曜", "金曜", "土曜", "日曜"]
        for yc in datecount:
            xs.append(weekstr[int(yc[0])])
            ys.append(int(yc[1]))
    else:
        for yc in datecount:
            xs.append(int(yc[0]))
            ys.append(int(yc[1]))
    #
    def makename(target):
        if isinstance(target, str):
            target = target.replace(",", "_")
        return ("_" + str(target) if target else "")
    name = keydate + makename(targetyear) + makename(targetmonth) + makename(targetweek) + makename(targethour)
    MakeGraph([xs], [ys], [keydate], "Tweet on " + name, keydate, "Tweet Count", 10, "enako_cos_tweet_" + name)

# Tweetなどの時間を必要な情報に変換して返す
def GetSimplificationDateTime(date):
    #print(date)
    # 必要な情報は西暦・月・曜日・時間の４つの情報を分けて収納する
    d, t = date.split("T")
    y, m, d = d.split("-")
    t1, t2, t3 = t.split(":")
    #
    dt = datetime.datetime(int(y), int(m), int(d))
    dt.weekday()
    #
    easydate = {}
    easydate["year"] = dt.year
    easydate["month"] = dt.month
    easydate["week"] = dt.weekday()
    easydate["hour"] = int(t1)

    return easydate


# Twitterから取得される日付文字列を最適化する
def getFixDateTime(date):
    #print("A")
    dates = date.split(" · ")
    t = dates[0]
    #print("t" + t)
    d = dates[1]
    #print("d" + d)
    tt = t[2:]
    #print("tt" + tt)
    tt = tt.split(":")
    #print("tt" + tt)
    tt1 = tt[0]
    #print("tt1" + tt1)
    tt2 = tt[1]
    #print("tt2" + tt2)
    if "午後" in t:
        tt1 = str(int(tt1) + 12)
    #
    t = tt1.zfill(2) + "_" + tt2.zfill(2)
    #print("t" + t)
    #
    
    d = d.split("年")
    d1 = d[0]
    #print(d1)
    d = d[1].split("月")
    d2 = d[0]
    #print(d2)
    d = d[1].split("日")
    d3 = d[0]
    #print(d3)
    d2 = d2.zfill(2)
    d3 = d3.zfill(2)
    #print(d1 + "_" + d2 + "_" + d3 + "_" + t)
    date = d1 + "_" + d2 + "_" + d3 + "_" + t
    return date