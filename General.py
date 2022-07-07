import re
import datetime

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

#########################################################################################################