# -*- coding: utf-8 -*-
import itertools
import json
import re
from collections import Counter
from datetime import datetime

import pandas as pd
import pkuseg
from pyecharts import options as opts
from pyecharts.charts import *
from pyecharts.components import Table
from pyecharts.globals import ThemeType
from pyecharts.options import ComponentTitleOpts
from snownlp import SnowNLP
from tqdm import tqdm


class WeChatAnalysis(object):
    def __init__(
        self,
        message_path_list: list,
        width: int = 1680
    ):
        self.width = width
        self.message_path_list = message_path_list
        self.user_map = {0: "我", 1: "TA", 2: "我们"}
        self.color_map = {0: "#76f2f2", 1: "#fc97af"}
        self.message_type_map = {
            1: "文本",
            3: "图片",
            34: "语音",
            42: "名片",
            43: "视频",
            47: "表情",
            48: "定位",
            49: "分享链接",
            50: "通话",
            62: "小视频",
            10000: f"撤回（{self.user_map[1]}）",
            10002: f"撤回（{self.user_map[0]}）",
        }
        self.num2week = {
            1: "Monday",
            2: "Tuesday",
            3: "Wednesday",
            4: "Thursday",
            5: "Friday",
            6: "Saturday",
            0: "Sunday",
        }
        self.week2num = {
            "Monday": 1,
            "Tuesday": 2,
            "Wednesday": 3,
            "Thursday": 4,
            "Friday": 5,
            "Saturday": 6,
            "Sunday": 0,
        }
        self.seg = pkuseg.pkuseg(postag=True)
        with open("data/stopwords.json") as f:
            self.stopwords = json.load(f)
        with open("data/emotion_ontology.json", "r") as f:
            data = json.load(f)
        self.Joy = data["Joy"]
        self.Like = data["Like"]
        self.Surprise = data["Surprise"]
        self.Anger = data["Anger"]
        self.Depress = data["Depress"]
        self.Fear = data["Fear"]
        self.Dislike = data["Dislike"]
        self.Positive = self.Joy + self.Like + self.Surprise
        self.Negative = self.Anger + self.Depress + self.Fear + self.Dislike
        self.message_list, self.df = self.message_etl

    @staticmethod
    def format_time(create_time, fmt: str = "%Y-%m-%d %H:%M:%S"):
        date = datetime.strftime(datetime.fromtimestamp(create_time), fmt)
        return date

    def emotion_calculate(self, row):
        word_list = []
        content_cut = []
        if "腾讯会议" not in row and "有限公司" not in row:
            pattern = re.compile(r"\[[\u4e00-\u9fa5]+]")
            row = pattern.sub(r"", row)
            pattern = re.compile(r"[\u4e00-\u9fa5]+")
            row = "".join(re.findall(pattern, row))
            if row:
                content_cut = self.seg.cut(row)
                for word, flag in content_cut:
                    word = word.strip().lower()
                    if word and word not in self.stopwords:
                        word_list.append(word)
        positive = 0
        negative = 0
        anger = 0
        dislike = 0
        fear = 0
        depress = 0
        surprise = 0
        like = 0
        joy = 0
        word_set = set(word_list)
        for word in word_set:
            freq = word_list.count(word)
            if word in self.Positive:
                positive += freq
            if word in self.Negative:
                negative += freq
            if word in self.Anger:
                anger += freq
            if word in self.Dislike:
                dislike += freq
            if word in self.Fear:
                fear += freq
            if word in self.Depress:
                depress += freq
            if word in self.Surprise:
                surprise += freq
            if word in self.Like:
                like += freq
            if word in self.Joy:
                joy += freq
        polarity = positive + negative
        sentiment_dict = {
            "Anger": anger,
            "Dislike": dislike,
            "Fear": fear,
            "Like": like,
            "Depress": depress,
            "Surprise": surprise,
            "Joy": joy,
        }
        sentiment_type = max(sentiment_dict.items(), key=lambda x: x[1])[0]

        if sentiment_dict[sentiment_type] == 0:
            sentiment_type = "None"
        return polarity, sentiment_type, content_cut

    @property
    def message_etl(self):
        message_list = []
        data = []
        for message_path in self.message_path_list:
            with open(message_path, encoding="utf-8") as f:
                js_data = json.loads(f.read().replace("var data = ", ""))
                data += js_data["message"]
                self.user_map[1] = js_data["owner"]["name"]
        for row in tqdm(iterable=data, desc=f"「{self.user_map[1]}」聊天记录预处理", total=len(data)):
            create_time = row["m_uiCreateTime"]
            duration = 0
            emoji_md5 = ""
            call_hour = []
            if row.get("m_uiMessageType") == 50:
                dt = re.findall(r"<duration>(\d+)<", row.get("m_nsContent"))
                duration = int(dt[0]) if dt else 0
                start_hour = int(self.format_time(create_time, "%H"))
                end_hour = int(self.format_time(create_time + duration, "%H"))
                if end_hour < start_hour:
                    call_hour = [hour for hour in range(start_hour, 24)] + [hour for hour in range(0, end_hour + 1)]
                else:
                    call_hour = [hour for hour in range(start_hour, end_hour + 1)]
            elif row.get("m_uiMessageType") == 47:
                emoji_md5 = re.findall(r"md5=\"(.*?)\"", row.get("m_nsContent"))[0]
            if row.get("m_uiMessageType") == 1:
                polarity, sentiment_type, content_cut = self.emotion_calculate(row.get("m_nsContent"))
                s = SnowNLP(row.get("m_nsContent"))
                sentiments = s.sentiments
            else:
                polarity = 0
                sentiments = 0
                sentiment_type = "None"
                content_cut = []
            item = {
                "user": 1 if row["m_nsFromUsr"] else 0,
                "create_time": self.format_time(create_time),
                "message_type": row["m_uiMessageType"],
                "content": row["m_nsContent"],
                "date": self.format_time(create_time, "%Y-%m-%d"),
                "year": self.format_time(create_time, "%Y"),
                "month": self.format_time(create_time, "%m"),
                "year_month": self.format_time(create_time, "%Y-%m"),
                "week": self.week2num.get(self.format_time(create_time, "%A")),
                "day": self.format_time(create_time, "%d"),
                "hour": self.format_time(create_time, "%H"),
                "minute": self.format_time(create_time, "%M"),
                "second": self.format_time(create_time, "%S"),
                "duration": duration,
                "call_hour": call_hour,
                "emoji_md5": emoji_md5,
                "polarity": polarity,
                "sentiment_type": sentiment_type,
                "sentiments": sentiments,
                "content_cut": content_cut,
                "content_len": len(row["m_nsContent"]) if row["m_uiMessageType"] == 1 else 0,
            }
            message_list.append(item)
        df = pd.DataFrame(message_list)
        return message_list, df

    @staticmethod
    def title_name(title, subtitle, title_style):
        table = Table()
        table.set_global_opts(
            title_opts=ComponentTitleOpts(
                title=title,
                subtitle=subtitle,
                title_style=title_style,
                subtitle_style={"style": "font-size: 16px; font-weight:bold;"},
            )
        )
        return table

    def chat_info(self):
        start_date = datetime.strptime(self.message_list[0].get("create_time"), "%Y-%m-%d %H:%M:%S")
        end_date = datetime.strptime(self.message_list[-1].get("create_time"), "%Y-%m-%d %H:%M:%S")
        duration_hour = self.df["duration"].sum() / 60 / 60
        avg_duration_hour = duration_hour / (end_date - start_date).days
        user_info = {
            f"第一次聊天": str(start_date),
            f"我们认识已经": f"{end_date - start_date}",
            f"我们累计通话时长": f"{duration_hour:.2f} 小时, {avg_duration_hour:.2f} 小时/天",
            f"{self.user_map[0]}发送的消息数": self.df[(self.df["user"] == 0)].shape[0],
            f"{self.user_map[1]}发送的消息数": self.df[(self.df["user"] == 1)].shape[0],
            f"{self.user_map[0]}发送的总字数": self.df[(self.df["user"] == 0)]["content_len"].sum(),
            f"{self.user_map[1]}发送的总字数": self.df[(self.df["user"] == 1)]["content_len"].sum(),
            f"{self.user_map[0]}发送长消息数": self.df[(self.df["user"] == 0) & (self.df["content_len"] >= 50)][
                "content_len"
            ].shape[0],
            f"{self.user_map[1]}发送长消息数": self.df[(self.df["user"] == 1) & (self.df["content_len"] >= 50)][
                "content_len"
            ].shape[0],
            f"{self.user_map[0]}发送最长消息": self.df[(self.df["user"] == 0) & (self.df["content_len"] >= 50)][
                "content_len"
            ].max(),
            f"{self.user_map[1]}发送最长消息": self.df[(self.df["user"] == 1) & (self.df["content_len"] >= 50)][
                "content_len"
            ].max(),
        }
        chat_info = {
            f"消息总条数": self.df.shape[0],
            f"文字": self.df[(self.df["message_type"] == 1)].shape[0],
            f"图片": self.df[(self.df["message_type"] == 3)].shape[0],
            f"语音": self.df[(self.df["message_type"] == 34)].shape[0],
            f"名片": self.df[(self.df["message_type"] == 42)].shape[0],
            f"视频": self.df[(self.df["message_type"] == 43) | (self.df["message_type"] == 62)].shape[0],
            f"表情": self.df[(self.df["message_type"] == 47)].shape[0],
            f"定位": self.df[(self.df["message_type"] == 48)].shape[0],
            f"通话": self.df[(self.df["message_type"] == 50)].shape[0],
            f"撤回": self.df[(self.df["message_type"] == 10000) | (self.df["message_type"] == 10002)].shape[0],
            f"分享链接": self.df[(self.df["message_type"] == 49)].shape[0],
        }
        user_info = "\n".join([f"{key}: {value}" for key, value in user_info.items()])
        chat_info = "\n".join([f"{key}: {value}" for key, value in chat_info.items()])
        return [
            self.title_name(title="", subtitle=user_info, title_style=None),
            self.title_name(title="", subtitle=chat_info, title_style=None),
        ]

    @staticmethod
    def calendar_render(data, title, max_num, min_num=0) -> list:
        calendar_list = []
        for year, value in data.items():
            calendar = (
                Calendar(init_opts=opts.InitOpts(theme=ThemeType.MACARONS))
                .add(
                    series_name="",
                    yaxis_data=[[day, int(count)] for day, count in value.items() if count],
                    calendar_opts=opts.CalendarOpts(
                        range_=str(year),
                        daylabel_opts=opts.CalendarDayLabelOpts(name_map="cn"),
                        monthlabel_opts=opts.CalendarMonthLabelOpts(name_map="cn"),
                    ),
                )
                .set_global_opts(
                    title_opts=opts.TitleOpts(title=title, subtitle=year),
                    legend_opts=opts.LegendOpts(is_show=True),
                    visualmap_opts=opts.VisualMapOpts(
                        max_=int(max_num),
                        min_=int(min_num),
                        orient="horizontal",
                        is_piecewise=True,
                        pos_top="230px",
                        pos_left="100px",
                    ),
                )
            )
            calendar_list.append(calendar)
        return calendar_list

    def wordcloud_render(self, data) -> list:
        wc_list = []
        for user, value in data.items():
            wc = WordCloud(init_opts=opts.InitOpts(theme=ThemeType.MACARONS))
            data_pair = sorted(list(Counter(value).items()), key=lambda t: t[1], reverse=True)
            wc.add(
                series_name=f"聊天词云({self.user_map.get(user)})",
                data_pair=data_pair,
                word_size_range=[10, 100],
                shape="diamond",
            )
            wc.set_global_opts(
                title_opts=opts.TitleOpts(
                    title=f"聊天词云({self.user_map.get(user)})", title_textstyle_opts=opts.TextStyleOpts(font_size=23)
                ),
                tooltip_opts=opts.TooltipOpts(is_show=True),
            )
            wc_list.append(wc)
        return wc_list

    def bar_line_render(self, data, title, reversal_axis=False, off=False, is_int=True, is_bar=True):
        if is_bar:
            bar = Bar(init_opts=opts.InitOpts(theme=ThemeType.MACARONS))
        else:
            bar = Line(init_opts=opts.InitOpts(theme=ThemeType.MACARONS))
        for user, value in data.items():
            bar.add_xaxis([self.num2week.get(one, one) for one in list(value.keys())])
            if is_int:
                y_values = [int(one) for one in list(value.values())]
            else:
                y_values = list(value.values())
            bar.add_yaxis(
                self.user_map.get(user),
                y_values,
                itemstyle_opts={"normal": {"color": self.color_map.get(user, "#87f7cf")}},
            )
        bar.set_global_opts(
            title_opts=opts.TitleOpts(title=title),
            legend_opts=opts.LegendOpts(pos_right="15%"),
        )
        if reversal_axis:
            bar.reversal_axis()
        if off:
            bar.set_series_opts(label_opts=opts.LabelOpts(position="insideLeft", formatter="{b}:{c}"))
            bar.set_global_opts(xaxis_opts=opts.AxisOpts(is_show=False), yaxis_opts=opts.AxisOpts(is_show=False))
        return bar

    def radar_render(self, data):
        radar = (
            Radar(init_opts=opts.InitOpts(theme=ThemeType.MACARONS))
            .add_schema(schema=[opts.RadarIndicatorItem(name=one) for one in self.message_type_map.values()])
            .add("", data)
            .set_global_opts(title_opts=opts.TitleOpts(title="聊天记录类型图"))
        )
        return radar

    @staticmethod
    def heatmap_render(data, max_num, title):
        hour_list = [str(i) for i in range(24)]
        week_list = ["周日", "周一", "周二", "周三", "周四", "周五", "周六"]

        heat = (
            HeatMap(init_opts=opts.InitOpts(theme=ThemeType.MACARONS))
            .add_xaxis(hour_list)
            .add_yaxis("", week_list, data)
            .set_global_opts(
                title_opts=opts.TitleOpts(title=title),
                visualmap_opts=opts.VisualMapOpts(
                    min_=0,
                    max_=max_num,
                    is_calculable=True,
                    orient="horizontal",
                    pos_left="center",
                    is_piecewise=True,
                ),
            )
        )
        return heat

    def year_analyser(self):
        result = self.df.groupby(["user", "year"]).size().unstack(fill_value=0).to_dict("index")
        return self.bar_line_render(result, "每年发送消息频率图")

    def month_analyser(self):
        result = self.df.groupby(["user", "month"]).size().unstack(fill_value=0).to_dict("index")
        return self.bar_line_render(result, "每月发送消息频率图")

    def week_analyser(self):
        result = self.df.groupby(["user", "week"]).size().unstack(fill_value=0).to_dict("index")
        return self.bar_line_render(result, "每周发送消息频率图")

    def day_analyser(self):
        result = self.df.groupby(["user", "day"]).size().unstack(fill_value=0).to_dict("index")
        return self.bar_line_render(result, "每天发送消息频率图")

    def hour_analyser(self):
        result = self.df.groupby(["user", "hour"]).size().unstack(fill_value=0).to_dict("index")
        return self.bar_line_render(result, "每时发送消息频率图")

    def date_analyser(self):
        result = self.df.groupby(["year", "date"]).size().unstack(fill_value=0).to_dict("index")
        max_num = self.df.groupby(["date"]).size().max()
        return self.calendar_render(result, "每年聊天频率图", max_num)

    def polarity_analyser(self):
        max_num = self.df.groupby(["date"])["polarity"].sum().max()
        min_num = self.df.groupby(["date"])["polarity"].sum().min()
        result = self.df.groupby(["year", "date"])["polarity"].sum().unstack(fill_value=0).to_dict("index")
        return self.calendar_render(result, "每年聊天情感得分日历图(Emotion Ontology)", max_num, min_num)

    def polarity_month_analyser(self):
        result = self.df.groupby(["user", "year_month"])["polarity"].sum().unstack(fill_value=0).to_dict("index")
        return self.bar_line_render(result, "每月聊天情感得分走势图(Emotion Ontology)", is_bar=False)

    def polarity_day_analyser(self):
        result = self.df.groupby(["user", "day"])["polarity"].sum().unstack(fill_value=0).to_dict("index")
        return self.bar_line_render(result, "每日聊天情感得分走势图(Emotion Ontology)")

    def sentiments_analyser(self):
        max_num = self.df.groupby(["date"])["sentiments"].sum().max()
        result = self.df.groupby(["year", "date"])["sentiments"].sum().unstack(fill_value=0).to_dict("index")
        return self.calendar_render(result, "每年聊天情感得分日历图(SnowNLP)", max_num)

    def sentiments_month_analyser(self):
        result = self.df.groupby(["user", "year_month"])["sentiments"].sum().unstack(fill_value=0).to_dict("index")
        return self.bar_line_render(result, "每月聊天情感得分走势图(SnowNLP)", is_bar=False)

    def sentiments_day_analyser(self):
        result = self.df.groupby(["user", "day"])["sentiments"].sum().unstack(fill_value=0).to_dict("index")
        return self.bar_line_render(result, "每日聊天情感得分走势图(SnowNLP)")

    def sentiments_heatmap_analyser(self):
        df = self.df[self.df["sentiment_type"] != "None"]
        result = df.groupby(["year_month", "sentiment_type"]).size().unstack(fill_value=0).to_dict()
        data = []
        max_num = 0
        for week, value in result.items():
            for hour, count in value.items():
                if count > max_num:
                    max_num = count
                data.append([hour, week, int(count)])
        week_list = ["Like", "Depress", "Anger", "Dislike", "Fear", "Joy", "Surprise"]
        hour_list = list(set(list(df["year_month"].to_list())))
        hour_list.sort()
        heat = (
            HeatMap(init_opts=opts.InitOpts(theme=ThemeType.MACARONS))
            .add_xaxis(hour_list)
            .add_yaxis("", week_list, data)
            .set_global_opts(
                title_opts=opts.TitleOpts(title="情感热力图（每月）"),
                visualmap_opts=opts.VisualMapOpts(
                    min_=0,
                    max_=max_num,
                    is_calculable=True,
                    orient="horizontal",
                    pos_left="center",
                    is_piecewise=True,
                ),
            )
        )
        return heat

    def sentiments_week_heatmap_analyser(self):
        df = self.df[self.df["sentiment_type"] != "None"]
        result = df.groupby(["week", "sentiment_type"]).size().unstack(fill_value=0).to_dict()
        data = []
        max_num = 0
        for week, value in result.items():
            for hour, count in value.items():
                if count > max_num:
                    max_num = count
                data.append([hour, week, int(count)])
        week_list = ["Like", "Depress", "Anger", "Dislike", "Fear", "Joy", "Surprise"]
        hour_list = list(set(list(df["week"].to_list())))
        hour_list.sort()
        heat = (
            HeatMap(init_opts=opts.InitOpts(theme=ThemeType.MACARONS))
            .add_xaxis(hour_list)
            .add_yaxis("", week_list, data)
            .set_global_opts(
                title_opts=opts.TitleOpts(title="情感热力图（星期）"),
                visualmap_opts=opts.VisualMapOpts(
                    min_=0,
                    max_=max_num,
                    is_calculable=True,
                    orient="horizontal",
                    pos_left="center",
                    is_piecewise=True,
                ),
            )
        )
        return heat

    def content_len_month_analyser(self):
        result = self.df.groupby(["user", "year_month"])["content_len"].mean().unstack(fill_value=0).to_dict("index")
        return self.bar_line_render(result, "每月聊天消息长度", is_int=False, is_bar=False)

    def content_len_day_analyser(self):
        result = self.df.groupby(["user", "day"])["content_len"].mean().unstack(fill_value=0).to_dict("index")
        return self.bar_line_render(result, "每日聊天消息长度", is_int=False)

    def week_hour_analyser(self):
        result = self.df.groupby(["hour", "week"]).size().unstack(fill_value=0).to_dict()
        data = []
        max_num = 0
        for week, value in result.items():
            for hour, count in value.items():
                if count > max_num:
                    max_num = count
                data.append([int(hour), week, int(count)])
        return self.heatmap_render(data, max_num, "每周时聊天热力图")

    def week_hour_duration_analyser(self):
        data = []
        date_hour = {}
        for one in self.message_list:
            if one.get("message_type") == 50:
                if one.get("date") not in date_hour.keys():
                    date_hour[one.get("date")] = []
                date_hour[one.get("date")] += one.get("call_hour")
        week_hour_map = {}
        for one_date, call_hour in date_hour.items():
            week = datetime.strptime(one_date, "%Y-%m-%d").weekday()
            call_hour = list(set(call_hour))
            if week not in week_hour_map.keys():
                week_hour_map[week] = []
            week_hour_map[week] += call_hour
        max_num = 0
        for one_week, week_hour in week_hour_map.items():
            for hour, count in sorted(list(Counter(week_hour).items()), key=lambda t: t[0]):
                if count > max_num:
                    max_num = count
                data.append([hour, one_week, count])
        return self.heatmap_render(data, max_num, "每周时通话时长热力图")

    def duration_analyser(self):
        result = self.df.groupby(["year", "date"]).sum()["duration"].unstack(fill_value=0).to_dict("index")
        max_num = self.df.groupby(["date"]).sum()["duration"].max()
        return self.calendar_render(result, "每年通话时长频率图", max_num)

    def emoji_analyser(self):
        pattern = re.compile(r"\[[\u4e00-\u9fa5]+]")
        wo_emoji = re.findall(
            pattern, "".join(list(self.df[(self.df["user"] == 0) & (self.df["message_type"] == 1)]["content"]))
        )
        ta_emoji = re.findall(
            pattern, "".join(list(self.df[(self.df["user"] == 1) & (self.df["message_type"] == 1)]["content"]))
        )
        wo = dict(Counter(wo_emoji).most_common(15))
        ta = dict(Counter(ta_emoji).most_common(15))
        wo = sorted(wo.items(), key=lambda x: x[1], reverse=False)
        ta = sorted(ta.items(), key=lambda x: x[1], reverse=False)
        bar_list = [
            self.bar_line_render({0: dict(wo)}, f"表情排行榜（{self.user_map[0]}）", reversal_axis=True, off=True),
            self.bar_line_render({1: dict(ta)}, f"表情排行榜（{self.user_map[1]}）", reversal_axis=True, off=True),
        ]
        return bar_list

    def message_type_analyser(self):
        result = self.df.groupby(["user", "message_type"]).size().unstack(fill_value=0).to_dict("index")
        data = [
            [result[0].get(one, 0) for one in self.message_type_map.keys()],
            [result[1].get(one, 0) for one in self.message_type_map.keys()],
        ]
        return self.radar_render(data)

    def content_analyser(self):
        wo_content = list(self.df[(self.df["user"] == 0) & (self.df["message_type"] == 1)]["content_cut"])
        ta_content = list(self.df[(self.df["user"] == 1) & (self.df["message_type"] == 1)]["content_cut"])
        data = []
        for content in [wo_content, ta_content]:
            result = []
            for row in content:
                if row:
                    for word, flag in row:
                        word = word.strip().lower()
                        if word and word not in self.stopwords:
                            result.append(word)
            data.append(result)
        data = {index: value for index, value in enumerate(data)}
        return self.wordcloud_render(data)

    def content_graph_analyser(self):
        wo_content = list(self.df[(self.df["user"] == 0) & (self.df["message_type"] == 1)]["content_cut"])
        ta_content = list(self.df[(self.df["user"] == 1) & (self.df["message_type"] == 1)]["content_cut"])

        my_nodes = []
        your_nodes = []
        nodes = [
            {
                "name": self.user_map.get(0),
                "symbolSize": 10,
                "draggable": "False",
                "label": {
                    "normal": {"show": "True"},
                },
                "value": 1,
            },
            {
                "name": self.user_map.get(1),
                "symbolSize": 10,
                "draggable": "False",
                "label": {
                    "normal": {"show": "True"},
                },
                "value": 1,
            },
        ]
        links = []
        for index, content in enumerate([wo_content, ta_content]):
            result = []
            for row in content:
                if row:
                    for word, flag in row:
                        if flag in ("n", "nr", "ns", "nt", "nw", "nz"):
                            word = word.strip().lower()
                            if word and word not in self.stopwords and len(word) > 1:
                                result.append(f"{word} | {flag}")
            result = sorted(list(Counter(result).items()), key=lambda t: t[1], reverse=True)
            for key, count in result[:100]:
                if index == 0:
                    my_nodes.append(key)
                else:
                    your_nodes.append(key)

        my_nodes = set(my_nodes) - set(self.user_map.values())
        your_nodes = set(your_nodes) - set(self.user_map.values())
        same_nodes = my_nodes & your_nodes
        for same_node in same_nodes:
            same_node_name, same_category = same_node.split(" | ")
            nodes.append(
                {
                    "name": same_node,
                    "symbolSize": 5,
                    "category": same_category,
                    "draggable": "False",
                    "label": {
                        "normal": {"show": "True"},
                    },
                    "value": 1,
                }
            )
            links.append({"source": self.user_map.get(0), "target": same_node})
            links.append({"target": self.user_map.get(0), "source": same_node})
            links.append({"source": self.user_map.get(1), "target": same_node})
            links.append({"target": self.user_map.get(1), "source": same_node})
        my_nodes = my_nodes - same_nodes
        for my_node in my_nodes:
            my_node_name, my_category = my_node.split(" | ")
            nodes.append(
                {
                    "name": my_node,
                    "symbolSize": 5,
                    "category": my_category,
                    "draggable": "False",
                    "label": {
                        "normal": {"show": "True"},
                    },
                    "value": 1,
                }
            )
            links.append({"source": self.user_map.get(0), "target": my_node})
            links.append({"target": self.user_map.get(0), "source": my_node})
        your_nodes = your_nodes - same_nodes
        for your_node in your_nodes:
            your_node_name, your_category = your_node.split(" | ")
            nodes.append(
                {
                    "name": your_node,
                    "symbolSize": 5,
                    "category": your_category,
                    "draggable": "False",
                    "label": {
                        "normal": {"show": "True"},
                    },
                    "value": 1,
                }
            )
            links.append({"source": self.user_map.get(1), "target": your_node})
            links.append({"target": self.user_map.get(1), "source": your_node})
        categories = [
            {"name": ""},
            {"name": "n"},
            {"name": "nr"},
            {"name": "ns"},
            {"name": "nt"},
            {"name": "nw"},
            {"name": "nz"},
        ]
        c = (
            Graph()
            .add(
                "",
                nodes,
                links,
                categories,
                repulsion=1000,
                linestyle_opts=opts.LineStyleOpts(curve=0.2),
                label_opts=opts.LabelOpts(is_show=False),
            )
            .set_global_opts(
                legend_opts=opts.LegendOpts(is_show=False),
                title_opts=opts.TitleOpts(title="关系图"),
            )
        )
        return c

    def main(self):
        title = self.title_name(
            f"{self.user_map[1]} | 聊天记录分析",
            subtitle="",
            title_style={"style": "font-size: 32px; font-weight:bold;text-align: center;"},
        )
        chart_config = [
            {
                "cid": title.chart_id,
                "width": f"{self.width}px",
                "height": "70px",
                "top": "0px",
                "left": "0px",
            }
        ]
        page = Page(
            layout=Page.DraggablePageLayout,
            page_title=f"{self.user_map[1]} | 聊天记录分析",
        )
        chart_list = [
            *self.chat_info(),
            *list(itertools.chain(*zip(self.date_analyser(), self.duration_analyser()))),
            self.year_analyser(),
            self.month_analyser(),
            self.week_analyser(),
            self.day_analyser(),
            self.week_hour_analyser(),
            self.week_hour_duration_analyser(),
            self.hour_analyser(),
            self.message_type_analyser(),
            *list(itertools.chain(*zip(self.polarity_analyser(), self.sentiments_analyser()))),
            self.polarity_month_analyser(),
            self.sentiments_month_analyser(),
            self.polarity_day_analyser(),
            self.sentiments_day_analyser(),
            self.content_len_month_analyser(),
            self.content_len_day_analyser(),
            self.sentiments_heatmap_analyser(),
            self.sentiments_week_heatmap_analyser(),
            *self.emoji_analyser(),
            *self.content_analyser(),
            self.content_graph_analyser(),
        ]
        max_card_num = len(chart_list)
        for index, cid in enumerate([one.chart_id for one in chart_list]):
            if index + 3 < max_card_num:
                item = {
                    "cid": cid,
                    "width": f"{self.width / 2 - 10}px",
                    "height": "300px",
                    "top": f"{index // 2 * 300 + 70}px",
                    "left": f"{index % 2 * self.width / 2 + ((index % 2 * (- 1)) + 1) * 10}px",
                }
            elif index + 1 < max_card_num:
                item = {
                    "cid": cid,
                    "width": f"{self.width / 2 - 10}px",
                    "height": f"{self.width / 2 - 10}px",
                    "top": f"{index // 2 * 300 + 70}px",
                    "left": f"{index % 2 * self.width / 2 + ((index % 2 * (- 1)) + 1) * 10}px",
                }
            else:
                item = {
                    "cid": cid,
                    "width": f"{self.width - 20}px",
                    "height": f"{self.width / 2}px",
                    "top": f"{index // 2 * 300 + 70 + self.width / 2 - 300}px",
                    "left": f"10px",
                }
            chart_config.append(item)
        with open("result/chart_config.json", "w") as f:
            f.write(json.dumps(chart_config, ensure_ascii=False))
        page.add(title, *chart_list)
        page.render("result/render.html")
        Page.save_resize_html(
            source="result/render.html",
            cfg_file="result/chart_config.json",
            dest=f"result/{self.user_map[1]}.html",
        )


if __name__ == "__main__":
    import pygame

    pygame.init()
    info = pygame.display.Info()
    wa = WeChatAnalysis(
        message_path_list=[f"xxx/js/message.js"],
        width=info.current_w
    )
    wa.main()
