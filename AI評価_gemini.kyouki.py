import streamlit as st
import google.generativeai as genai
from sklearn.feature_extraction.text import TfidfVectorizer
from janome.tokenizer import Tokenizer
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.font_manager as fm
import numpy as np
import math
from collections import defaultdict, Counter
import os
import re

# Gemini APIの設定
genai.configure(api_key="AIzaSyBCCBXnKv6nniqcrqoi-st8KOincxE102g")
model = genai.GenerativeModel('gemini-pro')

# Janomeトークナイザー
tokenizer = Tokenizer()

def tokenize(text):
    tokens = tokenizer.tokenize(text)
    return [token.surface for token in tokens if token.surface != 'の' and token.part_of_speech.split(',')[0] in ['名詞', '代名詞']]

def extract_keywords(text):
    vectorizer = TfidfVectorizer(tokenizer=tokenize,token_pattern=None)
    vectors = vectorizer.fit_transform([text])
    feature_names = vectorizer.get_feature_names_out()
    dense = vectors.todense()
    denselist = dense.tolist()
    keywords = sorted([(word, score) for word, score in zip(feature_names, denselist[0]) if score > 0.1], key=lambda x: x[1], reverse=True)
    return keywords

# 共起ネットワーク関連の関数
def find_cooccurrences(text, keywords, window_size=30):
    cooccurrence_matrix = defaultdict(Counter)
    tokens = tokenize(text)
    for i in range(len(tokens)):
        if tokens[i] in [k[0] for k in keywords]:
            window = tokens[i+1:i+1+window_size]
            for word in window:
                if word in [k[0] for k in keywords]:
                    cooccurrence_matrix[tokens[i]][word] += 1
                    cooccurrence_matrix[word][tokens[i]] += 1
    return cooccurrence_matrix

def draw_cooccurrence_network(cooccurrences, tfidf_scores):
    G = nx.Graph()
    for word, cooccur in cooccurrences.items():
        for co_word, weight in cooccur.items():
            G.add_edge(word, co_word, weight=weight)
    
    pos = nx.spring_layout(G)
    weights = nx.get_edge_attributes(G, 'weight')
    
    # 日本語フォントを指定
    font_paths = [
        "C:\\windows\\Fonts\\YUMIN.TTF",
        "C:\\windows\\Fonts\\YUMINDB.TTF",
        "C:\\windows\\Fonts\\YUMINL.TTF"
    ]
    for font_path in font_paths:
        if os.path.exists(font_path):
            prop = fm.FontProperties(fname=font_path)
            break
    else:
        print("指定されたフォントが見つかりません。表示に問題がある可能性があります。")
        prop = fm.FontProperties()

    # ノードのサイズと色を設定
    node_sizes = []
    for node in G.nodes():
        node_sizes.append(tfidf_scores.get(node, 0) * 30000)

    # エッジの幅を設定
    edge_widths = []
    for (u, v, d) in G.edges(data=True):
        log_weight = math.log(d['weight'] + 1)
        scaled_width = log_weight * 2
        edge_widths.append(scaled_width)

    # ノードの色をカラーマップを使用して変更
    colors_array = cm.Pastel2(np.linspace(0.1, 0.9, len(G.nodes())))
    node_colors = [colors_array[i % len(colors_array)] for i in range(len(G.nodes()))]

    fig, ax = plt.subplots(figsize=(12, 8))
    nx.draw(G, pos, ax=ax, with_labels=True, node_color=node_colors, node_size=node_sizes, font_size=10, font_weight='bold', font_family=prop.get_name())
    nx.draw_networkx_edges(G, pos, ax=ax, width=edge_widths, edge_color="lightblue")
    nx.draw_networkx_edge_labels(G, pos, edge_labels=weights, font_family=prop.get_name())
    
    return fig

def parse_evaluation_points(input_text):
    """評価ポイントを解析する"""
    split_pattern = r'[　 ,、。\n]+'
    points = re.split(split_pattern, input_text)
    return [point.strip() for point in points if point.strip()]

def generate_initial_evaluation(user_input, keywords, evaluation_points):
    """初期評価を生成する"""
    prompt = f"""以下の文章の特徴語を分析して、文章の評価をして下さい:
{user_input}

特徴語: {', '.join([word for word, score in keywords])}

評価する点:
{chr(10).join('- ' + point for point in evaluation_points)}"""
    
    response = model.generate_content(prompt)
    return response.text

def generate_additional_evaluation(previous_evaluation, additional_points, corrections):
    """追加評価を生成する"""
    prompt = f"""前回の評価:
{previous_evaluation}

追加で評価する点:
{additional_points}

修正が必要な点:
{corrections}

上記の情報を踏まえて、より詳細で正確な評価を行ってください。
特に修正が必要な点については、どのように評価を修正すべきか具体的に説明してください。"""

    response = model.generate_content(prompt)
    return response.text

# Streamlit UIの構築
st.title('文章評価アプリ')

# ユーザー入力
user_input = st.text_area('文章を入力')
evaluation_points_input = st.text_area('評価する点を入力してください（複数の場合は改行、カンマ、または空白で区切ってください）')
evaluation_points = parse_evaluation_points(evaluation_points_input)

# 初期分析の実行
if st.button('分析を実行'):
    with st.spinner('分析中...'):
        # キーワード抽出
        if 'keywords_data' not in st.session_state:
            keywords = extract_keywords(user_input)
            st.session_state.keywords_data = pd.DataFrame(keywords, columns=['単語', 'スコア'])
            st.session_state.keywords_data = st.session_state.keywords_data.sort_values(by='スコア', ascending=False).reset_index(drop=True)
            st.session_state.keywords_data.insert(0, '順位', range(1, len(st.session_state.keywords_data) + 1))
        else:
            keywords = [(row['単語'], row['スコア']) for _, row in st.session_state.keywords_data.iterrows()]

        # 共起ネットワーク作成
        if 'cooccurrence_fig' not in st.session_state:
            cooccurrences = find_cooccurrences(user_input, keyworSds)
            tfidf_scores = {word: score for word, score in keywords}
            st.session_state.cooccurrence_fig = draw_cooccurrence_network(cooccurrences, tfidf_scores)

        # 初期評価の生成
        evaluation = generate_initial_evaluation(user_input, keywords, evaluation_points)
        st.session_state.last_evaluation = evaluation

    # 結果の表示
    st.success('分析が完了しました')
    st.table(st.session_state.keywords_data)
    st.pyplot(st.session_state.cooccurrence_fig)
    
    st.write("評価ポイント:")
    for point in evaluation_points:
        st.write(f"- {point}")
    
    st.write("AI評価:")
    st.write(evaluation)

# 追加評価と修正のセクション
if 'last_evaluation' in st.session_state:
    st.markdown("---")
    st.subheader("追加評価と修正")
    
    additional_points = st.text_area("より詳しく評価したい点があれば入力してください")
    corrections = st.text_area("AIの評価で間違っている点や修正が必要な点があれば入力してください")
    
    if st.button('追加評価・修正を実行'):
        with st.spinner('追加分析中...'):
            updated_evaluation = generate_additional_evaluation(
                st.session_state.last_evaluation,
                additional_points,
                corrections
            )
        
        st.success('追加分析が完了しました')
        st.write("更新されたAI評価:")
        st.write(updated_evaluation)
        
        # 新しい評価結果を保存
        st.session_state.last_evaluation = updated_evaluation
