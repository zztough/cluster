import streamlit as st
import spacy
from spacy_streamlit import visualize_ner
import jieba
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN, Birch
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score, davies_bouldin_score
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib # For font management
import matplotlib.font_manager # For font management
import os # For checking font file existence
from scipy.cluster.hierarchy import dendrogram, linkage

# A simple text classifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline

# New imports for advanced visualizations
from wordcloud import WordCloud
import plotly.express as px
import plotly.graph_objects as go

# --- Page Configuration ---
st.set_page_config(page_title="中文NLP智能分析平台", page_icon="🤖", layout="wide")

# --- Custom CSS for Dark Theme (OpenAI-like) ---
custom_css = """
<style>
    /* Base styles */
    body {
        font-family: 'Noto Sans CJK SC', 'Helvetica Neue', Arial, sans-serif; /* Ensure CJK font is prioritized */
        color: #D1D5DB; /* Light gray text */
        background-color: #0F172A; /* Dark blue-gray background */
    }
    .stApp {
        background-color: #0F172A; /* Dark blue-gray background */
    }

    /* Titles and Headers - Unified Sizes */
    h1, h2, h3, h4, h5 {
        color: #F3F4F6; /* Lighter text for headers */
    }
    h1 { /* For the main page title in render_homepage */
        text-align: center;
        padding-bottom: 20px;
        font-size: 2.6em; /* Unified */
        font-weight: 600;
        border-bottom: 2px solid #374151;
    }
    h2 { /* Main section headers (st.header for "1. 输入文本", "2. 查看分析结果") */
        margin-top: 2rem;
        margin-bottom: 1rem;
        padding-bottom: 0.5rem;
        font-size: 2.0em; /* Unified & Increased */
        font-weight: 500;
        border-bottom: 1px solid #374151;
    }
    h3 { /* Sub-section headers (st.subheader for "✂️ 中文分词结果", "🔦 高亮实体:") */
        font-size: 1.7em; /* Unified & Increased */
        font-weight: 500;
        color: #E5E7EB;
        margin-top: 1.5rem;
        margin-bottom: 0.8rem;
    }
    h4 { /* Sub-sub-sections (e.g., "📋 识别到的实体列表", "📏 聚类评估指标") */
        font-size: 1.4em; /* Unified & Increased */
        font-weight: 500;
        color: #E0E0E0;
        margin-top: 1.5rem;
        margin-bottom: 0.7rem;
        padding-bottom: 0.4rem;
        border-bottom: 1px dashed #4B5563;
    }
    h5 { /* Smaller titles (e.g., word frequency, KMeans/DBSCAN 参数) */
        font-size: 1.2em; /* Unified & Increased */
        font-weight: 500;
        color: #E5E7EB;
        margin-top: 1rem;
        margin-bottom: 0.5rem;
    }

    /* Sidebar styling */
    div[data-testid="stSidebar"] > div:first-child {
        background-color: #1E293B;
        border-right: 1px solid #374151;
    }
    div[data-testid="stSidebar"] .stRadio > label { /* Sidebar Navigation Radio Label */
        font-size: 1.2em !important; /* Increased */
        font-weight: 500;
        color: #E5E7EB;
        padding-bottom: 5px; /* Add some space below the main label */
    }
     div[data-testid="stSidebar"] .stRadio div[role="radiogroup"] > label { /* Sidebar Radio button options */
        padding: 7px 0px !important; /* Increased padding */
        font-size: 1.1em !important; /* Increased */
    }
    div[data-testid="stSidebar"] p { /* Sidebar text like © 2024 */
        color: #9CA3AF;
    }
     div[data-testid="stSidebar"] h1, /* Sidebar Header */
     div[data-testid="stSidebar"] h2,
     div[data-testid="stSidebar"] h3,
     div[data-testid="stSidebar"] h4 {
        color: #F3F4F6;
     }

    /* Input widgets styling - Increase label font size for main content area */
    .main div[data-testid="stTextInput"] label,
    .main div[data-testid="stTextArea"] label,
    .main div[data-testid="stSelectbox"] label,
    .main div[data-testid="stNumberInput"] label,
    .main div[data-testid="stRadio"] > label, /* For main content radio buttons */
    .main div[data-testid="stFileUploader"] label,
    .main div[data-testid="stDateInput"] label,
    .main div[data-testid="stTimeInput"] label,
    .main div[data-testid="stColorPicker"] label,
    .main div[data-testid="stSlider"] label {
        font-size: 1.15em !important; /* Increased font size for widget labels in main area */
        color: #E5E7EB !important;
        margin-bottom: 0.4rem !important; 
    }

    div[data-testid="stTextInput"] input,
    div[data-testid="stTextArea"] textarea,
    div[data-testid="stSelectbox"] div[data-baseweb="select"] > div, /* Selectbox */
    div[data-testid="stNumberInput"] input {
        background-color: #374151 !important; /* Darker input background */
        color: #F3F4F6 !important; /* Light text in inputs */
        border: 1px solid #4B5563 !important;
        border-radius: 0.375rem !important; /* Tailwind-like rounded-md */
    }
    div[data-testid="stTextArea"] textarea {
        min-height: 150px; /* Ensure text area is reasonably sized */
    }

    /* Button styling */
    div[data-testid="stButton"] > button {
        background-color: #2563EB; /* OpenAI-like blue */
        color: white;
        border: none;
        padding: 0.6rem 1.2rem;
        border-radius: 0.375rem;
        font-weight: 500;
        transition: background-color 0.2s ease;
    }
    div[data-testid="stButton"] > button:hover {
        background-color: #1D4ED8; /* Darker blue on hover */
    }
    div[data-testid="stButton"] > button:focus {
        outline: none !important;
        box-shadow: 0 0 0 3px rgba(59, 130, 246, 0.5) !important; /* Blue focus ring */
    }
    
    /* Radio buttons in main content */
    div[data-testid="stRadio"] label {
        font-size: 1em;
    }

    /* Slider styling */
    div[data-testid="stSlider"] {
        /* color: #60A5FA; */ /* Blue for slider track/thumb - Use Streamlit's default theming for this if possible */
    }


    /* Expander styling */
    .st-expander {
        border: 1px solid #374151;
        border-radius: 0.5rem; /* Tailwind rounded-lg */
        /* background-color: #1E293B; */ /* Background for expander content area - Let Streamlit handle content bg */
    }
    .st-expander header {
        background-color: #374151; /* Header of expander */
        color: #F3F4F6;
        padding: 0.75rem 1rem;
        border-top-left-radius: 0.5rem;
        border-top-right-radius: 0.5rem;
        border-bottom: 1px solid #1E293B; /* Separator for header */
        font-size: 1.25em !important; /* Increased font size for expander headers */
        font-weight: 500 !important;
    }
    .st-expander header:hover {
        background-color: #4B5563;
    }
    .st-expander div[data-testid="stExpanderDetails"] {
         background-color: #1E293B; /* Content area of expander */
         padding: 1rem;
         border-bottom-left-radius: 0.5rem;
         border-bottom-right-radius: 0.5rem;
    }


    /* Dataframes, Tables, Info/Warning/Error boxes */
    div[data-testid="stDataFrame"],
    div[data-testid="stTable"] { /* st.table is not used, but good to have */
        border-radius: 0.375rem;
        overflow: hidden; /* Ensures border-radius clips content */
    }
    div[data-testid="stInfo"],
    div[data-testid="stWarning"],
    div[data-testid="stError"],
    div[data-testid="stSuccess"] {
        border-radius: 0.375rem;
        padding: 1rem;
        color: #F3F4F6; /* Ensure text is light */
    }
    div[data-testid="stInfo"] { background-color: rgba(59, 130, 246, 0.2); border-left: 5px solid #3B82F6; } /* Blueish Info */
    div[data-testid="stSuccess"] { background-color: rgba(16, 185, 129, 0.2); border-left: 5px solid #10B981; } /* Greenish Success */
    div[data-testid="stWarning"] { background-color: rgba(245, 158, 11, 0.2); border-left: 5px solid #F59E0B; } /* Amber/Orange Warning */
    div[data-testid="stError"] { background-color: rgba(239, 68, 68, 0.2); border-left: 5px solid #EF4444; } /* Reddish Error */


    /* Metric styling */
    div[data-testid="stMetric"] {
        background-color: #1E293B;
        border: 1px solid #374151;
        padding: 1rem 1.5rem;
        border-radius: 0.5rem;
    }
    div[data-testid="stMetric"] > label { /* Metric label */
        color: #9CA3AF; /* Muted label color */
        font-size: 0.9em;
    }
    div[data-testid="stMetric"] > div[data-testid="stMetricValue"] { /* Metric value */
        color: #F3F4F6;
        font-size: 2em; 
        font-weight: 600;
    }
    
    /* Plotly chart container - make sure it blends */
    div[data-testid="stPlotlyChart"] {
        border-radius: 0.375rem;
        overflow: hidden; /* Clip contents like corners */
    }

    /* Matplotlib chart container */
    div[data-testid="stImage"] > img { /* For st.pyplot */
         border-radius: 0.375rem;
         background-color: transparent; /* Ensure image background doesn't override */
    }
    
    /* Horizontal Divider */
    hr {
        border-top: 1px solid #374151;
    }

    /* Links */
    a {
        color: #60A5FA; /* Light blue for links */
        text-decoration: none;
    }
    a:hover {
        text-decoration: underline;
        color: #93C5FD; /* Lighter blue on hover */
    }
    
    /* Feature card styling for homepage */
    .feature-card {
        background-color: #1E293B; 
        padding: 1.5rem;
        border-radius: 0.5rem; 
        border: 1px solid #374151;
        margin-bottom: 1.2rem; /* Increased margin */
        transition: transform 0.2s ease-in-out, box-shadow 0.2s ease-in-out;
        height: calc(100% - 1.2rem); /* Fill height minus margin for better alignment */
        display: flex; /* For vertical alignment of content */
        flex-direction: column; /* Stack content vertically */
    }
    .feature-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 8px 16px rgba(0,0,0,0.25); /* Enhanced shadow */
    }
    .feature-card h3 { 
        font-size: 1.4em; /* Slightly larger */
        color: #F3F4F6;
        margin-top: 0;
        margin-bottom: 0.75rem; /* Increased margin */
        border-bottom: none; 
        line-height: 1.3;
    }
    .feature-card p { 
        font-size: 0.95em;
        color: #D1D5DB;
        line-height: 1.6;
        flex-grow: 1; /* Allow p to take available space */
    }
    /* End of Feature card styling */

    /* Ensure Streamlit's default spinner is visible on dark background */
    .stSpinner > div {
        border-top-color: #2563EB !important; /* Primary button color for spinner */
        border-right-color: transparent !important;
        border-bottom-color: transparent !important;
        border-left-color: transparent !important;
    }

</style>
"""
st.markdown(custom_css, unsafe_allow_html=True)

# --- Model Loading ---
@st.cache_resource # Use st.cache_resource for models
def load_spacy_model(model_name="zh_core_web_sm"):
    """加载spaCy模型，如果未找到或发生其他错误则提示并停止。"""
    try:
        nlp = spacy.load(model_name)
        return nlp
    except OSError:
        st.error(f"SpaCy模型 '{model_name}' 未找到。请在终端运行: python -m spacy download {model_name}")
        st.stop()
    except Exception as e:
        st.error(f"加载SpaCy模型 '{model_name}' 时发生意外错误: {e}")
        st.error("这可能是由于依赖库版本不兼容（例如 NumPy 与 spaCy/thinc 的兼容性问题）或其他配置问题。请检查控制台输出以获取详细信息，并确保您的环境配置正确。")
        st.info("尝试解决方案：1. 检查NumPy版本 (建议 < 2.0)。 2. 重新安装相关库。")
        st.stop()
    # st.stop() should prevent execution from reaching here if an exception occurred.
    # Adding a fallback return or raise for extreme defensiveness, though st.stop() should suffice.
    # This line should ideally not be reached if st.stop() works as intended.
    st.error("SpaCy模型加载失败后，脚本意外继续执行。请检查错误日志。")
    return None # Should not be reached if st.stop() is effective

# nlp_spacy = load_spacy_model() # This global instance might need to be re-evaluated or removed if model is chosen dynamically per task
# For NER, we will call load_spacy_model with the selected model name.

# --- Sample Data and Classifier Training (Minimalistic) ---
# 注意：这是一个非常基础的分类器，仅用于演示。实际应用中需要更大、更均衡的数据集和更复杂的模型。
classification_texts = [
    "国足在世界杯预选赛中取得关键胜利，体育迷欢欣鼓舞。",
    "最新的全球经济报告指出，股市面临回调风险，财经领域需谨慎。",
    "某明星新电影票房大卖，引发娱乐界热议。",
    "人工智能技术取得新突破，科技公司纷纷布局。",
    "奥运会中国代表团再添金牌，体育健儿表现出色。",
    "央行宣布降息，旨在刺激经济增长，财经市场反应积极。",
    "年度音乐盛典落下帷幕，多位歌手获奖，娱乐氛围浓厚。",
    "新型芯片发布，计算能力大幅提升，科技创新永无止境。"
]
classification_labels = ["体育", "财经", "娱乐", "科技", "体育", "财经", "娱乐", "科技"]

@st.cache_resource # Cache the trained classifier
def train_text_classifier(texts, labels, classifier_choice="MultinomialNB"):
    """训练一个简单的文本分类器，可选择分类算法。"""
    
    selected_classifier = None
    if classifier_choice == "MultinomialNB":
        selected_classifier = MultinomialNB()
    elif classifier_choice == "LogisticRegression":
        selected_classifier = LogisticRegression(random_state=42, solver='liblinear')
    elif classifier_choice == "LinearSVC":
        selected_classifier = LinearSVC(random_state=42, dual='auto')
    else:
        st.error(f"未知的分类器选项: {classifier_choice}，将使用默认的MultinomialNB。")
        selected_classifier = MultinomialNB()

    # 使用jieba进行中文分词的TF-IDF向量化器
    model = Pipeline([
        ('tfidf', TfidfVectorizer(tokenizer=lambda x: list(jieba.cut(x)))), 
        ('clf', selected_classifier)
    ])
    model.fit(texts, labels)
    return model

# classifier = train_text_classifier(classification_texts, classification_labels) # Deferred to be dynamic based on selection
class_names = sorted(list(set(classification_labels))) # 获取唯一的类别名并排序

# --- Matplotlib Font Fix for Chinese ---
# For best results, place a Chinese font file (e.g., SimHei.ttf or NotoSansCJKsc-Regular.otf)
# in the 'nlpp/' directory alongside app.py.
CHINESE_FONT_FILENAME = "NotoSansCJKsc-Regular.otf" # Or your chosen font file

def setup_matplotlib_font():
    try:
        font_path = os.path.join(os.path.dirname(__file__), CHINESE_FONT_FILENAME)
        
        font_successfully_set = False
        if os.path.exists(font_path):
            # Use the font file directly if it exists
            matplotlib.font_manager.fontManager.addfont(font_path)
            font_prop = matplotlib.font_manager.FontProperties(fname=font_path)
            matplotlib.rcParams['font.family'] = font_prop.get_name()
            # Prepend to sans-serif list as well
            matplotlib.rcParams['font.sans-serif'] = [font_prop.get_name()] + matplotlib.rcParams.get('font.sans-serif', [])
            # st.sidebar.info(f"已成功加载字体: {font_prop.get_name()} (来自文件: {CHINESE_FONT_FILENAME})")
            font_successfully_set = True
        else:
            # If local font file not found, try a list of common Chinese font family names
            # These names must be known to the system's Matplotlib font manager
            common_chinese_fonts = [
                'WenQuanYi Micro Hei', 'Noto Sans CJK SC', 'SimHei', 
                'Source Han Sans SC', 'AR PL UKai CN', 'AR PL UMing CN', 
                'WenQuanYi Zen Hei', 'DejaVu Sans Fallback' 
            ]
            
            original_sans_serif = list(matplotlib.rcParams.get('font.sans-serif', [])) # Make a mutable copy
            
            # Check if any of the common Chinese fonts are available and set it as the primary family
            font_manager = matplotlib.font_manager.fontManager
            available_font_names = [f.name for f in font_manager.ttflist]
            
            for font_name in common_chinese_fonts:
                if font_name in available_font_names:
                    matplotlib.rcParams['font.family'] = font_name
                    # Ensure this font is also at the start of the sans-serif list
                    if font_name in original_sans_serif:
                        original_sans_serif.remove(font_name)
                    matplotlib.rcParams['font.sans-serif'] = [font_name] + original_sans_serif
                    # st.sidebar.info(f"已成功加载系统字体: {font_name}")
                    font_successfully_set = True
                    break
            
            if not font_successfully_set:
                st.sidebar.warning(
                    f"未在 '{font_path}' 找到字体文件 '{CHINESE_FONT_FILENAME}'，且未能从常见系统字体列表中加载中文字体。"
                    f"图表中的中文可能无法正确显示。请放置一个TTF中文字体 (如 SimHei.ttf) 到 'nlpp/' 目录下并重启应用。"
                )

        matplotlib.rcParams['axes.unicode_minus'] = False  # Ensure minus sign displays correctly

        # Dark theme adjustments for Matplotlib
        matplotlib.rcParams['text.color'] = '#E5E7EB'       # Light gray for text
        matplotlib.rcParams['axes.labelcolor'] = '#D1D5DB'  # Slightly darker for labels
        matplotlib.rcParams['xtick.color'] = '#9CA3AF'      # Muted for ticks
        matplotlib.rcParams['ytick.color'] = '#9CA3AF'
        matplotlib.rcParams['axes.edgecolor'] = '#4B5563'   # Border color for axes
        matplotlib.rcParams['figure.facecolor'] = 'none'    # Transparent figure background
        matplotlib.rcParams['axes.facecolor'] = 'none'      # Transparent axes background
        matplotlib.rcParams['savefig.transparent'] = True   # Ensure saved figures are transparent

    except Exception as e:
        st.sidebar.error(f"配置Matplotlib中文字体时出错: {e}")

setup_matplotlib_font() # Call the function to set the font

# --- Plotting Functions (Refactored and Enhanced) ---

def plot_scatter_clusters(ax, X_reduced, labels, cluster_model_name,
                          colormap='viridis', marker_size=50, marker_alpha=0.7,
                          marker_styles_enabled=False,
                          bg_color='rgba(0,0,0,0)', show_grid=True, grid_color='#4B5563'):
    """绘制聚类结果的散点图，具有增强的可视化选项。"""
    ax.clear() # Clear previous plot on the axis
    ax.set_facecolor(bg_color)

    unique_labels = sorted(list(set(labels)))
    is_dbscan_with_outliers = cluster_model_name == "DBSCAN" and -1 in unique_labels
    
    current_cmap = plt.get_cmap(colormap)
    
    plot_labels_for_colors = [l for l in unique_labels if l != -1] if is_dbscan_with_outliers else unique_labels
    if not plot_labels_for_colors: # Handle cases with only noise or no clusters
        plot_labels_for_colors = unique_labels # Avoid error if only noise points
        
    num_plot_clusters = max(1, len(plot_labels_for_colors)) # Ensure at least 1 for color mapping
    colors_for_plot = current_cmap(np.linspace(0, 1, num_plot_clusters))
    color_map_dict = {label: colors_for_plot[i] for i, label in enumerate(plot_labels_for_colors)}

    marker_options = ['o', 's', '^', 'D', 'v', '<', '>', 'p', '*', 'H', 'X']
    legend_elements = []

    for i, label_val in enumerate(unique_labels):
        idx = (labels == label_val)
        current_marker = marker_options[i % len(marker_options)] if marker_styles_enabled else 'o'
        
        if label_val == -1 and is_dbscan_with_outliers:
            ax.scatter(X_reduced[idx, 0], X_reduced[idx, 1], color='gray', label='离群点', 
                       alpha=marker_alpha, marker='x', s=marker_size)
            legend_elements.append(matplotlib.lines.Line2D([0], [0], marker='x', color='w', label='离群点', markerfacecolor='gray', markersize=10, linestyle='None'))
        elif X_reduced[idx].shape[0] > 0 : # Ensure there are points in the cluster
            cluster_color = color_map_dict.get(label_val, current_cmap(0.0)) # Default color if label somehow not in dict
            ax.scatter(X_reduced[idx, 0], X_reduced[idx, 1], color=cluster_color, 
                       label=f'聚类 {label_val}', alpha=marker_alpha, 
                       marker=current_marker, edgecolors='w', s=marker_size)
            legend_elements.append(matplotlib.lines.Line2D([0], [0], marker=current_marker, color='w', label=f'聚类 {label_val}', markerfacecolor=cluster_color, markersize=10))

    if legend_elements:
        ax.legend(handles=legend_elements, title="图例", bbox_to_anchor=(1.05, 1), loc='upper left')
    
    ax.set_xlabel("Component 1", fontsize=10)
    ax.set_ylabel("Component 2", fontsize=10)
    
    if show_grid:
        ax.grid(True, color=grid_color, linestyle='--', alpha=0.5)
    else:
        ax.grid(False)
    plt.tight_layout(rect=[0, 0, 0.85, 1]) # Adjust layout to make space for legend

def plot_dynamic_dendrogram(ax, tfidf_matrix_dense, linkage_method, metric, 
                            orientation='top', color_threshold=None):
    """绘制树状图（恢复到较简单版本）。"""
    ax.clear()
    ax.set_facecolor('none') # Set transparent background for axes

    if tfidf_matrix_dense is None or tfidf_matrix_dense.shape[0] < 2:
        ax.text(0.5, 0.5, "数据不足以生成树状图", ha="center", va="center", transform=ax.transAxes)
        return

    try:
        linkage_matrix_val = linkage(tfidf_matrix_dense, method=linkage_method, metric=metric)
        
        # Simplified dendrogram plotting, closer to original implicit behavior
        # Default truncation and leaf display will be handled by scipy if not specified, or use simple defaults.
        p_dendro = min(30, tfidf_matrix_dense.shape[0])
        doc_labels = None # No detailed labels by default in this simplified version
        if tfidf_matrix_dense.shape[0] <= p_dendro + 10: # Basic heuristic for showing some labels if few docs
             doc_labels = [f"文{i}" for i in range(tfidf_matrix_dense.shape[0])]
        
        dendrogram(
            linkage_matrix_val, 
            ax=ax, 
            orientation=orientation, 
            truncate_mode='lastp', # A common default for readability 
            p=p_dendro,              # A common default for readability
            leaf_rotation=90. if orientation in ['top', 'bottom'] else 0., 
            leaf_font_size=8., 
            show_contracted=True,
            labels=doc_labels, # Simplified labels
            color_threshold=color_threshold # Retain color threshold if provided
        )
        
        ax.set_title(f"层次聚类树状图 ({linkage_method} linkage)", fontsize=14)
        if orientation in ['top', 'bottom']:
            ax.set_xlabel("文档索引或聚类", fontsize=10)
            ax.set_ylabel("距离/差异度", fontsize=10)
        else:
            ax.set_ylabel("文档索引或聚类", fontsize=10)
            ax.set_xlabel("距离/差异度", fontsize=10)
        
        if color_threshold and color_threshold > 0:
             if orientation in ['top', 'bottom']:
                ax.axhline(y=color_threshold, c='grey', lw=1, linestyle='dashed')
             else: # left, right
                ax.axvline(x=color_threshold, c='grey', lw=1, linestyle='dashed')
        plt.tight_layout()
    except Exception as e_dendro:
        st.error(f"生成树状图时出错: {e_dendro}")
        ax.text(0.5, 0.5, f"生成树状图错误: {e_dendro}", ha="center", va="center", transform=ax.transAxes, color='red')

# --- Homepage Rendering Function ---
def render_homepage():
    st.markdown("""
    <div style="text-align: center; padding-top: 1rem; padding-bottom: 1rem;">
        <span style="font-size: 4.5em; line-height: 1;">🚀</span>
        <h1 style="font-size: 3.2em; font-weight: 700; color: #F9FAFB; margin-top: 0.5rem; margin-bottom: 0.75rem; letter-spacing: -0.5px;">
            中文NLP智能分析平台
        </h1>
        <p style="font-size: 1.3em; color: #D1D5DB; max-width: 750px; margin: 0 auto 1.5rem auto; line-height: 1.7;">
            一站式满足您的中文自然语言处理需求。探索文本的深层价值，从智能分词到高级聚类分析，体验前沿AI技术。
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<hr style='border-top: 1px solid #374151; margin-top: 1rem; margin-bottom: 2rem;'>", unsafe_allow_html=True)
    
    st.markdown("<h2 style='text-align: center; font-size: 2.2em; font-weight: 600; color: #F3F4F6; margin-top: 2rem; margin-bottom: 2.5rem;'>核心功能一览 ✨</h2>", unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
        <div class="feature-card">
            <h3>📝 智能分词 & 词频</h3>
            <p>精准切分中文文本，统计高频词汇，并生成直观的词云图和词频统计图。支持Jieba和SpaCy引擎，深入洞察文本构成。</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="feature-card">
            <h3>📊 中文文本分类</h3>
            <p>将文本自动归类到预定义主题（如体育、财经、娱乐、科技）。支持多种经典分类算法，提供各类别概率分布，辅助决策。</p>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class="feature-card">
            <h3>👁️‍🗨️ 命名实体识别 (NER)</h3>
            <p>从文本中自动识别并分类关键实体（人名、地名、机构名等）。可灵活选择不同规模的SpaCy预训练模型，满足不同精度需求。</p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class="feature-card">
            <h3>🧩 文本聚类分析</h3>
            <p>无监督地将相似文本自动分组，揭示数据集中的潜在结构与话题。提供多种聚类算法和丰富的可视化工具，助力探索性数据分析。</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("<hr style='border-top: 1px solid #374151; margin-top: 3rem; margin-bottom: 1.5rem;'><p style='text-align: center; font-size: 1.1em; color: #9CA3AF;'>请从左侧导航栏选择一项功能开始您的分析之旅！</p>", unsafe_allow_html=True)


# nlp_spacy = load_spacy_model() # This global instance might need to be re-evaluated or removed if model is chosen dynamically per task
# For NER, we will call load_spacy_model with the selected model name.

# --- UI Sections ---
# The following st.title and st.markdown are removed to avoid redundancy with render_homepage() on the homepage.
# For other pages, specific headers like "1. 输入文本" are used.
# st.title("中文NLP智能分析平台 🤖") 
# st.markdown("""欢迎使用本平台！请在下方选择您要执行的自然语言处理任务。
# 您可以直接粘贴文本或上传TXT文件进行分析。
# """)

# --- Sidebar for Navigation ---
st.sidebar.header("🧭 导航")
analysis_options = [
    "🏠 主页",
    "📝 中文分词", 
    "👁️‍🗨️ 命名实体识别", 
    "📊 中文文本分类", 
    "🧩 文本聚类分析"
]
selected_analysis = st.sidebar.radio("选择功能:", analysis_options)

# --- Input Area ---
# Moved the condition for showing input area outside homepage
if selected_analysis != "🏠 主页":
    st.header("⌨️ 1. 输入文本")
    input_method = st.radio("选择输入方式:", ("粘贴文本", "上传TXT文件"), horizontal=True, key="input_method_radio")

    raw_texts = []  # 用于存储所有待处理的文本行

    if input_method == "粘贴文本":
        text_area_input = st.text_area("在此处粘贴文本（对于聚类，每行代表一个独立文档）:", "", height=200, key="paste_area")
        if text_area_input:
            raw_texts = [line.strip() for line in text_area_input.split('\n') if line.strip()]
    elif input_method == "上传TXT文件":
        uploaded_files = st.file_uploader("上传一个或多个TXT文件 (UTF-8编码):", type=["txt"], accept_multiple_files=True, key="file_uploader")
        if uploaded_files:
            for uploaded_file in uploaded_files:
                try:
                    string_data = uploaded_file.read().decode("utf-8")
                    raw_texts.extend([line.strip() for line in string_data.split('\n') if line.strip()])
                except Exception as e:
                    st.error(f"读取文件 {uploaded_file.name} 失败: {e}")

    # --- Example Data Button ---
    if not raw_texts:
        if st.button("载入示例数据进行测试", key="load_example_button"):
            if selected_analysis == "🧩 文本聚类分析":
                raw_texts = [
                    "深度学习是机器学习的一个重要分支，它在图像识别领域取得了巨大成功。",
                    "自然语言处理关注计算机如何理解和生成人类语言，应用广泛。",
                    "机器学习算法，如支持向量机和决策树，常用于数据挖掘任务。",
                    "苹果公司最近发布了新款iPhone，配备了更强大的A系列仿生芯片。",
                    "特斯拉是全球领先的电动汽车制造商，其自动驾驶技术备受关注。",
                    "最近的金融市场波动较大，投资者在进行股票交易时应保持谨慎。",
                    "中国国家足球队正在积极备战即将到来的亚洲杯预选赛。",
                    "NBA篮球联赛常规赛激战正酣，各支球队为季后赛名额展开激烈争夺。"
                ]
                st.toast("已加载聚类示例数据。", icon="📄")
            elif selected_analysis == "📊 中文文本分类":
                raw_texts = ["最新的体育新闻报道了昨晚的足球比赛结果。"]
                st.toast("已加载分类示例数据。", icon="📄")
            else: # For word segmentation and NER
                raw_texts = ["我爱北京天安门，天安门上太阳升。伟大领袖毛主席，指引我们向前进。"]
                st.toast("已加载通用示例数据。", icon="📄")
else: # if selected_analysis == "🏠 主页":
    raw_texts = [] # Ensure raw_texts is empty or not used for homepage


# --- Process and Display --- 
current_text_to_process = "" # 用于单文本分析任务
if selected_analysis != "🏠 主页" and raw_texts: # Ensure this logic only runs for analysis pages with text
    current_text_to_process = raw_texts[0]
    if len(raw_texts) > 1 and selected_analysis not in ["🧩 文本聚类分析"]:
        st.info(f"检测到 {len(raw_texts)} 段文本。对于'{selected_analysis}'，默认处理第一段。如需批量处理，请选择'文本聚类分析'或分别操作。", icon="ℹ️")

if selected_analysis != "🏠 主页": # Only add divider if not on homepage
    st.divider()

# --- Main logic for different analyses ---
if selected_analysis == "🏠 主页":
    render_homepage()
    # Footer for homepage can be part of render_homepage or here
    # st.sidebar.markdown("--- "*10) # This is already at the end of the script, keep it there
    # st.sidebar.info("© 2024 中文NLP智能分析平台")

elif raw_texts: # For other analysis options, only proceed if raw_texts exist
    st.header("🔍 2. 查看分析结果")

    # 1. Chinese Word Segmentation
    if selected_analysis == "📝 中文分词":
        if current_text_to_process:
            st.subheader("✂️ 中文分词结果")
            
            # --- Model Selection for Word Segmentation ---
            segmentation_model_options = ["Jieba", "SpaCy"]
            selected_segmentation_model = st.selectbox("选择分词引擎:", segmentation_model_options, key="segmentation_model_selector")
            st.write("---原始文本---") # Changed from st.write("**原始文本:**", current_text_to_process)
            st.text(current_text_to_process) # Using st.text for better block display of original text
            st.write("---分词结果---")

            with st.spinner(f"正在使用 {selected_segmentation_model} 进行分词..."):
                processed_tokens_for_cloud = []
                if selected_segmentation_model == "Jieba":
                    seg_list_jieba = list(jieba.cut(current_text_to_process))
                    st.info(" / ".join(seg_list_jieba))
                    processed_tokens_for_cloud = seg_list_jieba
                    if seg_list_jieba:
                        st.markdown("---")
                        st.markdown("##### 📊 Top 15 词频统计 (Jieba)")
                        word_counts = pd.Series(seg_list_jieba).value_counts().nlargest(15)
                        if not word_counts.empty:
                            word_counts_df = word_counts.reset_index()
                            word_counts_df.columns = ['词语', '频率']
                            fig_bar_jieba = px.bar(word_counts_df, 
                                                 x='频率', 
                                                 y='词语', 
                                                 orientation='h', 
                                                 color='词语', 
                                                 title="词频统计 (Jieba)",
                                                 labels={'频率':'频率', '词语':'词语'},
                                                 template="plotly_dark") # Apply dark theme
                            fig_bar_jieba.update_layout(yaxis={'categoryorder':'total ascending'}, showlegend=False)
                            st.plotly_chart(fig_bar_jieba, use_container_width=True)
                        else:
                            st.text("无有效词语进行统计。")

                elif selected_segmentation_model == "SpaCy":
                    nlp_segmentation_model = load_spacy_model("zh_core_web_sm") 
                    if nlp_segmentation_model:
                        doc_spacy = nlp_segmentation_model(current_text_to_process)
                        # Filter out punctuation and spaces for cleaner word cloud and frequency count
                        seg_list_spacy = [token.text for token in doc_spacy if not token.is_punct and not token.is_space and token.text.strip()]
                        st.info(" / ".join(seg_list_spacy))
                        processed_tokens_for_cloud = seg_list_spacy
                        if seg_list_spacy:
                            st.markdown("---")
                            st.markdown("##### 📊 Top 15 词频统计 (SpaCy)")
                            word_counts_spacy = pd.Series(seg_list_spacy).value_counts().nlargest(15)
                            if not word_counts_spacy.empty:
                                word_counts_spacy_df = word_counts_spacy.reset_index()
                                word_counts_spacy_df.columns = ['词语', '频率']
                                fig_bar_spacy = px.bar(word_counts_spacy_df, 
                                                     x='频率', 
                                                     y='词语', 
                                                     orientation='h', 
                                                     color='词语',
                                                     title="词频统计 (SpaCy)",
                                                     labels={'频率':'频率', '词语':'词语'},
                                                     template="plotly_dark") # Apply dark theme
                                fig_bar_spacy.update_layout(yaxis={'categoryorder':'total ascending'}, showlegend=False)
                                st.plotly_chart(fig_bar_spacy, use_container_width=True)
                            else:
                                st.text("无有效词语进行统计。")
                    else:
                        st.error("SpaCy模型加载失败，无法进行分词。")
                
                # --- Word Cloud Visualization (Common for both Jieba and SpaCy) ---
                if processed_tokens_for_cloud:
                    st.markdown("---")
                    st.markdown("##### ☁️ 词云图")
                    try:
                        # Try to get font path for WordCloud - simplified and more robust
                        font_path_wc = os.path.join(os.path.dirname(__file__), CHINESE_FONT_FILENAME)
                        
                        if not os.path.exists(font_path_wc):
                            st.warning(f"词云无法生成：未在 'nlpp/' 目录下找到中文字体文件 '{CHINESE_FONT_FILENAME}'。请确保该文件存在。")
                            font_path_wc = None # Explicitly set to None if not found

                        if font_path_wc:
                            wordcloud_text = " ".join(processed_tokens_for_cloud)
                            if wordcloud_text.strip():
                                wc = WordCloud(
                                    font_path=font_path_wc, # Directly use the validated path
                                    width=800, 
                                    height=400, 
                                    background_color='white', # Wordcloud background itself
                                    colormap='viridis', # Example colormap, can be changed
                                    collocations=False 
                                ).generate(wordcloud_text)
                                
                                fig_wc, ax_wc = plt.subplots(figsize=(10,5))
                                ax_wc.imshow(wc, interpolation='bilinear')
                                ax_wc.axis("off")
                                # For dark theme Streamlit, ensure Matplotlib plot bg is transparent or matches
                                fig_wc.patch.set_alpha(0) # Make Matplotlib figure background transparent
                                ax_wc.patch.set_alpha(0)  # Make Matplotlib axes background transparent
                                st.pyplot(fig_wc)
                            else:
                                st.text("文本内容过少或无有效词语，无法生成词云。")
                        # No else needed here, warning for font_path_wc=None is handled above
                                
                    except Exception as e_wc:
                        st.error(f"生成词云时出错: {e_wc}")
        else:
            st.info("请输入或上传文本以进行分词。", icon="👈")

    # 2. Named Entity Recognition (NER)
    elif selected_analysis == "👁️‍🗨️ 命名实体识别":
        if current_text_to_process:
            st.subheader("🏷️ 命名实体识别 (NER) 结果")

            # --- Model Selection for NER ---
            ner_model_options = {
                "小模型 (zh_core_web_sm)": "zh_core_web_sm",
                "中等模型 (zh_core_web_md)": "zh_core_web_md",
                "大模型 (zh_core_web_lg)": "zh_core_web_lg",
            }
            selected_ner_model_display_name = st.selectbox(
                "选择SpaCy NER模型 (大模型首次加载较慢，需已下载):", 
                list(ner_model_options.keys()), 
                key="ner_model_selector"
            )
            selected_ner_model_name = ner_model_options[selected_ner_model_display_name]

            # Load the selected spaCy model for NER
            # The load_spacy_model function is cached, so it will be efficient after the first load of a model.
            nlp_ner_model = load_spacy_model(selected_ner_model_name)

            if nlp_ner_model is None:
                # Error handling is done in load_spacy_model, which calls st.stop()
                # So, execution should not reach here if a model fails to load.
                st.error("NER模型加载失败，无法继续。") # Fallback, should not be seen normally
            else:
                with st.spinner(f"正在使用 {selected_ner_model_display_name} 进行NER分析..."):
                    st.write("**原始文本:**", current_text_to_process)
                    doc = nlp_ner_model(current_text_to_process)
                    
                    # Get labels from the current model's NER pipe
                    ner_labels = []
                    if nlp_ner_model.has_pipe("ner"):
                        ner_labels = list(nlp_ner_model.get_pipe("ner").labels)
                    
                    st.subheader("🔦 高亮实体:") # Added emoji and using subheader for emphasis
                    visualize_ner(doc, labels=ner_labels, show_table=False, title="", displacy_options={"colors": {"ORG": "#7DF9FF", "PERSON": "#FFC0CB", "LOC": "#LIGHTGREEN", "GPE":"#FFD700" }})
                    
                    if doc.ents:
                        st.markdown("#### 📋 识别到的实体列表")
                        entities = [(ent.text, ent.label_) for ent in doc.ents]
                        df_ents = pd.DataFrame(entities, columns=["实体文本", "实体类型"])
                        st.dataframe(df_ents, use_container_width=True)

                        if not df_ents.empty:
                            st.markdown("---")
                            st.markdown("##### 📊 实体类型统计")
                            entity_type_counts = df_ents["实体类型"].value_counts().reset_index()
                            entity_type_counts.columns = ['实体类型', '数量'] # Rename for Plotly
                            if not entity_type_counts.empty:
                                fig_ner_counts = px.bar(entity_type_counts, 
                                                        x='实体类型', 
                                                        y='数量', 
                                                        color='实体类型',
                                                        title="命名实体类型分布",
                                                        labels={'实体类型':'实体类型', '数量':'出现次数'},
                                                        template="plotly_dark") # Apply dark theme
                                fig_ner_counts.update_layout(xaxis_title="实体类型", yaxis_title="数量", showlegend=True)
                                st.plotly_chart(fig_ner_counts, use_container_width=True)
                            else:
                                st.text("未识别到可统计的实体类型。")
                    else:
                        st.info("未在本段文本中识别到命名实体。")
        else:
            st.info("请输入或上传文本以进行NER分析。", icon="👈")

    # 3. Chinese Text Classification
    elif selected_analysis == "📊 中文文本分类":
        if current_text_to_process:
            st.subheader("🎯 中文文本分类结果")

            # --- Classifier Model Selection ---
            classifier_options = {
                "朴素贝叶斯 (MultinomialNB)": "MultinomialNB",
                "逻辑回归 (LogisticRegression)": "LogisticRegression",
                "支持向量机 (LinearSVC)": "LinearSVC"
            }
            selected_classifier_display_name = st.selectbox(
                "选择分类算法:",
                list(classifier_options.keys()),
                key="classifier_selector"
            )
            selected_classifier_type = classifier_options[selected_classifier_display_name]

            # Train or load the selected classifier from cache
            classifier_model = train_text_classifier(classification_texts, classification_labels, classifier_choice=selected_classifier_type)

            if classifier_model:
                with st.spinner(f"正在使用 {selected_classifier_display_name} 进行文本分类..."):
                    st.write("**分析文本:**", current_text_to_process[:200] + ("..." if len(current_text_to_process) > 200 else ""))
                    prediction = classifier_model.predict([current_text_to_process])[0]
                    
                    st.success(f"**预测类别:** {prediction}")

                    # Check if the classifier has predict_proba method
                    if hasattr(classifier_model, "predict_proba") and callable(classifier_model.predict_proba):
                        probabilities = classifier_model.predict_proba([current_text_to_process])[0]
                        prob_df = pd.DataFrame({'类别': class_names, '概率': probabilities})
                        prob_df = prob_df.sort_values(by='概率', ascending=False).reset_index(drop=True)
                        st.markdown("#### 📈 各类别概率")
                        st.dataframe(prob_df, use_container_width=True)

                        st.markdown("---")
                        # Gauge chart for the top prediction
                        if not prob_df.empty:
                            top_prediction_label = prob_df.iloc[0]['类别']
                            top_prediction_prob = prob_df.iloc[0]['概率']
                            
                            st.markdown(f"##### 🎯 主要预测类别 ({top_prediction_label}) 置信度")
                            fig_gauge = go.Figure(go.Indicator(
                                mode = "gauge+number",
                                value = top_prediction_prob * 100, # Convert to percentage
                                title = {'text': f"预测为: {top_prediction_label}", 'font': {'color': "white"}},
                                gauge = {'axis': {'range': [None, 100]},
                                         'bar': {'color': "#636EFA"}, # Using a color from plotly_dark
                                         'steps' : [
                                             {'range': [0, 50], 'color': "rgba(255,255,255,0.1)"},
                                             {'range': [50, 80], 'color': "rgba(255,255,255,0.2)"}],
                                         'threshold' : {'line': {'color': "red", 'width': 4}, 'thickness': 0.75, 'value': 90}},
                                number = {'font': {'color': "white"}}
                                ))
                            fig_gauge.update_layout(height=300, margin=dict(t=50, b=50, l=50, r=50), template="plotly_dark") # Apply dark theme
                            st.plotly_chart(fig_gauge, use_container_width=True)

                        # Visualizing probabilities with Plotly bar chart
                        st.markdown("---")
                        st.markdown("##### 📊 各类别概率分布图")
                        if not prob_df.empty:
                            fig_class_probs = px.bar(prob_df, 
                                                     x='概率', 
                                                     y='类别', 
                                                     orientation='h',
                                                     color='类别',
                                                     title="文本分类各类别概率",
                                                     labels={'概率':'预测概率', '类别':'文本类别'},
                                                     template="plotly_dark") # Apply dark theme
                            fig_class_probs.update_layout(yaxis={'categoryorder':'total ascending'}, 
                                                      showlegend=False,
                                                      xaxis_ticksuffix="%") # Add % suffix to probability axis
                            # Convert probability to percentage for display on hover and ticks if desired
                            # fig_class_probs.update_traces(x=[val * 100 for val in prob_df['概率']])
                            st.plotly_chart(fig_class_probs, use_container_width=True)
                        else:
                            st.text("无概率信息可供可视化。")
                    elif selected_classifier_type == "LinearSVC":
                        st.info(f"注意: {selected_classifier_display_name} 模型不直接提供概率输出。决策函数值可用于评估置信度，但此处未显示。")
                    else:
                        st.warning(f"当前选择的分类器 {selected_classifier_display_name} 不支持概率输出。")
            else:
                st.error("分类器模型未能成功加载或训练。")
                
        else:
            st.info("请输入或上传文本以进行文本分类。只有在输入单个文本段落时，此功能才可用。", icon="👈")

    # 4. Text Clustering Analysis
    elif selected_analysis == "🧩 文本聚类分析":
        if len(raw_texts) < 2:
            st.warning("文本聚类至少需要两条文本。请粘贴多行文本或上传包含多行/多个文件。", icon="⚠️")
        else:
            st.subheader("⚙️ 文本聚类分析参数设置")
            cluster_model_name = st.selectbox(
                "选择聚类模型:",
                ("KMeans", "AgglomerativeClustering", "DBSCAN", "BIRCH"),
                key="cluster_model_select"
            )
            # Initialize clusters to None before the try block
            clusters = None 
            model_instance = None
            num_clusters_found = 0
            tfidf_matrix_dense = None # Ensure it's defined for later use, even if try block fails early
            linkage_agg = 'ward' # Default for Agglomerative, if needed before set
            metric_agg = 'euclidean' # Default for Agglomerative

            try:
                tfidf_vectorizer = TfidfVectorizer(
                    tokenizer=lambda x: list(jieba.cut(x)), max_df=0.90, min_df=2, stop_words=None
                )
                tfidf_matrix = tfidf_vectorizer.fit_transform(raw_texts)
                tfidf_matrix_dense = tfidf_matrix.toarray() 

                if tfidf_matrix.shape[0] < 2 or tfidf_matrix.shape[1] < 1:
                    st.error("经过TF-IDF处理后，文本数据不足或特征过少。")
                    st.stop()
                
                max_possible_clusters = min(tfidf_matrix.shape[0] - 1 if tfidf_matrix.shape[0] > 1 else 1, 20)
                if max_possible_clusters < 1 and cluster_model_name not in ["DBSCAN"]:
                     st.warning("文本数量过少，无法有效聚类。")
                     st.stop()
                
                # --- Model Specific Parameters and Execution (copied from your last correct version) ---
                if cluster_model_name == "KMeans":
                    st.markdown("##### ✨ KMeans 参数")
                    num_clusters_kmeans = st.slider("K值:", min_value=2, max_value=max_possible_clusters if max_possible_clusters >=2 else 2, value=min(4, max_possible_clusters) if max_possible_clusters >=2 else 2, key="k_kmeans")
                    if tfidf_matrix.shape[0] < num_clusters_kmeans: st.error("文本数少于K值!"); st.stop()
                    with st.spinner(f"KMeans (K={num_clusters_kmeans}) 聚类中..."):
                        model_instance = KMeans(n_clusters=num_clusters_kmeans, random_state=42, n_init='auto')
                        clusters = model_instance.fit_predict(tfidf_matrix)
                        num_clusters_found = num_clusters_kmeans
                elif cluster_model_name == "AgglomerativeClustering":
                    st.markdown("##### 🔗 Agglomerative Clustering 参数")
                    num_clusters_agg = st.slider("簇数量:", min_value=2, max_value=max_possible_clusters if max_possible_clusters >=2 else 2, value=min(4, max_possible_clusters) if max_possible_clusters >=2 else 2, key="k_agg")
                    linkage_agg = st.selectbox("Linkage方法:", ('ward', 'complete', 'average', 'single'), key="linkage_agg")
                    metric_agg = 'euclidean' 
                    if tfidf_matrix.shape[0] < num_clusters_agg: st.error("文本数少于簇数量!"); st.stop()
                    with st.spinner(f"Agglomerative (k={num_clusters_agg}, linkage={linkage_agg}) 聚类中..."):
                        model_instance = AgglomerativeClustering(n_clusters=num_clusters_agg, metric=metric_agg, linkage=linkage_agg)
                        clusters = model_instance.fit_predict(tfidf_matrix_dense)
                        num_clusters_found = num_clusters_agg
                elif cluster_model_name == "DBSCAN":
                    st.markdown("##### 🧐 DBSCAN 参数")
                    eps_dbscan = st.number_input("Epsilon (eps - 邻域半径):", min_value=0.01, value=0.5, step=0.01, format="%.2f", key="eps_dbscan")
                    min_samples_dbscan = st.slider("Min Samples (核心对象最小样本数):", min_value=1, max_value=max(5, tfidf_matrix.shape[0] // 10), value=max(1,min(5, tfidf_matrix.shape[0] // 10 if tfidf_matrix.shape[0] // 10 >0 else 1)), key="min_samples_dbscan")
                    with st.spinner(f"DBSCAN (eps={eps_dbscan}, min_samples={min_samples_dbscan}) 聚类中..."):
                        model_instance = DBSCAN(eps=eps_dbscan, min_samples=min_samples_dbscan, metric='cosine')
                        clusters = model_instance.fit_predict(tfidf_matrix)
                        num_clusters_found = len(set(clusters)) - (1 if -1 in clusters else 0)
                        st.info(f"DBSCAN 找到 {num_clusters_found} 个簇和 {list(clusters).count(-1)} 个离群点。")
                elif cluster_model_name == "BIRCH":
                    st.markdown("##### 🌳 BIRCH 参数")
                    threshold_birch = st.number_input("Threshold (CF子簇半径阈值):", min_value=0.01, value=0.5, step=0.05, format="%.2f", key="threshold_birch")
                    branching_factor_birch = st.slider("Branching Factor (CF树分支因子):", min_value=2, max_value=100, value=50, key="branching_birch")
                    n_clusters_birch_options = [None] + list(range(2, (max_possible_clusters if max_possible_clusters >=2 else 2) + 1))
                    default_k_birch = min(3, max_possible_clusters) if max_possible_clusters >=2 else None
                    if default_k_birch == None and len(n_clusters_birch_options) > 1: default_k_birch = 2
                    num_clusters_birch_selected = st.select_slider("目标簇数量 (K - None则自动确定):", options=n_clusters_birch_options, value=default_k_birch, key="k_birch_slider")
                    with st.spinner(f"BIRCH (threshold={threshold_birch}, n_clusters={num_clusters_birch_selected}) 聚类中..."):
                        model_instance = Birch(threshold=threshold_birch, branching_factor=branching_factor_birch, n_clusters=num_clusters_birch_selected)
                        clusters = model_instance.fit_predict(tfidf_matrix)
                        if hasattr(model_instance, 'n_clusters_') and model_instance.n_clusters_ is not None: num_clusters_found = model_instance.n_clusters_
                        elif num_clusters_birch_selected is not None: num_clusters_found = num_clusters_birch_selected
                        else: num_clusters_found = len(set(clusters))
                        st.info(f"BIRCH 找到/设定 {num_clusters_found} 个簇。")

            except ValueError as ve:
                st.error(f"聚类分析时发生数值错误: {ve}")
                clusters = None # Ensure clusters is None if an error occurs
            except Exception as e:
                st.error(f"聚类分析时发生未知错误: {e}")
                clusters = None # Ensure clusters is None if an error occurs

            # --- Displaying Results & Metrics & Visualizations (common for all models) ---
            # This block will only execute if clusters were successfully computed (clusters is not None)
            if clusters is not None:
                st.subheader("📊 文本聚类分析结果")
                df_cluster_results = pd.DataFrame({'原始文本': raw_texts, '聚类标签': clusters})
                st.markdown("#### 📋 文本聚类分配:")
                st.dataframe(df_cluster_results, height=200, use_container_width=True)

                st.markdown("#### 📏 聚类评估指标:")
                if num_clusters_found >= 2 and num_clusters_found < tfidf_matrix.shape[0]:
                    try:
                        sil_score = silhouette_score(tfidf_matrix, clusters, metric='cosine')
                        db_score = davies_bouldin_score(tfidf_matrix_dense, clusters)
                        st.metric(label="轮廓系数 (Silhouette Score)", value=f"{sil_score:.3f}", help="范围[-1, 1]，越接近1越好。使用余弦距离计算。")
                        st.metric(label="戴维斯-布尔丁指数 (Davies-Bouldin)", value=f"{db_score:.3f}", help="值越小越好。")
                    except ValueError as e_metric: st.warning(f"计算评估指标时出错: {e_metric}。");
                    except Exception as e_metric_other: st.error(f"计算评估指标时发生未知错误: {e_metric_other}")
                else:
                    st.info(f"找到 {num_clusters_found} 个簇。评估指标在簇数量为 2 到 (样本数-1) 之间时更有意义。")
                
                st.markdown("### 🖼️ 可视化展板") # Was "--- Visualizations ---"
                with st.expander("降维散点图设置与可视化", expanded=True):
                    st.markdown("#### 📈 降维散点图") # Was "**聚类结果可视化 (降维散点图):**"
                    
                    vis_col_main_1, vis_col_main_2 = st.columns([2,1]) # Main columns for scatter controls

                    with vis_col_main_1: # Left column for primary controls
                        reduction_method = st.selectbox("降维方法:", ["PCA", "t-SNE"], key="reduction_scatter")
                        available_colormaps = plt.colormaps()
                        filtered_colormaps = [cm for cm in available_colormaps if not cm.endswith('_r')]
                        selected_colormap = st.selectbox("选择调色板:", filtered_colormaps, index=filtered_colormaps.index('viridis') if 'viridis' in filtered_colormaps else 0, key="colormap_scatter")
                        scatter_marker_styles_enabled = st.checkbox("为不同簇启用不同标记样式", False, key="scatter_marker_styles_cb")
                        scatter_show_grid = st.checkbox("显示网格", True, key="scatter_show_grid_cb")
                        
                    with vis_col_main_2: # Right column for size, alpha and colors
                        scatter_marker_size = st.slider("标记大小:", min_value=10, max_value=200, value=50, step=10, key="scatter_marker_size_slider")
                        scatter_marker_alpha = st.slider("标记透明度:", min_value=0.1, max_value=1.0, value=0.7, step=0.05, key="scatter_marker_alpha_slider")
                        
                        # Horizontal layout for background and grid color pickers
                        color_col1, color_col2 = st.columns(2)
                        with color_col1:
                            scatter_bg_color = st.color_picker("背景颜色", "#1E293B", key="scatter_bg_color_picker")
                        with color_col2:
                            if scatter_show_grid:
                                scatter_grid_color = st.color_picker("网格颜色", "#4B5563", key="scatter_grid_color_picker")
                            else:
                                scatter_grid_color = "#4B5563" # Default even if not shown, to prevent error
                    
                    perplexity_value = min(30.0, float(max(1, tfidf_matrix_dense.shape[0] - 2)))
                    if perplexity_value < 5: perplexity_value = 5
                    n_components_for_reduction = 2
                    
                    pca_explained_variance_ratio = None # Initialize

                    if tfidf_matrix_dense.shape[1] < n_components_for_reduction:
                        st.warning(f"TF-IDF特征数 ({tfidf_matrix_dense.shape[1]}) 少于降维目标 ({n_components_for_reduction})，无法进行有效降维可视化。")
                        reduced_features = tfidf_matrix_dense[:, :tfidf_matrix_dense.shape[1]] 
                        if tfidf_matrix_dense.shape[1] == 1: 
                             reduced_features = np.hstack((reduced_features, np.zeros_like(reduced_features)))
                        elif tfidf_matrix_dense.shape[1] == 0:
                             reduced_features = np.zeros((tfidf_matrix_dense.shape[0], 2)) 
                    else:
                        if reduction_method == "PCA":
                            reducer = PCA(n_components=n_components_for_reduction, random_state=42)
                            reduced_features = reducer.fit_transform(tfidf_matrix_dense)
                            pca_explained_variance_ratio = reducer.explained_variance_ratio_
                        else: # t-SNE
                            reducer = TSNE(n_components=n_components_for_reduction, random_state=42, perplexity=perplexity_value, n_iter=300, init='pca', learning_rate='auto')
                            reduced_features = reducer.fit_transform(tfidf_matrix_dense)
                    
                    fig_scatter, ax_scatter = plt.subplots(figsize=(8, 6)) 
                    fig_scatter.patch.set_alpha(0) # Ensure figure background is transparent
                    # ax_scatter.patch.set_alpha(0) # ax facecolor is handled by plot_scatter_clusters bg_color

                    plot_scatter_clusters(ax_scatter, reduced_features, clusters, cluster_model_name,
                                          colormap=selected_colormap, 
                                          marker_size=scatter_marker_size, 
                                          marker_alpha=scatter_marker_alpha,
                                          marker_styles_enabled=scatter_marker_styles_enabled,
                                          bg_color=scatter_bg_color, 
                                          show_grid=scatter_show_grid, 
                                          grid_color=scatter_grid_color)
                    ax_scatter.set_title(f"{reduction_method} 可视化 ({cluster_model_name})", fontsize=14)
                    st.pyplot(fig_scatter)

                    if pca_explained_variance_ratio is not None:
                        st.markdown("#### 💡 PCA 解释方差贡献：")
                        pca_col1, pca_col2, pca_col3 = st.columns(3)
                        pca_col1.metric("主成分 1", f"{pca_explained_variance_ratio[0]:.2%}")
                        if len(pca_explained_variance_ratio) > 1:
                            pca_col2.metric("主成分 2", f"{pca_explained_variance_ratio[1]:.2%}")
                            pca_col3.metric("累计贡献", f"{np.sum(pca_explained_variance_ratio):.2%}")
                        else:
                            pca_col2.metric("累计贡献", f"{np.sum(pca_explained_variance_ratio):.2%}")

                if cluster_model_name == "AgglomerativeClustering":
                    with st.expander("层次聚类树状图设置与可视化"):
                        st.markdown("#### 🌳 层次聚类树状图") # Was "**层次聚类树状图:**"
                        current_linkage_method = locals().get('linkage_agg', 'ward') 
                        current_metric = locals().get('metric_agg', 'euclidean')
                        
                        # Simplified dendrogram controls
                        dendro_orientation = st.selectbox("树状图方向:", ['top', 'bottom', 'left', 'right'], index=0, key="dendro_orientation_simple_select")
                        default_color_thresh_simple = 0.7 * linkage(tfidf_matrix_dense, method=current_linkage_method, metric=current_metric)[:,2].max() if tfidf_matrix_dense is not None and tfidf_matrix_dense.size > 0 and tfidf_matrix_dense.shape[0] >1 else 0.0
                        dendro_color_threshold_simple = st.number_input("颜色阈值 (距离, 0禁用):", min_value=0.0, value=default_color_thresh_simple, step=0.1, key="dendro_color_thr_simple_input")
                        if dendro_color_threshold_simple <= 0: dendro_color_threshold_simple = None
                        
                        with st.spinner("生成树状图中..."):
                            fig_dendro, ax_dendro = plt.subplots(figsize=(10, 7)) # Reverted to fixed height
                            plot_dynamic_dendrogram(ax_dendro, tfidf_matrix_dense, 
                                                    linkage_method=current_linkage_method, 
                                                    metric=current_metric,
                                                    orientation=dendro_orientation, 
                                                    color_threshold=dendro_color_threshold_simple)
                            st.pyplot(fig_dendro)
                            
            else: # This 'else' pairs with 'if clusters is not None:'
                # This will be shown if the clustering process in the try-except block failed and clusters remained None,
                # or if it's the initial state before any clustering is attempted for the current raw_texts.
                if selected_analysis == "🧩 文本聚类分析": # Only show this if we are in the clustering section
                    st.info("请配置参数并运行聚类模型，结果和可视化将在此显示。")

else: # No raw_texts AND not on homepage
    if selected_analysis and selected_analysis != "🏠 主页": # Only show if an analysis was selected but no text
         st.info("请在上方粘贴文本、上传文件或载入示例数据以开始分析。", icon="⬆️")

st.sidebar.markdown("--- "*10)
st.sidebar.info("© 2024 中文NLP智能分析平台") 