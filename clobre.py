import os
import json
import urllib.request
import zipfile
import io
import streamlit as st
import pytesseract
from pdf2image import convert_from_path

# ================= Configuration =================
OLLAMA_URL = "http://localhost:11434/api/generate"

# Adjust the paths for Tesseract and Poppler according to your laptop environment
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
#POPPLER_PATH = r"C:\Program Files\poppler\poppler-26.02.0\Library\bin"
POPPLER_PATH = r"C:\Program Files\poppler\poppler-26.09.0\Library\bin"

# AI Model Options (Order as requested)
ALL_MODELS = [
    #"gemma3:12b",
    "gemma4:26b",
    "gpt-oss:20b",
    #"command-r"
    #"gpt-oss:120b",
    #"qwen3.6:27b",
    #"qwen2.5-coder:7b",
    #"qwen3-coder:14b"
    #"qwen2.5-coder:32b",
    #"qwen3-coder:30b"
]

# Output Directory for Batch Processing
OUTPUT_DIR = "./batch_reviews"
os.makedirs(OUTPUT_DIR, exist_ok=True)
# =================================================

# =================================================

# =================================================

# 1. ブラウザのタブにロゴを設定 (page_iconに画像パスを指定)
st.set_page_config(
    page_title="CLOBRE: Closed LLM-Operationalized Batch Reviewing Engine", 
    page_icon="clobre.png", 
    layout="wide"
)

# 2. ロゴとタイトルを横並びにする
if os.path.exists("clobre.png"):
    # ロゴ用(1)とタイトル用(6)の比率でカラムを作成
    col_logo, col_title = st.columns([1, 6])
    
    with col_logo:
        # カラムの幅に合わせて自動調整。垂直位置の微調整用に少し上にマージンを入れても良いです
        st.image("clobre.png", use_container_width=True)
        
    with col_title:
        st.title("CLOBRE: Closed LLM-Operationalized Batch Reviewing Engine")
        #st.caption("Closed LLM-Operationalized Batch Reviewing Engine")
        st.caption("100% Local / Completely Offline / Live Batch Streaming & Single-File Output")
else:
    # 画像がない場合のフォールバック
    st.title("CLOBRE")
    st.caption("100% Local / Completely Offline / Live Batch Streaming & Single-File Output")

# --- ⚙️ Prompt Templates Definition ---

# 1. Academic Paper Review Template
PROMPT_ACADEMIC_REVIEW = """Review the following academic paper based STRICTLY on the provided paper text.
The report must be written in English, but each section must be immediately followed by its Japanese translation.

=========================================
[CRITICAL EVALUATION GUIDELINES]
- Maintain an objective, balanced, and constructive academic tone. 
- Every weakness, error, or comment you point out MUST be backed by 100% explicit evidence from the text.
- YOU MUST RIGOROUSLY CRITIQUE REPRODUCIBILITY & SCIENTIFIC VALIDITY. If the text lacks necessary details for replication, state that the paper fails to meet standard scientific disclosure requirements.
=========================================

Follow this exact structure:

0. Overall Writing Quality & Impression
[Provide a balanced evaluation of how well the paper is structured and written as a scientific text. Assess its logical flow, clarity, and overall readability. Write English review here]
(Japanese Translation):
[ここに和訳を記述]

1. Novelty & Significance
[Evaluate the novelty and theoretical/practical contributions of the paper based on the text. Write English review here]
(Japanese Translation):
[ここに和訳を記述]

2. Reproducibility & Methodological Rigor (CRITICAL CHECK)
[Exhaustively evaluate whether the paper provides sufficient information for an independent researcher to replicate the results. You MUST comment on the following three sub-items explicitly based on the text:
  A) Methodological Details: Are materials, configurations, hyper-parameters, statistical tests, or experimental protocols fully described? What is missing?
  B) Data Availability & Transparency: Does the paper state where the raw data, code, or materials are hosted (e.g., GitHub, Zenodo, repository URL)? If not explicitly mentioned, flag this as a critical flaw.
  C) Sample Size & Baseline Controls: Are the sample sizes, control groups, and benchmarks scientifically sound and clearly justified?
If any of these are missing or vague, conclude whether the paper is "Virtually Unreviewable" due to insufficient disclosure. Write English review here]
(Japanese Translation):
[ここに和訳を記述。上記A, B, Cの観点を含め、再現性と手法の厳密性を詳細に評価してください。]

3. Specific Comments / Scientific Weaknesses
[Identify any specific flaws, logical gaps, or unsubstantiated claims in the research itself. Base every comment strictly on factual evidence from the text. Write English review here]
(Japanese Translation):
[ここに和訳を記述]

4. Technical Errors, Typos, and Grammatical Flaws
[CRITICAL RULES FOR TYPOS/GRAMMAR]:
- List obvious typos, grammatical errors, or clear formatting mistakes found in the text, up to a maximum of 20 items.
- IGNORE all minor formatting artifacts or text breaking caused by the PDF-to-text extraction process. 
- ONLY report clear, genuine linguistic mistakes made by the authors. If there are none, explicitly state so.
[Write English review here]
(Japanese Translation):
[ここに和訳を記述]

---
PAPER TEXT:
{user_text}"""

# 2. Author Response/Rebuttal Evaluation Template
PROMPT_RESPONSE_EVAL = """Review and evaluate the following author response to the reviewers' comments based STRICTLY and ONLY on the provided text.
The evaluation report must be written in English, but each section must be immediately followed by its Japanese translation.

=========================================
[EVALUATION GUIDELINES]
- Maintain an objective, balanced, and constructive academic tone.
- Assess whether the authors have genuinely, politely, and scientifically addressed the reviewers' concerns.
- Every weakness, error, or comment you point out MUST be backed by 100% explicit evidence from the text.
=========================================

Follow this exact structure:

0. Overall Impression & Tone of Response
[Evaluate the overall quality, completeness, and politeness of the authors' responses. Is the logic flow convincing? Write English review here]
(Japanese Translation):
[ここに和訳を記述]

1. Strength of Rebuttals / Revisions
[Evaluate how effectively the authors defended their position or how well they revised the paper based on the comments. Write English review here]
(Japanese Translation):
[ここに和訳を記述]

2. Insufficient Points / Remaining Issues
[Identify any reviewer comments that were avoided, insufficiently answered, or logically flawed in the response. Base every comment strictly on factual evidence from the text. Write English review here]
(Japanese Translation):
[ここに和訳を記述]

3. Technical Errors or Typos in Response
[List obvious typos or clear grammatical errors made by the authors in this response document, up to a maximum of 20 items. IGNORE PDF extraction artifacts. Write English review here]
(Japanese Translation):
[ここに和訳を記述]

---
RESPONSE TEXT:
{user_text}"""

# 3. Abstract / Text Brush-up Template
PROMPT_BRUSH_UP = """You are an expert academic reviewer and English editor. 
Review the following abstract or manuscript text, and suggest logical and stylistic improvements to make it highly polished for prestigious scientific conferences or journals.
The evaluation report must be written in English, but each section must be immediately followed by its Japanese translation.

Follow this exact structure:

1. Overall Impression & Logic Flow
[Provide a comprehensive review of the strengths and clarity of the logic flow. Write English review here]
(Japanese Translation):
[ここに和訳を記述]

2. Specific Phrasing & Structural Suggestions
[Detail concrete suggestions on how to improve grammar, vocabulary, or the structural layout of the arguments. Write English review here]
(Japanese Translation):
[ここに和訳を記述]

3. Polished Version Proposal
[Provide the final, fully polished version of the text. Write English text here]
(Japanese Translation):
[必要に応じて、洗練された文章に対応する日本語訳や解説を記述]

---
TEXT TO BRUSH UP:
{user_text}"""

# 4. Graduate Seminar Abstract Brush-up Template
PROMPT_SEMINAR_BRUSH_UP = """You are an expert academic advisor and a senior scientist in a diverse biological sciences department. 
Your primary task is to CRITICALLY DECONSTRUCT the following draft abstract for a Graduate Research Seminar based on strict institutional rules, and provide a deep, highly strategic logical reconstruction.
The evaluation report must be written in English, but each section must be immediately followed by its Japanese translation.

=========================================
[GRADUATE SEMINAR MANDATORY RULES]
1. Citation Count: Must cite 7 or more papers in total.
2. Paper Types: At least 6 of the cited papers must be original research articles (NOT review papers).
3. Recency: At least 1 cited paper must be published within the last 3 years.
4. Cohesiveness: Must synthesize data from these papers into a single, well-structured, cohesive narrative.
5. Citation Format: In-text citations must be numbered sequentially in the order of appearance (e.g., [1], [2], [3]).
   * [CRITICAL FLEXIBILITY]: You CAN and SHOULD change the presentation order of the cited papers if rearranging them creates a more logical, impactful, or coherent argument.
6. Target Audience: The audience includes insect researchers, plant scientists, experimentalists, and theorists. The language must be accessible yet rigorous for all.
7. Logic & Depth (Anti-Cataloging Rule): Avoid shallow catalogs. Do NOT just list phenotypes or data (e.g., simply listing knockout/overexpression phenotypes of various plant miRNAs is shallow and boring). The narrative must have a deep, consistent, and unifying logical core or biological mechanism.

[LOGICAL FLOW GUIDELINES for a Diverse Audience]
- **The "Why" First**: Clearly state the universal biological question or overarching puzzle that bridges all sub-disciplines within the first two sentences.
- **Premise before Consequence**: Ensure information flows logically from foundational premises to specific discoveries.
- **The Core Thread**: Connect different papers not by "Also, Paper [2] showed X", but by mechanistic logic (e.g., "To understand the causal mechanism behind, [2] investigated...").
=========================================

Follow this exact structure:

1. Rule Compliance & Citation Re-ordering Report
[Analyze if the draft meets all numerical constraints. Explain if you rearranged the citation order to improve the narrative flow, providing a clear mapping of the sequence change. Write English review here]
(Japanese Translation):
[ここに和訳を記述]

2. Comprehensive Deconstruction of Flaws & Gaps (Deep Diagnostics)
[CRITICAL TASK: Exhaustively identify and list ALL problems, logical gaps, and structural weaknesses in the original draft. 
- Dissect the presentation order of information: What did the author explain too early or too late? 
- Point out where it falls into the "shallow cataloging" trap. 
- Explain exactly WHY a plant or insect researcher, an experimentalist, or a theorist would lose interest or get confused at specific lines. Be brutally honest and detailed. Write English review here]
(Japanese Translation):
[ここに和訳を記述]

3. Architectural Strategy & Logic Design
[Explain the strategy behind your proposed fix. Why is the new information sequence superior? How does your design bridge different sub-disciplines? Focus deeply on the "mechanistic thread" you are weaving. Write English review here]
(Japanese Translation):
[ここに和訳を記述]

4. Line-by-Line Specific Phrasing & Transition Suggestions
[Provide granular suggestions on sentence structure and word choice. Explain the logical purpose of each transition. Write English review here]
(Japanese Translation):
[ここに和訳を記述]

5. Polished Version Proposal with In-Line Rationale (The Final Output)
[CRITICAL FORMAT: Provide the final, fully polished, high-impact version of the abstract in English. 
*AFTER EVERY 1-2 SENTENCES* in this polished text, you MUST insert a bracketed rationale/commentary block (e.g., "[POLISHING POINT: Re-ordered this to establish the universal premise first so that theorists can follow the biological significance. Swapped Citation [2] and [3] to ensure cause precedes effect.]") explaining the exact purpose and strategic value of that specific part. Ensure strict sequential numbering, [2], [3] matches your optimized order.]
(Japanese Translation / Commentary):
[洗練された文章（英文）全体の日本語訳を記述し、さらに各「[POLISHING POINT]」に対応する日本語の詳しい解説・意図をここにまとめて記述してください。他分野の聴衆の心を掴むための議論の仕方のポイントを深く掘り下げてください。]

---
DRAFT ABSTRACT TO BRUSH UP:
{user_text}"""

# --- ⚙️ System Settings (Sidebar) ---
with st.sidebar:
    st.header("⚙️ Batch Configuration")
    
    selected_models = st.multiselect(
        "🤖 Select AI Models", 
        options=ALL_MODELS, 
        default=[ALL_MODELS[0], ALL_MODELS[1]] if len(ALL_MODELS) > 1 else ALL_MODELS
    )
    
    st.markdown("---")
    st.subheader("📋 Task Template")
    st.write("Select the template that matches your goal:")
    
    task_choice = st.selectbox(
        "Choose Task", 
        [
            "1. Academic Paper Review",
            "2. Author Response Evaluation",
            "3. Abstract & Text Brush-up",
            "4. Graduate School Seminar Brush-up"
        ]
    )
    
    if "1. Academic Paper Review" in task_choice:
        base_prompt_template = PROMPT_ACADEMIC_REVIEW
    elif "2. Author Response Evaluation" in task_choice:
        base_prompt_template = PROMPT_RESPONSE_EVAL
    elif "3. Abstract & Text Brush-up" in task_choice:
        base_prompt_template = PROMPT_BRUSH_UP
    else:
        base_prompt_template = PROMPT_SEMINAR_BRUSH_UP

    st.markdown("---")
    custom_system_prompt = st.text_area("Edit Prompt Template", value=base_prompt_template, height=350)

# --- 📝 Main Area: Inputs ---
st.subheader("📂 Upload Files for Batch Processing")
uploaded_files = st.file_uploader(
    "Drag and drop multiple PDF or TXT files here", 
    type=["pdf", "txt"], 
    accept_multiple_files=True
)

st.markdown("---")
st.subheader("✍️ Direct Text Input (Alternative)")
user_text_input = st.text_area("If you are not uploading files, paste your text here to process across the selected models:", height=150)

# --- 🚀 Batch Execution Logic ---

if "stop_requested" not in st.session_state:
    st.session_state.stop_requested = False

col_start, col_stop = st.columns(2) 
with col_start:
    start_clicked = st.button("🚀 Start Batch Processing", key="batch_start_btn")
with col_stop:
    if st.button("🛑 Stop / Cancel Processing", key="batch_stop_btn"):
        st.session_state.stop_requested = True
        st.warning("Stopping process... Please wait for the current segment to halt.")

if start_clicked:
    st.session_state.stop_requested = False
    
    if not selected_models:
        st.error("Please select at least one AI model.")
        st.stop()
        
    targets = []
    
    if uploaded_files:
        for f in uploaded_files:
            if st.session_state.stop_requested:
                st.error("⛔ Processing was cancelled by user.")
                st.session_state.stop_requested = False
                st.stop()
                
            file_ext = f.name.split('.')[-1].lower() if '.' in f.name else ''
            
            if file_ext == 'txt':
                with st.spinner(f"📝 Reading text file: {f.name} ..."):
                    try:
                        stringio = io.StringIO(f.getvalue().decode("utf-8"))
                        full_text = stringio.read()
                        targets.append({"name": f.name, "text": full_text, "type": "txt"})
                    except Exception as e:
                        st.error(f"❌ Failed to read {f.name}: {e}")
                        
            elif file_ext == 'pdf':
                with st.spinner(f"📄 Extracting text via OCR: {f.name} ..."):
                    temp_path = f"temp_{f.name}"
                    with open(temp_path, "wb") as temp_f:
                        temp_f.write(f.getbuffer())
                    try:
                        images = convert_from_path(temp_path, dpi=300, poppler_path=POPPLER_PATH)
                        full_text = "".join([pytesseract.image_to_string(img, lang='eng') for img in images])
                        targets.append({"name": f.name, "text": full_text, "type": "pdf"})
                        
                        # OCR結果テキストの自動保存
                        base_filename_str = f.name.rsplit('.', 1)[0] if '.' in f.name else f.name
                        ocr_filename = f"{base_filename_str}_ocr_extracted.txt"
                        ocr_path = os.path.join(OUTPUT_DIR, ocr_filename)
                        with open(ocr_path, "w", encoding="utf-8") as ocr_f:
                            ocr_f.write(full_text)
                        st.toast(f"💾 OCR text auto-saved: {ocr_filename}", icon="📝")
                        
                    except Exception as e:
                        st.error(f"❌ Failed OCR on {f.name}: {e}")
                    finally:
                        if os.path.exists(temp_path):
                            os.remove(temp_path)
                            
    elif user_text_input:
        targets.append({"name": "direct_input_text", "text": user_text_input, "type": "direct"})
        
    if not targets:
        st.error("No input data found. Please upload files or paste your text into the text area.")
        st.stop()

    st.markdown("### 📊 Input Parsing Summary")
    for t in targets:
        st.info(f"✅ **{t['name']}**: Loaded successfully with **{len(t['text'])}** characters.")

    st.markdown("---")

    total_tasks = len(targets) * len(selected_models)
    st.success(f"📦 Starting batch processing for {len(targets)} inputs × {len(selected_models)} models (= Total {total_tasks} tasks)...")
    
    progress_bar = st.progress(0)
    task_count = 0
    
    # 全論文・全モデルの結果を「1つに統合」するためのHTMLバッファ
    all_reports_html = ""
    html_style = """
    <style>
        body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; line-height: 1.6; color: #333; max-width: 1200px; margin: 0 auto; padding: 20px; }
        .paper-block { border: 2px solid #1f77b4; border-radius: 8px; padding: 20px; margin-bottom: 40px; background-color: #f8f9fa; }
        .paper-title { color: #1f77b4; margin-top: 0; border-bottom: 2px solid #1f77b4; padding-bottom: 10px; }
        .model-block { border-left: 4px solid #2ca02c; padding-left: 15px; margin: 20px 0; background: #ffffff; padding: 15px; border-radius: 4px; box-shadow: 0 1px 3px rgba(0,0,0,0.1); }
        .model-name { color: #2ca02c; font-weight: bold; font-size: 1.2em; margin-top: 0; }
        pre { white-space: pre-wrap; font-family: inherit; font-size: 1em; margin: 0; }
        hr { border: 0; border-top: 1px solid #ccc; margin: 20px 0; }
    </style>
    """
    all_reports_html += f"<!DOCTYPE html><html><head><meta charset='utf-8'>{html_style}<title>Consolidated Batch Review Report</title></head><body>"
    all_reports_html += "<h1>📝 CLOBRE: Closed LLM-Operationalized Batch Reviewing Engine</h1><hr>"

    for target in targets:
        raw_name = target["name"]
        base_filename_str = raw_name.rsplit('.', 1)[0] if '.' in raw_name else raw_name
        
        # 論文単位での自動保存用HTMLファイルパス
        output_filename = f"{base_filename_str}_all_reviews.html"
        output_path = os.path.join(OUTPUT_DIR, output_filename)
        
        # 論文個別のHTMLヘッダー書き込み
        with open(output_path, "w", encoding="utf-8") as out_f:
            out_f.write(f"<!DOCTYPE html><html><head><meta charset='utf-8'>{html_style}<title>{raw_name} - Reviews</title></head><body>")
            out_f.write(f"<div class='paper-block'><h2 class='paper-title'>📄 File: {raw_name}</h2>")

        # 統合ファイル用にも論文の枠を追加
        all_reports_html += f"<div class='paper-block'><h2 class='paper-title'>📄 File: {raw_name}</h2>"

        for model in selected_models:
            if st.session_state.stop_requested:
                st.error("⛔ Batch processing was cancelled by user midway.")
                st.session_state.stop_requested = False 
                st.stop() 
                
            task_count += 1
            st.markdown(f"### 📄 Result Block [{task_count}/{total_tasks}]: `{target['name']}` via **{model}**")
            
            try:
                final_prompt = custom_system_prompt.format(user_text=target["text"])
            except KeyError:
                final_prompt = f"{custom_system_prompt}\n\n【Target Text】\n{target['text']}"
            
            payload = {
                "model": model,
                "prompt": final_prompt,
                "stream": True,
                "options": {"num_ctx": 128000}
                #"options": {"num_ctx": 32000}
            }
            data = json.dumps(payload).encode('utf-8')
            req = urllib.request.Request(OLLAMA_URL, data=data, headers={'Content-Type': 'application/json'})
            
            try:
                response_placeholder = st.empty()
                full_response = ""
                
                with urllib.request.urlopen(req, timeout=None) as response:
                    for line in response:
                        if st.session_state.stop_requested:
                            break
                            
                        if line:
                            chunk = json.loads(line.decode('utf-8'))
                            text_chunk = chunk.get("response", "")
                            full_response += text_chunk
                            response_placeholder.markdown(full_response)
                            if chunk.get("done", False):
                                break
                                
                if st.session_state.stop_requested:
                    st.error("⛔ Processing halted during streaming.")
                    st.session_state.stop_requested = False
                    st.stop()
                
                # HTMLの各ブロックを生成
                model_html_chunk = f"<div class='model-block'><div class='model-name'>🤖 Model: {model}</div><pre>{full_response.strip()}</pre></div>"
                
                # 1. 論文単位の自動保存ファイルに「追記（append）」
                with open(output_path, "a", encoding="utf-8") as out_f:
                    out_f.write(model_html_chunk)
                
                # 2. 全論文まとめ用の変数にも「追記」
                all_reports_html += model_html_chunk
                
                st.markdown("---")
                st.toast(f"💾 Appended to HTML: {output_filename} ({model})", icon="✅")
                
            except Exception as e:
                st.error(f"❌ Error during execution: {e}")
            
            progress_bar.progress(task_count / total_tasks)
            
        # 各論文の全モデルループが終了した時点で、個別HTMLファイルの閉じタグを書き込む
        try:
            with open(output_path, "a", encoding="utf-8") as out_f:
                out_f.write("</div></body></html>")
        except Exception as e:
            st.error(f"❌ Failed to close HTML file for {raw_name}: {e}")
            
        # 統合レポート用にも論文ブロックの閉じタグを追加
        all_reports_html += "</div>"
            
    # 全論文・全モデルの処理が完全に終了したら、統合HTMLの閉じタグを追加
    all_reports_html += "</body></html>"
    
    st.balloons()
    st.success(f"🎉 All tasks completed! HTML reports safely compiled inside the `{OUTPUT_DIR}` directory.")

    # --- 📥 Main Download Area: 唯一の「統合レポートダウンロードボタン」 ---
    st.markdown("### 📦 Download Consolidated Report")
    st.info("The button below will download a single HTML file containing all reviews from all models across all papers. (OCR files are excluded and kept safe in your auto-save directory).")
    
    st.download_button(
        label="📥 Download All-in-One Integrated Report (HTML)",
        data=all_reports_html,
        file_name="all_papers_consolidated_report.html",
        mime="text/html"
    )
