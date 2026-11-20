import streamlit as st
import os
import tempfile
from difflib import SequenceMatcher
import base64
import json
import re
import pandas as pd
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()
if not os.environ.get("OPENAI_API_KEY"):
    os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]

FEEDBACK_LOG = Path("audit_feedback.jsonl")


def read_feedback_examples(limit=20):
    if not FEEDBACK_LOG.exists():
        return ""

    entries = []
    with FEEDBACK_LOG.open("r", encoding="utf-8") as log_file:
        for line in log_file:
            if line.strip():
                try:
                    entries.append(json.loads(line))
                except json.JSONDecodeError:
                    continue

    relevant_entries = entries[-limit:]
    if not relevant_entries:
        return ""

    return "\n".join(
        f"- {entry['output_type']} for {entry['file_names']}: "
        f"{entry['status']}. {entry['comments']}"
        for entry in relevant_entries
        if entry.get("status") and entry.get("comments") is not None
    )


def save_feedback(output_type, file_names, status, comments, ai_output):
    entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "output_type": output_type,
        "file_names": file_names,
        "status": status,
        "comments": comments,
        "ai_output": ai_output,
    }
    with FEEDBACK_LOG.open("a", encoding="utf-8") as log_file:
        log_file.write(json.dumps(entry, ensure_ascii=False) + "\n")


def load_and_split(uploaded_file):
    temp_path = None

    try:
        with tempfile.NamedTemporaryFile(
            suffix=".pdf",
            delete=False,
        ) as temp_file:
            temp_file.write(uploaded_file.getvalue())
            temp_path = temp_file.name

        documents = PyPDFLoader(temp_path).load()
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=800,
            chunk_overlap=100,
        )
        return splitter.split_documents(documents)
    finally:
        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)


@st.cache_resource
def build_vectorstore(_chunks):
    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-small",
        api_key=os.environ["OPENAI_API_KEY"],
    )
    return Chroma.from_documents(
        documents=_chunks,
        embedding=embeddings,
    )


def extract_image_text(uploaded_file):
    image_bytes = uploaded_file.getvalue()
    encoded_image = base64.b64encode(image_bytes).decode("utf-8")
    mime_type = uploaded_file.type or "image/png"

    model = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    response = model.invoke([
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": (
                        "Extract all visible text from this image. "
                        "Preserve headings and line breaks. "
                        "If the image contains little or no text, describe its visible content."
                    ),
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:{mime_type};base64,{encoded_image}",
                    },
                },
            ],
        }
    ])
    return response.content


def get_document_text(uploaded_file):
    if uploaded_file.type == "application/pdf":
        chunks = load_and_split(uploaded_file)
        return "\n".join(chunk.page_content for chunk in chunks)

    extracted_text = extract_image_text(uploaded_file)
    return extracted_text


def parse_ai_json(response_text):
    try:
        return json.loads(response_text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", response_text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group())
            except json.JSONDecodeError:
                return None
    return None


st.set_page_config(page_title="Document Comparing", page_icon="📄")

st.markdown(
    """
    <style>
    .stApp {
        background-color: #f3f5f7;
        color: #242425;
    }
    [data-testid="stHeader"] {
        background-color: #ffffff;
    }
    [data-testid="stSidebar"] {
        background-color: #071d49;
    }
    [data-testid="stSidebar"] * {
        color: #ffffff;
    }
    h1, h2, h3 {
        color: #071d49;
    }
    a {
        color: #a82018;
    }
    div.stButton > button {
        color: #ffffff;
        background-color: #071d49;
        border: 1px solid #071d49;
    }
    div.stButton > button:hover {
        color: #ffffff;
        background-color: #a82018;
        border-color: #a82018;
    }
    [data-testid="stFileUploader"] {
        background-color: #ffffff;
        border: 1px solid #b5bbc8;
        border-radius: 0.5rem;
    }
    [data-testid="stExpander"] {
        background-color: #ffffff;
        border: 1px solid #b5bbc8;
        border-radius: 0.5rem;
    }
    [data-testid="stMetric"] {
        background-color: #e6e8ed;
        border-left: 4px solid #80a0af;
        padding: 0.75rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

page = st.sidebar.radio(
    "Navigate",
    ["Document Comparison", "Methodology", "About Us"],
)

if page == "Methodology":
    st.title("Methodology")
    st.write(
        "The workflow converts uploaded documents into reviewable text, extracts financial "
        "details, and compares document pairs. AI findings are intended to support auditor review."
    )

    st.subheader("Process flow")
    st.graphviz_chart(
        """
        digraph document_audit {
            rankdir=TB;
            node [shape=box];
            upload [label="Upload up to 4 documents"];
            file_type [label="Identify file type"];
            pdf [label="PDF: load text and split into chunks"];
            image [label="Image: extract visible text with AI"];
            extract [label="Extract items, quantities, unit prices,\\nline totals, adjustments, and grand total"];
            compare [label="Compare document pairs; prioritize\\nitem, quantity, and unit-price differences"];
            review [label="Auditor reviews each output"];
            feedback [label="Save review feedback for later prompts"];
            upload -> file_type;
            file_type -> pdf [label="PDF"];
            file_type -> image [label="Image"];
            pdf -> extract;
            image -> extract;
            extract -> compare;
            compare -> review;
            review -> feedback [label="When saved"];
        }
        """,
        use_container_width=True,
    )

    st.subheader("1. Document intake and text extraction")
    st.markdown(
        """
        - The app accepts up to four PDF or image files per comparison.
        - PDFs are temporarily saved so the PDF loader can read them. Their text is split into
          overlapping chunks of 800 characters with 100 characters of overlap; the temporary file
          is deleted after loading.
        - Images are sent to a vision-capable OpenAI model to transcribe visible text or describe
          their contents.
        """
    )

    st.subheader("2. Per-document financial extraction")
    st.markdown(
        """
        For each document, an AI model is asked to return line items (item, quantity, unit price,
        and line total), adjustments such as fees, taxes or surcharges, and the grand total.
        The interface displays these results and lets the user check each line item, adjustment,
        and total separately.
        """
    )

    st.subheader("3. Pairwise comparison")
    st.markdown(
        """
        Each pair of extracted document texts is sent to an AI auditor prompt. It prioritizes
        matching line items and explicitly comparing stated quantities, units, and unit prices;
        missing items and unclear values are flagged rather than inferred. Other discrepancies,
        including taxes, surcharges, and totals, are secondary checks. Common points and individual
        differences can each be checked and annotated by the user.

        A displayed text-similarity percentage is calculated with a character-sequence comparison.
        It is a rough text-overlap indicator, not a semantic or financial accuracy score.
        """
    )

    st.subheader("4. Feedback and limitations")
    st.markdown(
        """
        When a user saves a check, its status, comments, and AI output are appended to a local
        `audit_feedback.jsonl` log. Recent entries are supplied as examples in later AI prompts;
        this is prompt-based feedback reuse, not model training. The log may contain sensitive
        financial details, so restrict access and follow your data-retention requirements.

        AI can miss, misread, or miscalculate information, especially in low-quality scans or complex
        layouts. Verify results against the originals and applicable accounting and tax rules.
        """
    )
    st.stop()

if page == "About Us":
    st.title("About Us")
    st.write(
        "We help teams review business documents by highlighting item, quantity, "
        "and price differences for people to verify."
    )

    st.subheader("Why auditing matters")
    st.markdown(
        """
        - **Protect financial accuracy:** comparing invoices and supporting documents can help uncover
          pricing, quantity, tax, and calculation discrepancies, which provides statistics on our import and export invoices and helps identify potential overpayments or underpayments.
        - **Strengthen controls:** a documented review trail supports accountability and helps teams
          resolve exceptions consistently.
        - **Keep people in the loop:** automated findings are prompts for review, not a substitute for
          professional judgment or source-document verification.
        """
    )

    st.subheader("Revenue recovery and import GST")
    st.markdown(
        """
        A careful review of import declarations, customs records, supplier invoices, and payment
        documents can help identify import GST that may have been overpaid, duplicated, misclassified,
        or left unclaimed. Confirm each potential recovery against the original records and applicable
        tax rules, deadlines, and eligibility requirements. Recoverability varies by jurisdiction and
        business circumstances; a difference flagged by this tool is not, by itself, proof that a
        refund or credit is due.
        """
    )
    st.info("Use these results as audit support and have a qualified tax professional verify GST claims.")
    st.stop()

st.title("Auditor Comparison Tool")

with st.expander("Important: educational capstone project disclaimer", expanded=True):
    st.warning(
        """
        IMPORTANT NOTICE: This web application is a prototype developed for educational purposes only. The information provided here is NOT intended for real-world usage and should not be relied upon for making any decisions, especially those related to financial, legal, or healthcare matters.

Furthermore, please be aware that the LLM may generate inaccurate or incorrect information. You assume full responsibility for how you use any generated output.

Always consult with qualified professionals for accurate and personalised advice.
        """
    )

uploaded_files = st.file_uploader(
    "📄 Upload a Document",
    type=["pdf", "png", "jpg", "jpeg", "gif", "bmp", "webp"],
    accept_multiple_files=True,
)

if uploaded_files:
    uploaded_files = uploaded_files[:4]
    for uploaded_file in uploaded_files:
        st.success(f"Loaded: {uploaded_file.name}")
else:
    st.info("Upload documents to enable document comparison.")

start_comparison = st.button(
    "Start Comparison",
    disabled=not uploaded_files,
)

if start_comparison:
    try:
        document_texts = {}

        with st.spinner("Reading and processing documents..."):
            for uploaded_file in uploaded_files:
                text = get_document_text(uploaded_file)
                document_texts[uploaded_file.name] = text

                build_vectorstore([
                    Document(
                        page_content=text,
                        metadata={"source": uploaded_file.name},
                    )
                ])

        st.success("Documents processed successfully.")

        if len(document_texts) < 2:
            st.info("Upload at least two documents to compare them.")
        else:
            st.subheader("Comparison Results")
            filenames = list(document_texts)

            table_model = ChatOpenAI(
                model="gpt-4o-mini",
                temperature=0,
            )

            st.subheader("Item and Amount Breakdown")
            for filename in filenames:
                document_text = document_texts[filename]
                feedback_examples = read_feedback_examples()
                table_response = table_model.invoke(
                    "Act as an auditor extracting financial data from the document below. "
                    "Return only valid JSON with this structure: "
                    '{"items":[{"item":"...","quantity":"...","unit_price":"...","line_total":"..."}],'
                    '"adjustments":[{"name":"...","amount":"..."}],"grand_total":"..."} '
                    "List every identifiable purchased item or charge. "
                    "Use N/A only when information is absent; do not invent values. "
                    "Include surcharges, fees, taxes, discounts, and all adjustments. "
                    "Use the stated grand total, or calculate it only if absent.\n\n"
                    "Use previously verified auditor feedback to improve future checks:\n"
                    f"{feedback_examples or 'No verified feedback is available yet.'}\n\n"
                    f"Document ({filename}):\n{document_text[:12000]}"
                )

                with st.expander(filename, expanded=True):
                    extracted = parse_ai_json(table_response.content)
                    if extracted is None:
                        st.error("The AI extraction could not be converted to structured data.")
                        st.code(table_response.content)
                        continue

                    items = extracted.get("items", [])
                    adjustments = extracted.get("adjustments", [])

                    st.markdown("**Line items**")
                    if items:
                        st.table(pd.DataFrame(items))
                        for index, item in enumerate(items):
                            item_label = item.get("item", f"Item {index + 1}")
                            st.markdown(f"**Check: {item_label}**")
                            item_status = st.radio(
                                "Correctness",
                                ["Not checked", "Correct", "Partially correct", "Incorrect"],
                                key=f"{filename}_item_{index}",
                                horizontal=True,
                            )
                            item_comments = st.text_input(
                                "Correction or explanation",
                                key=f"{filename}_item_comment_{index}",
                            )
                            if st.button("Save line-item check", key=f"{filename}_item_save_{index}"):
                                save_feedback(
                                    "line item",
                                    filename,
                                    item_status,
                                    item_comments,
                                    json.dumps(item, ensure_ascii=False),
                                )
                                st.success("Line-item check saved.")
                    else:
                        st.info("No line items were identified.")

                    st.markdown("**Adjustments and surcharges**")
                    if adjustments:
                        st.table(pd.DataFrame(adjustments))
                        for index, adjustment in enumerate(adjustments):
                            adjustment_name = adjustment.get("name", f"Adjustment {index + 1}")
                            st.markdown(f"**Check: {adjustment_name}**")
                            adjustment_status = st.radio(
                                "Correctness",
                                ["Not checked", "Correct", "Partially correct", "Incorrect"],
                                key=f"{filename}_adjustment_{index}",
                                horizontal=True,
                            )
                            adjustment_comments = st.text_input(
                                "Correction or explanation",
                                key=f"{filename}_adjustment_comment_{index}",
                            )
                            if st.button("Save adjustment check", key=f"{filename}_adjustment_save_{index}"):
                                save_feedback(
                                    "adjustment",
                                    filename,
                                    adjustment_status,
                                    adjustment_comments,
                                    json.dumps(adjustment, ensure_ascii=False),
                                )
                                st.success("Adjustment check saved.")
                    else:
                        st.info("No additional adjustments were identified.")

                    st.markdown("**Grand total**")
                    st.metric("Grand Total", str(extracted.get("grand_total", "N/A")))
                    grand_status = st.radio(
                        "Was the grand total checked correctly?",
                        ["Not checked", "Correct", "Partially correct", "Incorrect"],
                        key=f"{filename}_grand_status",
                        horizontal=True,
                    )
                    grand_comments = st.text_input(
                        "Correct grand total or explanation",
                        key=f"{filename}_grand_comment",
                    )
                    if st.button("Save grand-total check", key=f"{filename}_grand_save"):
                        save_feedback(
                            "grand total",
                            filename,
                            grand_status,
                            grand_comments,
                            str(extracted.get("grand_total", "N/A")),
                        )
                        st.success("Grand-total check saved.")

            for first_index, first_name in enumerate(filenames):
                for second_name in filenames[first_index + 1:]:
                    first_text = document_texts[first_name]
                    second_text = document_texts[second_name]
                    similarity = SequenceMatcher(
                        None,
                        first_text,
                        second_text,
                    ).ratio()

                    with st.expander(
                        f"{first_name} vs. {second_name}",
                        expanded=True,
                    ):
                        st.metric(
                            "Text similarity",
                            f"{similarity:.1%}",
                        )

                        comparison_model = ChatOpenAI(
                            model="gpt-4o-mini",
                            temperature=0,
                        )
                        feedback_examples = read_feedback_examples()
                        response = comparison_model.invoke(
                            "Act as an auditor comparing the purchasable line items in these two documents. "
                            "Make unit price and quantity differences the primary focus. "
                            "First identify corresponding items by description, even if wording differs. "
                            "For each corresponding item, compare the stated quantity and unit price "
                            "separately, and report the exact value from each document when available. "
                            "Flag missing items, mismatched units (for example each, box, kg), "
                            "and values that are unclear or absent; do not infer or fabricate values. "
                            "List items whose descriptions, quantities, and unit prices agree as common points. "
                            "Put item-level quantity or unit-price mismatches first in differences, "
                            "then other discrepancies. Treat subtotals, taxes, surcharges, and grand totals "
                            "as secondary checks, and mention them only after the line-item findings. "
                            "Do not let general text similarity replace item-by-item price and quantity checks. "
                            "Return only valid JSON using this structure: "
                            '{"similarities":["..."],"differences":[{"point":"...","finding":"..."}],'
                            '"auditor_notes":["..."]}. '
                            "In every difference, state which document has which quantity and/or unit price, "
                            "including currency and unit where the source provides them. "
                            "Use auditor_notes for secondary checks and verification caveats.\n\n"
                            "Use previously verified auditor feedback to improve future checks:\n"
                            f"{feedback_examples or 'No verified feedback is available yet.'}\n\n"
                            f"Document 1 ({first_name}):\n{first_text[:12000]}\n\n"
                            f"Document 2 ({second_name}):\n{second_text[:12000]}"
                        )
                        comparison = parse_ai_json(response.content)
                        if comparison is None:
                            st.error("The AI comparison could not be converted to structured data.")
                            st.code(response.content)
                            continue

                        st.markdown("**Common items (matching quantity and unit price)**")
                        similarities = comparison.get("similarities", [])
                        if similarities:
                            for index, point in enumerate(similarities):
                                st.markdown(f"- {point}")
                                similarity_status = st.radio(
                                    "Was this similarity correct?",
                                    ["Not checked", "Correct", "Partially correct", "Incorrect"],
                                    key=f"{first_name}_{second_name}_similarity_{index}",
                                    horizontal=True,
                                )
                                similarity_comments = st.text_input(
                                    "Similarity correction or explanation",
                                    key=f"{first_name}_{second_name}_similarity_comment_{index}",
                                )
                                if st.button(
                                    "Save similarity check",
                                    key=f"{first_name}_{second_name}_similarity_save_{index}",
                                ):
                                    save_feedback(
                                        "similarity",
                                        f"{first_name} vs. {second_name}",
                                        similarity_status,
                                        similarity_comments,
                                        point,
                                    )
                                    st.success("Similarity check saved.")
                        else:
                            st.info("No common points were identified.")

                        st.markdown("**Priority differences: unit prices and quantities**")
                        differences = comparison.get("differences", [])
                        if differences:
                            for index, difference in enumerate(differences):
                                finding = difference.get("finding", "")
                                point = difference.get("point", f"Difference {index + 1}")
                                st.markdown(f"- **{point}:** {finding}")
                                difference_status = st.radio(
                                    "Was this difference correct?",
                                    ["Not checked", "Correct", "Partially correct", "Incorrect"],
                                    key=f"{first_name}_{second_name}_difference_{index}",
                                    horizontal=True,
                                )
                                difference_comments = st.text_input(
                                    "Difference correction or explanation",
                                    key=f"{first_name}_{second_name}_difference_comment_{index}",
                                )
                                if st.button(
                                    "Save difference check",
                                    key=f"{first_name}_{second_name}_difference_save_{index}",
                                ):
                                    save_feedback(
                                        "difference",
                                        f"{first_name} vs. {second_name}",
                                        difference_status,
                                        difference_comments,
                                        json.dumps(difference, ensure_ascii=False),
                                    )
                                    st.success("Difference check saved.")
                        else:
                            st.info("No key differences were identified.")

                        st.markdown("**Auditor notes**")
                        for note in comparison.get("auditor_notes", []):
                            st.markdown(f"- {note}")

    except KeyError:
        st.error("OPENAI_API_KEY is not set. Add it to your .env file.")
    except Exception as error:
        st.error(f"Could not process the documents: {error}")
