# ==========================================
# AI COMPANY PORTFOLIO GENERATOR
# Structured Function Calling Version
# ==========================================

import os, re, json, sqlite3
import streamlit as st
from typing import TypedDict, List, Dict
from dotenv import load_dotenv
from bs4 import BeautifulSoup
from urllib.parse import urljoin

from playwright.sync_api import sync_playwright

from langgraph.graph import StateGraph, END
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser

from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, ListFlowable, ListItem
from reportlab.lib.styles import getSampleStyleSheet

# ---------------------------
# SETUP
# ---------------------------

load_dotenv()
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# ---------------------------
# DATABASE
# ---------------------------

def save_to_db(data):
    conn = sqlite3.connect("companies.db")
    cur = conn.cursor()
    cur.execute("""
    CREATE TABLE IF NOT EXISTS companies (
        name TEXT,
        website TEXT,
        json TEXT
    )
    """)
    cur.execute(
        "INSERT INTO companies VALUES (?,?,?)",
        (data["company_name"], data["contact"]["website"], json.dumps(data))
    )
    conn.commit()
    conn.close()

# ---------------------------
# GRAPH STATE
# ---------------------------

class GraphState(TypedDict):
    url: str
    raw_html: str
    clean_text: str
    emails: List[str]
    phones: List[str]
    addresses: List[str]
    socials: Dict[str,str]
    logo: str
    data: Dict
    pdf_path: str

# ---------------------------
# MULTI PAGE CRAWLER
# ---------------------------

def crawl_site(url, limit=10):
    pages = []

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()
        page.goto(url, timeout=60000)
        page.wait_for_timeout(3000)

        base = page.content()
        pages.append(base)

        soup = BeautifulSoup(base, "html.parser")

        links = []
        for a in soup.find_all("a", href=True):
            h = a["href"].lower()
            if any(x in h for x in [
                "about","company","service","product",
                "solution","team","leadership","contact"
            ]):
                links.append(urljoin(url, a["href"]))

        for l in list(set(links))[:limit]:
            try:
                page.goto(l)
                page.wait_for_timeout(2500)
                pages.append(page.content())
            except:
                pass

        browser.close()

    return " ".join(pages)

# ---------------------------
# SCRAPE NODE
# ---------------------------

def scrape_node(state):
    return {"raw_html": crawl_site(state["url"])}

# ---------------------------
# CLEAN NODE
# ---------------------------

def clean_node(state):
    soup = BeautifulSoup(state["raw_html"], "html.parser")
    for t in soup(["script","style","noscript"]):
        t.decompose()

    text = soup.get_text(" ")
    text = re.sub(r"[^\x00-\x7F]+", " ", text)

    return {"clean_text": text}

# ---------------------------
# CONTACT NODE
# ---------------------------

def contact_node(state):

    soup = BeautifulSoup(state["raw_html"], "html.parser")

    emails, phones, addresses = set(), set(), set()
    socials = {"linkedin":"","instagram":"","facebook":"","twitter":""}

    for a in soup.find_all("a", href=True):
        h = a["href"]

        if h.startswith("mailto:"):
            emails.add(h.replace("mailto:","").split("?")[0])

        if h.startswith("tel:"):
            phones.add(h.replace("tel:",""))

        if "linkedin.com/company" in h:
            socials["linkedin"] = h
        if "instagram.com" in h:
            socials["instagram"] = h
        if "facebook.com" in h:
            socials["facebook"] = h
        if "twitter.com" in h or "x.com" in h:
            socials["twitter"] = h

    text = soup.get_text("\n")

    for e in re.findall(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}", text):
        emails.add(e)

    for p in re.findall(r"\+?\d[\d\s\-]{8,15}\d", text):
        clean = re.sub(r"[^\d+]", "", p)
        if 10 <= len(clean) <= 13:
            phones.add(clean)

    for line in text.split("\n"):
        if any(k in line.lower() for k in ["india","street","road","floor","building","office"]):
            if 30 < len(line) < 200:
                addresses.add(line.strip())

    return {
        "emails": list(emails),
        "phones": list(phones),
        "addresses": list(addresses),
        "socials": socials
    }

# ---------------------------
# LOGO NODE
# ---------------------------

def logo_node(state):
    soup = BeautifulSoup(state["raw_html"], "html.parser")
    for img in soup.find_all("img"):
        src = img.get("src","")
        alt = img.get("alt","").lower()
        if "logo" in src.lower() or "logo" in alt:
            return {"logo": urljoin(state["url"], src)}
    return {"logo": ""}

# ---------------------------
# AI NODE (STRUCTURED)
# ---------------------------

parser = JsonOutputParser()

prompt = ChatPromptTemplate.from_template("""
You are a business data extractor.

Return data ONLY in this JSON schema:

{format_instructions}

TEXT:
{input}
""")

chain = prompt | llm | parser

def ai_node(state):

    data = chain.invoke({
        "input": state["clean_text"][:20000],
        "format_instructions": """
{
 "company_name": "",
 "tagline": "",
 "overview": "",
 "products_services": [],
 "industry": "",
 "technology_stack": [],
 "leadership": []
}
"""
    })

    return {"data": data}

# ---------------------------
# NORMALIZE NODE
# ---------------------------

def normalize_node(state):
    d = state["data"]

    d["contact"] = {
        "website": state["url"],
        "emails": state["emails"],
        "phones": state["phones"],
        "addresses": state["addresses"],
        "socials": state["socials"]
    }

    d["logo"] = state["logo"]

    save_to_db(d)

    return {"data": d}

# ---------------------------
# PDF NODE
# ---------------------------

def pdf_node(state):

    d = state["data"]
    filename = f"{d['company_name'].replace(' ','_')}.pdf"

    doc = SimpleDocTemplate(filename)
    styles = getSampleStyleSheet()
    e=[]

    e.append(Paragraph(d["company_name"], styles["Title"]))
    e.append(Paragraph(d["tagline"], styles["Italic"]))
    e.append(Spacer(1,12))

    def sec(t): e.append(Paragraph(t, styles["Heading2"]))

    sec("Overview")
    e.append(Paragraph(d["overview"], styles["BodyText"]))

    sec("Products & Services")
    e.append(ListFlowable([ListItem(Paragraph(x,styles["BodyText"])) for x in d["products_services"]]))

    sec("Technology Stack")
    e.append(ListFlowable([ListItem(Paragraph(x,styles["BodyText"])) for x in d["technology_stack"]]))

    sec("Leadership")
    e.append(ListFlowable([ListItem(Paragraph(x,styles["BodyText"])) for x in d["leadership"]]))

    doc.build(e)

    return {"pdf_path": filename}

# ---------------------------
# GRAPH
# ---------------------------

builder = StateGraph(GraphState)

builder.add_node("scrape", scrape_node)
builder.add_node("clean", clean_node)
builder.add_node("contact", contact_node)
builder.add_node("logo", logo_node)
builder.add_node("ai", ai_node)
builder.add_node("normalize", normalize_node)
builder.add_node("pdf", pdf_node)

builder.set_entry_point("scrape")

builder.add_edge("scrape","clean")
builder.add_edge("clean","contact")
builder.add_edge("contact","logo")
builder.add_edge("logo","ai")
builder.add_edge("ai","normalize")
builder.add_edge("normalize","pdf")
builder.add_edge("pdf", END)

graph = builder.compile()

# ---------------------------
# STREAMLIT UI
# ---------------------------

st.title("🏢 AI Portfolio Generator")

url = st.text_input("Enter Company Website")

if st.button("Generate"):

    if not url:
        st.warning("Enter URL")
        st.stop()

    with st.spinner("Crawling & analyzing..."):

        state = {
            "url": url,
            "raw_html":"",
            "clean_text":"",
            "emails":[],
            "phones":[],
            "addresses":[],
            "socials":{},
            "logo":"",
            "data":{},
            "pdf_path":""
        }

        result = graph.invoke(state)

    d = result["data"]

    if d.get("logo"):
        st.image(d["logo"], width=180)

    st.header(d["company_name"])
    st.subheader(d["tagline"])

    st.markdown("### Overview")
    st.write(d["overview"])

    st.markdown("### Products & Services")
    for x in d["products_services"]:
        st.write("-",x)

    st.markdown("### Industry")
    st.write(d["industry"])

    st.markdown("### Technology Stack")
    for x in d["technology_stack"]:
        st.write("-",x)

    st.markdown("### Leadership")
    for x in d["leadership"]:
        st.write("-",x)

    st.markdown("### Contact")
    c = d["contact"]
    st.write("Website:", c["website"])
    st.write("Emails:", ", ".join(c["emails"]))
    st.write("Phones:", ", ".join(c["phones"]))

    st.markdown("Addresses:")
    for a in c["addresses"]:
        st.write("-",a)

    st.markdown("Social Media:")
    for k,v in c["socials"].items():
        if v:
            st.write(k.capitalize(),":",v)

    with open(result["pdf_path"],"rb") as f:
        st.download_button("Download PDF", f, file_name=result["pdf_path"])
