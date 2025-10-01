print("[DEBUG] bot.py version: 2025-10-02-router-fixed")  # debug marker

import os
import re
import shutil
import asyncio
from typing import Dict, Optional

import discord
from discord.ext import commands
from dotenv import load_dotenv, find_dotenv

# LangChain / OpenAI
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.text_splitter import CharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_core.documents import Document

# ----------------- Env -----------------
load_dotenv(find_dotenv())
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
DISCORD_TOKEN = os.environ.get("DISCORD_TOKEN")

# ----------------- Config -----------------
DOC_PATH = os.getenv("DOC_PATH", "documents.txt")
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", 800))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", 200))
RETRIEVER_K = int(os.getenv("RETRIEVER_K", 8))

DISCLAIMER = "\n\n*_(Adam can make mistakes. Always verify important info.)_*"

FALLBACK_SNIPPET = (
    "I couldn't find this directly in our mentorship material. "
    "But based on my knowledge, here's what I can share:"
)

# ----------------- Tiny in-process memory (per run) -----------------
USER_PROFILE: Dict[int, Dict[str, str]] = {}  # {user_id: {"name": "drex"}}

# ----------------- Load & build docs -----------------
abs_doc_path = os.path.abspath(DOC_PATH)
print(f"[DEBUG] Using document file: {abs_doc_path}")

with open(DOC_PATH, "r", encoding="utf-8") as f:
    raw_text = f.read()

pattern = r"### Section:\s*(.*)"
parts = re.split(pattern, raw_text)

sections = []
docs = []
splitter = CharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)

for i in range(1, len(parts), 2):
    header = parts[i].strip()
    content = parts[i + 1].strip() if i + 1 < len(parts) else ""
    sections.append(header)
    for chunk in splitter.split_text(content):
        docs.append(Document(page_content=chunk, metadata={"section": header}))

print(f"[DEBUG] Sections found: {len(sections)}")
print(f"[DEBUG] Total chunks indexed: {len(docs)}")

# ----------------- Vector DB (auto-wipe fresh) -----------------
persist_dir = "db_sections"
if os.path.exists(persist_dir):
    shutil.rmtree(persist_dir, ignore_errors=True)

embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
db = Chroma.from_documents(docs, embeddings, persist_directory=persist_dir)
retriever = db.as_retriever(search_kwargs={"k": RETRIEVER_K})

# ----------------- Models -----------------
chat = ChatOpenAI(temperature=0, openai_api_key=OPENAI_API_KEY)

# Router prompt: return strictly CASUAL or MATERIAL
ROUTER_PROMPT = """You are a classifier. Decide if the user's question is:
- CASUAL  → greetings, small talk, feelings, personal info (“what’s my name?”), time/date/day/year, general chit-chat, non-course questions.
- MATERIAL → any question about the trading course, mentorship content, modules, definitions, strategies, examples, assignments, or anything likely answered by the course materials.

Return exactly one word: CASUAL or MATERIAL.

Question: {question}
Label:"""

def classify_query(question: str) -> str:
    try:
        out = chat.invoke(ROUTER_PROMPT.format(question=question.strip()))
        label = (out.content or "").strip().upper()
        return "MATERIAL" if "MATERIAL" in label else "CASUAL"
    except Exception as e:
        print("[ROUTER ERROR]", e)
        return "CASUAL"

# MATERIAL prompt (retrieval)
from langchain.prompts import PromptTemplate
QA_PROMPT = PromptTemplate(
    template=(
        "You are Adam AI, a friendly community support assistant for a trading mentorship.\n"
        "Use ONLY the provided context to answer like a concise, helpful coach. "
        "Explain briefly and directly.\n"
        "If the answer is not in the context, say exactly:\n"
        f"\"{FALLBACK_SNIPPET}\"\n\n"
        "Context:\n{context}\n\nQuestion: {question}\nAnswer:"
    ),
    input_variables=["context", "question"],
)

from langchain.chains import RetrievalQA
qa_chain = RetrievalQA.from_chain_type(
    llm=chat,
    retriever=retriever,
    chain_type="stuff",
    return_source_documents=True,
    chain_type_kwargs={"prompt": QA_PROMPT},
)

CASUAL_PROMPT = """You are Adam AI, a friendly *community support* assistant for a trading mentorship.
Reply in a brief, warm tone (1–3 sentences). Be helpful and human—no formalities.
If the user asked the time/date/day/year, answer directly.
Do not invent course citations. Do not add a “Read more” section.
User: {question}
Answer:"""

def casual_answer(question: str, name: Optional[str]) -> str:
    try:
        greet = f"Hey **{name}**! " if name else ""
        out = chat.invoke(CASUAL_PROMPT.format(question=question.strip()))
        return (greet + (out.content or "").strip() + DISCLAIMER).strip()
    except Exception as e:
        print("[CASUAL ERROR]", e)
        return (greet + "How can I help you today?" + DISCLAIMER).strip()

def extract_name(text: str) -> Optional[str]:
    t = text.strip().lower()
    for p in [r"my name is\s+(.+)", r"call me\s+(.+)", r"you can call me\s+(.+)", r"i am\s+(.+)", r"i'm\s+(.+)"]:
        m = re.search(p, t)
        if m:
            return re.sub(r"[^\w\s\-']", "", m.group(1)).strip()
    return None

def is_fallback(text: str) -> bool:
    return "couldn't find this directly" in (text or "").lower()

# ----------------- Discord bot -----------------
intents = discord.Intents.default()
bot = commands.Bot(command_prefix="!", intents=intents)

@bot.event
async def on_ready():
    synced = await bot.tree.sync()
    print(f"Bot {bot.user} is online. Slash commands synced: {[c.name for c in synced]}")

# ----------------- /ask command (always defer) -----------------
@bot.tree.command(
    name="ask",
    description="Ask anything under the sun, Adam is here to help 🌞"
)
async def ask(interaction: discord.Interaction, question: str):
    uid = interaction.user.id
    q = question.strip()

    await interaction.response.defer(ephemeral=True)

    name_from_user = extract_name(q)
    if name_from_user:
        USER_PROFILE.setdefault(uid, {})["name"] = name_from_user
        return await interaction.followup.send(
            f"Nice to meet you, **{name_from_user}**! I’ll use that name in this session. 😊",
            ephemeral=True,
        )

    route = classify_query(q)
    print(f"[DEBUG] Router label: {route}")

    try:
        if route == "CASUAL":
            name = USER_PROFILE.get(uid, {}).get("name")
            text = casual_answer(q, name)
            return await interaction.followup.send(text, ephemeral=True)

        def run_chain():
            return qa_chain.invoke({"query": q})

        response = await asyncio.to_thread(run_chain)
        answer = (response.get("answer") or response.get("result") or "").strip()

        sections_used = []
        for doc in response.get("source_documents", []):
            sec = doc.metadata.get("section")
            if sec and sec not in sections_used:
                sections_used.append(sec)

        name = USER_PROFILE.get(uid, {}).get("name")
        prefix = f"Hey **{name}**!\n\n" if name else ""

        if sections_used and not is_fallback(answer):
            bullets = "\n".join(f"• *{s}*" for s in sections_used)
            msg = f"{prefix}**Question:** {q}\n\n**Answer:** {answer}\n\n*Read more:*\n{bullets}{DISCLAIMER}"
        else:
            msg = f"{prefix}**Question:** {q}\n\n**Answer:** {answer}{DISCLAIMER}"

        await interaction.followup.send(msg, ephemeral=True)

    except Exception as e:
        print("[ERROR]", e)
        await interaction.followup.send("⚠️ Sorry, I was unable to process your question.", ephemeral=True)

bot.run(DISCORD_TOKEN)
