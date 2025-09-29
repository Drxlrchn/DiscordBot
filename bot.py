import os
import re
import shutil
import datetime
import asyncio
import discord
from discord.ext import commands
from dotenv import load_dotenv, find_dotenv

# LangChain
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

# ----------------- Load & build docs -----------------
DOC_PATH = os.getenv("DOC_PATH", "documents.txt")
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", 800))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", 200))

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
if sections[:5]:
    print("[DEBUG] First 5 section titles:", sections[:5])
print(f"[DEBUG] Total chunks indexed: {len(docs)}")

# ----------------- Vector DB (auto-wipe) -----------------
persist_dir = "db_sections"
if os.path.exists(persist_dir):
    shutil.rmtree(persist_dir)

embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
db = Chroma.from_documents(docs, embeddings, persist_directory=persist_dir)
db.persist()
retriever = db.as_retriever(search_kwargs={"k": 5})

# ----------------- LLM & QA chain -----------------
chat = ChatOpenAI(temperature=0, openai_api_key=OPENAI_API_KEY)

QA_PROMPT = PromptTemplate(
    template=(
        "You are a trading mentor AI. Use the context below to answer the question clearly and helpfully.\n"
        "If the answer is not in the context, say exactly:\n"
        "\"I couldn’t find this directly in our mentorship material. But based on my knowledge, here’s what I can share:\"\n\n"
        "Context:\n{context}\n\nQuestion: {question}\nAnswer:"
    ),
    input_variables=["context", "question"],
)

qa_chain = RetrievalQA.from_chain_type(
    llm=chat,
    retriever=retriever,
    chain_type="stuff",
    return_source_documents=True,
    chain_type_kwargs={"prompt": QA_PROMPT},
)

# ----------------- Discord bot -----------------
intents = discord.Intents.default()
bot = commands.Bot(command_prefix="!", intents=intents)

@bot.event
async def on_ready():
    synced = await bot.tree.sync()
    print(f"Bot {bot.user} is online. Slash commands synced: {[c.name for c in synced]}")

@bot.tree.command(name="ask", description="Ask the trading mentor AI privately (ephemeral)")
async def ask(interaction: discord.Interaction, question: str):
    q_lower = question.lower()

    # Quick answers (skip LLM)
    if "year" in q_lower:
        return await interaction.response.send_message(
            f"The current year is {datetime.datetime.now().year}.", ephemeral=True
        )
    if "date" in q_lower:
        return await interaction.response.send_message(
            f"Today's date is {datetime.date.today().strftime('%B %d, %Y')}.", ephemeral=True
        )
    if "time" in q_lower:
        return await interaction.response.send_message(
            f"The current time is {datetime.datetime.now().strftime('%I:%M %p')}.", ephemeral=True
        )

    # Defer once — no double acknowledgement
    await interaction.response.defer(ephemeral=True)

    try:
        def run_chain():
            return qa_chain.invoke({"query": question})

        response = await asyncio.to_thread(run_chain)
        answer = response["result"].strip()

        # Collect unique section names
        sections_used = []
        for doc in response.get("source_documents", []):
            sec = doc.metadata.get("section")
            if sec and sec not in sections_used:
                sections_used.append(sec)

        disclaimer = "\n\n*_(Adam can make mistakes. Always verify important info.)_*"

        # Only show “Read more” if answer didn’t fall back
        if sections_used and "I couldn’t find this directly" not in answer:
            bullets = "\n".join(f"• *{s}*" for s in sections_used)
            msg = f"**Question:** {question}\n\n**Answer:** {answer}\n\n*Read more:*\n{bullets}{disclaimer}"
        else:
            msg = f"**Question:** {question}\n\n**Answer:** {answer}{disclaimer}"

        await interaction.followup.send(msg, ephemeral=True)

    except Exception as e:
        print("[ERROR]", e)
        await interaction.followup.send("⚠️ Sorry, I was unable to process your question.", ephemeral=True)

bot.run(DISCORD_TOKEN)
