print("[DEBUG] bot.py version: 2025-10-01-04:00 with memory")

import os, re, shutil, datetime, asyncio, discord
from discord.ext import commands
from dotenv import load_dotenv, find_dotenv
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.text_splitter import CharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_core.documents import Document

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
retriever = db.as_retriever(search_kwargs={"k": 8})

# ----------------- LLM & QA chain -----------------
chat = ChatOpenAI(temperature=0, openai_api_key=OPENAI_API_KEY)

QA_PROMPT = PromptTemplate(
    template=(
        "You are a friendly **community support AI** helping users with trading topics.\n"
        "Below is a brief conversation history for context:\n{history}\n\n"
        "If the question is about the mentorship material, answer like a clear teacher or coach using the context below — not just copy-paste, but explain it.\n"
        "If the question is casual (like who are you, hi, time/date), answer in a friendly way like ChatGPT.\n"
        "If the answer is not in the context, say exactly:\n"
        "\"I couldn’t find this directly in our mentorship material. But based on my knowledge, here’s what I can share:\"\n\n"
        "Context:\n{context}\n\nQuestion: {question}\nAnswer:"
    ),
    input_variables=["history", "context", "question"],
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

# ---- tiny in-memory chat history per user ----
user_history = {}  # user_id -> list of (q,a)

def format_history(user_id):
    history = user_history.get(user_id, [])
    lines = []
    for q, a in history[-5:]:  # last 5 turns
        lines.append(f"User: {q}\nAdam: {a}")
    return "\n".join(lines) if lines else "No prior context."

@bot.event
async def on_ready():
    synced = await bot.tree.sync()
    print(f"Bot {bot.user} is online. Slash commands synced: {[c.name for c in synced]}")

@bot.tree.command(name="ask", description="Ask Adam (community support AI) privately (ephemeral)")
async def ask(interaction: discord.Interaction, question: str):
    q_lower = question.lower()

    # Quick answers for casual questions without LLM
    if any(greet in q_lower for greet in ["hi", "hello", "hey", "who are you"]):
        return await interaction.response.send_message(
            "👋 Hi! I’m Adam AI — your friendly community support bot here to help answer questions about our material or just chat casually. 😊",
            ephemeral=True
        )
    if "how are you" in q_lower:
        return await interaction.response.send_message(
            "I'm doing great, thanks for asking! How about you? 😄", ephemeral=True
        )
    if "year" in q_lower:
        return await interaction.response.send_message(
            f"The current year is {datetime.datetime.now().year}.", ephemeral=True
        )
    if "date" in q_lower or "day" in q_lower:
        return await interaction.response.send_message(
            f"Today is {datetime.date.today().strftime('%A, %B %d, %Y')}.", ephemeral=True
        )
    if "time" in q_lower:
        return await interaction.response.send_message(
            f"The current time is {datetime.datetime.now().strftime('%I:%M %p')}.", ephemeral=True
        )

    # Defer once — no double acknowledgement
    await interaction.response.defer(ephemeral=True)

    try:
        def run_chain():
            history_str = format_history(interaction.user.id)
            return qa_chain.invoke({"history": history_str, "query": question})

        response = await asyncio.to_thread(run_chain)
        answer = response["result"].strip()

        # Update memory
        user_history.setdefault(interaction.user.id, []).append((question, answer))

        # Collect unique section names
        sections_used = []
        for doc in response.get("source_documents", []):
            sec = doc.metadata.get("section")
            if sec and sec not in sections_used:
                sections_used.append(sec)

        disclaimer = "\n\n*_(Adam can make mistakes. Always verify important info.)_*"

        if sections_used and "I couldn’t find this directly" not in answer:
            bullets = "\n".join(f"🔹 {s}" for s in sections_used)
            msg = f"**Question:** {question}\n\n**Answer:** {answer}\n\n📚 **Related sections:**\n{bullets}{disclaimer}"
        else:
            msg = f"**Question:** {question}\n\n**Answer:** {answer}{disclaimer}"

        await interaction.followup.send(msg, ephemeral=True)

    except Exception as e:
        print("[ERROR]", e)
        await interaction.followup.send("⚠️ Sorry, I was unable to process your question.", ephemeral=True)

bot.run(DISCORD_TOKEN)
