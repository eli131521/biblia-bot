import os
import logging
from telegram import Update
from telegram.ext import ApplicationBuilder, MessageHandler, CommandHandler, filters, ContextTypes
from openai import OpenAI
from supabase import create_client
import httpx

# ── Configuração ──────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

TELEGRAM_TOKEN = os.environ["TELEGRAM_TOKEN"]
OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]
SUPABASE_URL   = os.environ["SUPABASE_URL"]
SUPABASE_KEY   = os.environ["SUPABASE_KEY"]

openai_client  = OpenAI(api_key=OPENAI_API_KEY)
supabase       = create_client(SUPABASE_URL, SUPABASE_KEY)

# Histórico por usuário (em memória)
historico: dict[int, list] = {}

# ── Funções principais ────────────────────────────────────────────────────

def gerar_embedding(texto: str) -> list[float]:
    """Gera embedding com OpenAI ada-002."""
    resp = openai_client.embeddings.create(
        model="text-embedding-ada-002",
        input=texto[:8000]
    )
    return resp.data[0].embedding


def buscar_estudos(embedding: list[float], quantidade: int = 20) -> list[dict]:
    """Busca os estudos mais relevantes no Supabase via match_documents."""
    resp = supabase.rpc("match_documents", {
        "query_embedding": embedding,
        "match_count": quantidade
    }).execute()
    return resp.data or []


def montar_contexto(docs: list[dict]) -> str:
    """Monta o contexto com os estudos encontrados."""
    partes = []
    for i, doc in enumerate(docs, 1):
        titulo = (doc.get("metadata") or {}).get("titulo") or \
                 (doc.get("metadata") or {}).get("source") or f"Estudo {i}"
        conteudo = (doc.get("content") or "")[:600]
        partes.append(f"[{i}] {titulo}\n{conteudo}")
    return "\n\n---\n\n".join(partes)


def gerar_resposta(pergunta: str, contexto: str, historico_msgs: list) -> str:
    """Gera resposta pastoral com GPT-4o-mini."""
    system = """Você é um assistente especializado em estudos bíblicos do pastor Eli Oliveira.
Use os estudos fornecidos como base para responder.
Responda em português do Brasil com profundidade, clareza e fidelidade às Escrituras.
Cite versículos relevantes. Seja pastoral e acolhedor.
Ao final, mencione brevemente as referências usadas."""

    msgs = [{"role": "system", "content": system}]

    # Histórico recente
    msgs.extend(historico_msgs[-8:])

    # Pergunta atual com contexto
    msgs.append({
        "role": "user",
        "content": f"ESTUDOS DE REFERÊNCIA:\n{contexto}\n\nPERGUNTA: {pergunta}"
    })

    resp = openai_client.chat.completions.create(
        model="gpt-4o-mini",
        messages=msgs,
        temperature=0.4,
        max_tokens=1200
    )
    return resp.choices[0].message.content


# ── Handlers do Telegram ──────────────────────────────────────────────────

async def cmd_start(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    nome = update.effective_user.first_name or "amigo"
    await update.message.reply_text(
        f"✦ Olá, {nome}! Bem-vindo aos Estudos Bíblicos.\n\n"
        "Faça qualquer pergunta sobre a Bíblia e eu buscarei nos estudos "
        "do pastor Eli Oliveira para responder com profundidade.\n\n"
        "Pode digitar sua pergunta! 🙏"
    )


async def cmd_novo(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    uid = update.effective_user.id
    historico[uid] = []
    await update.message.reply_text("✦ Conversa reiniciada! Pode fazer sua nova pergunta.")


async def responder(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    uid      = update.effective_user.id
    pergunta = update.message.text.strip()

    if not pergunta:
        return

    # Inicializa histórico do usuário
    if uid not in historico:
        historico[uid] = []

    # Indicador de digitação
    await ctx.bot.send_chat_action(chat_id=update.effective_chat.id, action="typing")

    try:
        log.info(f"Usuário {uid}: {pergunta[:80]}")

        # 1. Embedding da pergunta
        embedding = gerar_embedding(pergunta)

        # 2. Busca semântica
        docs = buscar_estudos(embedding, quantidade=20)
        log.info(f"Encontrados {len(docs)} estudos relevantes")

        # 3. Monta contexto
        contexto = montar_contexto(docs)

        # 4. Gera resposta
        resposta = gerar_resposta(pergunta, contexto, historico[uid])

        # 5. Atualiza histórico
        historico[uid].append({"role": "user",      "content": pergunta})
        historico[uid].append({"role": "assistant",  "content": resposta})

        # Mantém só últimas 10 mensagens
        historico[uid] = historico[uid][-10:]

        # Divide resposta se for muito longa (limite Telegram: 4096 chars)
        if len(resposta) <= 4096:
            await update.message.reply_text(resposta)
        else:
            for i in range(0, len(resposta), 4000):
                await update.message.reply_text(resposta[i:i+4000])

    except Exception as e:
        log.error(f"Erro ao responder usuário {uid}: {e}")
        await update.message.reply_text(
            "⚠️ Ocorreu um erro ao processar sua pergunta. Tente novamente em alguns instantes."
        )


# ── Inicialização ─────────────────────────────────────────────────────────

def main():
    log.info("Iniciando bot Palavra Viva...")
    app = ApplicationBuilder().token(TELEGRAM_TOKEN).build()

    app.add_handler(CommandHandler("start", cmd_start))
    app.add_handler(CommandHandler("novo",  cmd_novo))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, responder))

    log.info("Bot rodando! Aguardando mensagens...")
    app.run_polling(drop_pending_updates=True)


if __name__ == "__main__":
    main()
