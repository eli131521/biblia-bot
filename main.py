import os
import time
import logging
from datetime import datetime, timezone
from telegram import Update
from telegram.ext import ApplicationBuilder, MessageHandler, CommandHandler, filters, ContextTypes
from openai import OpenAI
from supabase import create_client

# ── Configuração ──────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

TELEGRAM_TOKEN = os.environ["TELEGRAM_TOKEN"]
OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]
SUPABASE_URL   = os.environ["SUPABASE_URL"]
SUPABASE_KEY   = os.environ["SUPABASE_KEY"]
ADMIN_ID       = int(os.environ.get("ADMIN_ID", "0"))

openai_client = OpenAI(api_key=OPENAI_API_KEY)
supabase      = create_client(SUPABASE_URL, SUPABASE_KEY)
historico: dict[int, list] = {}

# ── Core ──────────────────────────────────────────────────────────────────

def gerar_embedding(texto: str) -> list[float]:
    resp = openai_client.embeddings.create(model="text-embedding-ada-002", input=texto[:8000])
    return resp.data[0].embedding

def buscar_estudos(embedding: list[float], quantidade: int = 20) -> list[dict]:
    resp = supabase.rpc("match_documents", {"query_embedding": embedding, "match_count": quantidade}).execute()
    return resp.data or []

def montar_contexto(docs: list[dict]) -> str:
    partes = []
    for i, doc in enumerate(docs, 1):
        titulo = (doc.get("metadata") or {}).get("titulo") or (doc.get("metadata") or {}).get("source") or f"Estudo {i}"
        partes.append(f"[{i}] {titulo}\n{(doc.get('content') or '')[:600]}")
    return "\n\n---\n\n".join(partes)

def gerar_resposta(pergunta: str, contexto: str, hist: list) -> str:
    system = """Você é um assistente especializado em estudos bíblicos do pastor Eli Oliveira.
Use os estudos fornecidos como base. Responda em português do Brasil com profundidade e fidelidade às Escrituras.
Cite versículos relevantes. Seja pastoral e acolhedor. Mencione as referências ao final."""
    msgs = [{"role": "system", "content": system}]
    msgs.extend(hist[-8:])
    msgs.append({"role": "user", "content": f"ESTUDOS:\n{contexto}\n\nPERGUNTA: {pergunta}"})
    resp = openai_client.chat.completions.create(model="gpt-4o-mini", messages=msgs, temperature=0.4, max_tokens=1200)
    return resp.choices[0].message.content

def salvar_log(user_id, username, first_name, pergunta, resposta, docs_encontrados, tempo_ms):
    try:
        supabase.table("bot_logs").insert({
            "user_id": user_id,
            "username": username,
            "first_name": first_name,
            "pergunta": pergunta[:1000],
            "resposta": resposta[:2000],
            "docs_encontrados": docs_encontrados,
            "tempo_resposta_ms": tempo_ms,
        }).execute()
    except Exception as e:
        log.error(f"Erro ao salvar log: {e}")

# ── Handlers ──────────────────────────────────────────────────────────────

async def cmd_start(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    nome = update.effective_user.first_name or "amigo"
    await update.message.reply_text(
        f"✦ Olá, {nome}! Bem-vindo aos Estudos Bíblicos.\n\n"
        "Faça qualquer pergunta sobre a Bíblia e buscarei nos estudos "
        "do pastor Eli Oliveira para responder com profundidade.\n\n"
        "Pode digitar sua pergunta! 🙏"
    )

async def cmd_novo(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    historico[update.effective_user.id] = []
    await update.message.reply_text("✦ Conversa reiniciada! Pode fazer sua nova pergunta.")

async def cmd_stats(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    uid = update.effective_user.id
    if ADMIN_ID != 0 and uid != ADMIN_ID:
        await update.message.reply_text("⚠️ Comando restrito ao administrador.")
        return
    try:
        logs = supabase.table("bot_logs").select("user_id, first_name, username, criado_em").execute()
        dados = logs.data or []

        total = len(dados)
        hoje = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        hoje_count = sum(1 for d in dados if d.get("criado_em", "").startswith(hoje))
        usuarios_unicos = len(set(d["user_id"] for d in dados))

        contagem = {}
        for d in dados:
            nome = d.get("first_name") or d.get("username") or str(d["user_id"])
            contagem[nome] = contagem.get(nome, 0) + 1
        top5 = sorted(contagem.items(), key=lambda x: x[1], reverse=True)[:5]
        top5_txt = "\n".join([f"  {i+1}. {n}: {c} msgs" for i, (n, c) in enumerate(top5)])

        # Perguntas por dia (últimos 7 dias)
        from collections import Counter
        dias = Counter(d["criado_em"][:10] for d in dados if d.get("criado_em"))
        ultimos = sorted(dias.items())[-7:]
        dias_txt = "\n".join([f"  {dia}: {qtd}" for dia, qtd in ultimos])

        msg = (
            f"📊 *Estatísticas do Bot*\n\n"
            f"📨 Total de perguntas: *{total}*\n"
            f"👥 Usuários únicos: *{usuarios_unicos}*\n"
            f"📅 Hoje: *{hoje_count}*\n\n"
            f"🏆 *Top usuários:*\n{top5_txt}\n\n"
            f"📆 *Últimos 7 dias:*\n{dias_txt}"
        )
        await update.message.reply_text(msg, parse_mode="Markdown")
    except Exception as e:
        await update.message.reply_text(f"Erro: {e}")

async def responder(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    uid        = update.effective_user.id
    username   = update.effective_user.username or ""
    first_name = update.effective_user.first_name or ""
    pergunta   = update.message.text.strip()
    if not pergunta:
        return
    if uid not in historico:
        historico[uid] = []

    await ctx.bot.send_chat_action(chat_id=update.effective_chat.id, action="typing")
    inicio = time.time()
    try:
        embedding = gerar_embedding(pergunta)
        docs      = buscar_estudos(embedding, 20)
        contexto  = montar_contexto(docs)
        resposta  = gerar_resposta(pergunta, contexto, historico[uid])

        historico[uid].append({"role": "user",     "content": pergunta})
        historico[uid].append({"role": "assistant", "content": resposta})
        historico[uid] = historico[uid][-10:]

        salvar_log(uid, username, first_name, pergunta, resposta, len(docs), int((time.time()-inicio)*1000))

        for i in range(0, len(resposta), 4000):
            await update.message.reply_text(resposta[i:i+4000])

    except Exception as e:
        log.error(f"Erro [{uid}]: {e}")
        await update.message.reply_text("⚠️ Ocorreu um erro. Tente novamente.")

# ── Main ──────────────────────────────────────────────────────────────────

def main():
    log.info("Iniciando bot Palavra Viva...")
    app = ApplicationBuilder().token(TELEGRAM_TOKEN).build()
    app.add_handler(CommandHandler("start", cmd_start))
    app.add_handler(CommandHandler("novo",  cmd_novo))
    app.add_handler(CommandHandler("stats", cmd_stats))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, responder))
    log.info("Bot rodando!")
    app.run_polling(drop_pending_updates=True)

if __name__ == "__main__":
    main()

