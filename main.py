import os
import time
import logging
from datetime import datetime, timezone
from collections import Counter
from telegram import Update
from telegram.ext import ApplicationBuilder, MessageHandler, CommandHandler, filters, ContextTypes
from openai import OpenAI
from supabase import create_client

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

SYSTEM_PROMPT = """Você é o assistente bíblico pessoal do Eli Oliveira.

━━━━━━━━━━━━━━━━━━━━━━
REGRA PRINCIPAL — MUITO IMPORTANTE
━━━━━━━━━━━━━━━━━━━━━━
Você deve responder EXCLUSIVAMENTE com base nos estudos do Eli Oliveira que estão na base de dados.
NÃO use seu conhecimento geral da Bíblia para responder.
NÃO invente, complete ou amplie além do que está escrito nos estudos.
Se o estudo do Eli fala sobre determinado ponto, reproduza fielmente o que ele disse.
Se a base não tiver nada sobre o tema perguntado, diga claramente: "Não encontrei um estudo do Eli sobre esse tema ainda."

━━━━━━━━━━━━━━━━━━━━━━
COMO RESPONDER
━━━━━━━━━━━━━━━━━━━━━━
- Respostas curtas e diretas
- Use as palavras e expressões do próprio Eli quando possível
- Cite os versículos exatamente como aparecem nos estudos dele
- Não use bullet points, negrito ou formatação excessiva — responda como numa conversa
- Fale em português do Brasil
- Seja humano e acolhedor, sem religiosidade vazia
- Quando perguntarem sobre o testemunho do Eli, conte com as palavras dele, não com as suas

━━━━━━━━━━━━━━━━━━━━━━
SOBRE ELI OLIVEIRA
━━━━━━━━━━━━━━━━━━━━━━
Eli Oliveira é um homem comum, cristão, compositor e guitarrista do Marçal Talks com Pablo Marçal. Não é pastor. Esposo de Michele Monique há 18 anos, pai de Davih e Anna Rebecah. Nasceu e cresceu na igreja mas viveu anos preso na religiosidade. Foi dependente de álcool por 7 anos. Aos 9 anos foi abusado sexualmente. Em fevereiro de 2020, ao olhar nos olhos da filha Anna (1 ano), teve uma experiência sobrenatural que o libertou do alcoolismo — há mais de 6 anos sem beber. Faz parte da igreja que Jesus ensinou: servir aos pobres, órfãos, viúvas e necessitados. Sem denominação religiosa.

━━━━━━━━━━━━━━━━━━━━━━
QUANDO PERGUNTAREM "O QUE O ESTUDO DIZ"
━━━━━━━━━━━━━━━━━━━━━━
Reproduza fielmente o conteúdo dos estudos encontrados na base.
Não interprete além do que está escrito.
Se houver trechos diretos do Eli nos estudos, use-os.
"""

def gerar_embedding(texto: str) -> list[float]:
    resp = openai_client.embeddings.create(model="text-embedding-ada-002", input=texto[:8000])
    return resp.data[0].embedding

def buscar_estudos(embedding: list[float], quantidade: int = 20) -> list[dict]:
    resp = supabase.rpc("match_documents", {"query_embedding": embedding, "match_count": quantidade}).execute()
    return resp.data or []

def buscar_estudo_hoje() -> list[dict]:
    hoje = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    resp = supabase.table("documents") \
        .select("id, content, metadata, criado_date") \
        .gte("criado_date", hoje) \
        .order("id", desc=True) \
        .limit(5) \
        .execute()
    return resp.data or []

def montar_contexto(docs: list[dict]) -> str:
    if not docs:
        return "Nenhum estudo encontrado na base para essa pergunta."
    partes = []
    for i, doc in enumerate(docs, 1):
        meta   = doc.get("metadata") or {}
        titulo = meta.get("titulo") or meta.get("source") or f"Estudo {i}"
        texto  = (doc.get("content") or "")[:600]
        partes.append(f"[Estudo {i} — {titulo}]\n{texto}")
    return "\n\n---\n\n".join(partes)

def gerar_resposta(pergunta: str, contexto: str, hist: list) -> str:
    msgs = [{"role": "system", "content": SYSTEM_PROMPT}]
    msgs.extend(hist[-8:])
    msgs.append({
        "role": "user",
        "content": (
            f"ESTUDOS DO ELI ENCONTRADOS NA BASE:\n\n{contexto}\n\n"
            f"---\n"
            f"INSTRUÇÃO: Responda a pergunta abaixo usando APENAS o conteúdo dos estudos acima. "
            f"Não use conhecimento externo.\n\n"
            f"PERGUNTA: {pergunta}"
        )
    })
    resp = openai_client.chat.completions.create(
        model="gpt-4o-mini", messages=msgs, temperature=0.2, max_tokens=600
    )
    return resp.choices[0].message.content

def salvar_log(user_id, username, first_name, pergunta, resposta, docs_encontrados, tempo_ms):
    try:
        supabase.table("bot_logs").insert({
            "user_id": user_id, "username": username, "first_name": first_name,
            "pergunta": pergunta[:1000], "resposta": resposta[:2000],
            "docs_encontrados": docs_encontrados, "tempo_resposta_ms": tempo_ms,
        }).execute()
    except Exception as e:
        log.error(f"Erro ao salvar log: {e}")

# ── Handlers ──────────────────────────────────────────────────────────────

async def cmd_start(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    nome = update.effective_user.first_name or "amigo"
    await update.message.reply_text(
        f"Olá, {nome}! 🙏\n\n"
        "Aqui você acessa os estudos bíblicos do Eli Oliveira.\n\n"
        "Pode perguntar sobre qualquer tema dos estudos, sobre o testemunho do Eli, "
        "ou o que está vivendo. Estou aqui para ajudar.\n\n"
        "Qual é a sua pergunta?"
    )

async def cmd_novo(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    historico[update.effective_user.id] = []
    await update.message.reply_text("Conversa reiniciada! Pode perguntar.")

async def cmd_stats(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    uid = update.effective_user.id
    if ADMIN_ID != 0 and uid != ADMIN_ID:
        return
    try:
        logs  = supabase.table("bot_logs").select("user_id, first_name, username, criado_em").execute()
        dados = logs.data or []
        total = len(dados)
        hoje  = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        hoje_count     = sum(1 for d in dados if (d.get("criado_em") or "").startswith(hoje))
        usuarios_unicos = len(set(d["user_id"] for d in dados))
        contagem = {}
        for d in dados:
            nome = d.get("first_name") or d.get("username") or str(d["user_id"])
            contagem[nome] = contagem.get(nome, 0) + 1
        top5     = sorted(contagem.items(), key=lambda x: x[1], reverse=True)[:5]
        top5_txt = "\n".join([f"  {i+1}. {n}: {c}" for i, (n, c) in enumerate(top5)])
        dias     = Counter((d.get("criado_em") or "")[:10] for d in dados if d.get("criado_em"))
        dias_txt = "\n".join([f"  {k}: {v}" for k, v in sorted(dias.items())[-7:]])
        await update.message.reply_text(
            f"📊 *Estatísticas*\n\n"
            f"📨 Total: *{total}*\n👥 Únicos: *{usuarios_unicos}*\n📅 Hoje: *{hoje_count}*\n\n"
            f"🏆 *Top usuários:*\n{top5_txt}\n\n"
            f"📆 *Últimos 7 dias:*\n{dias_txt}",
            parse_mode="Markdown"
        )
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
        # Detecta pergunta sobre estudo de hoje
        p = pergunta.lower()
        if any(x in p for x in ["hoje", "registrado hoje", "estudo de hoje", "último estudo"]):
            docs = buscar_estudo_hoje()
            if not docs:
                await update.message.reply_text("Não encontrei estudos registrados hoje ainda.")
                return
            contexto = montar_contexto(docs)
        else:
            embedding = gerar_embedding(pergunta)
            docs      = buscar_estudos(embedding, 20)
            contexto  = montar_contexto(docs)

        resposta = gerar_resposta(pergunta, contexto, historico[uid])

        historico[uid].append({"role": "user",      "content": pergunta})
        historico[uid].append({"role": "assistant",  "content": resposta})
        historico[uid] = historico[uid][-10:]

        salvar_log(uid, username, first_name, pergunta, resposta, len(docs), int((time.time()-inicio)*1000))

        for i in range(0, len(resposta), 4000):
            await update.message.reply_text(resposta[i:i+4000])

    except Exception as e:
        log.error(f"Erro [{uid}]: {e}")
        await update.message.reply_text("Ocorreu um erro. Tente novamente em instantes.")

def main():
    log.info("Iniciando Palavra Viva Bot v4...")
    app = ApplicationBuilder().token(TELEGRAM_TOKEN).build()
    app.add_handler(CommandHandler("start", cmd_start))
    app.add_handler(CommandHandler("novo",  cmd_novo))
    app.add_handler(CommandHandler("stats", cmd_stats))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, responder))
    log.info("Bot rodando!")
    app.run_polling(drop_pending_updates=True)

if __name__ == "__main__":
    main()
