import os, time, logging
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

REGRA PRINCIPAL — NUNCA QUEBRE ESTA REGRA:
Responda EXCLUSIVAMENTE com base nos estudos do Eli Oliveira fornecidos abaixo.
Não use conhecimento geral da Bíblia. Não invente. Não complete com o que você sabe.
Se os estudos não cobrirem o tema, diga: "Não encontrei um estudo do Eli sobre esse tema."

COMO RESPONDER:
- Curto e direto
- Use as palavras do próprio Eli quando possível
- Sem formatação excessiva (sem bullet points, sem negrito) — conversa natural
- Português do Brasil
- Humano e acolhedor, sem religiosidade vazia
- Quando o usuário responder com um número (ex: "1", "2"), entenda como escolha de um item da lista que você acabou de apresentar e aprofunde naquele tema

CONTEXTO DE CONVERSA:
Você tem acesso ao histórico da conversa. Use-o para entender mensagens curtas como "1", "esse tema", "fale mais", "o que mais tem".

SOBRE ELI OLIVEIRA:
Homem comum, cristão, compositor e guitarrista do Marçal Talks com Pablo Marçal. Não é pastor. Esposo de Michele Monique há 18 anos, pai de Davih e Anna Rebecah. Liberto do alcoolismo há 6 anos após olhar nos olhos da filha Anna. Abusado aos 9 anos. Não segue denominação religiosa — segue a igreja que Jesus ensinou: servir pobres, órfãos, viúvas e necessitados."""

def gerar_embedding(texto: str) -> list[float]:
    resp = openai_client.embeddings.create(model="text-embedding-ada-002", input=texto[:8000])
    return resp.data[0].embedding

def buscar_por_embedding(embedding: list, quantidade: int = 20) -> list[dict]:
    resp = supabase.rpc("match_documents", {"query_embedding": embedding, "match_count": quantidade}).execute()
    return resp.data or []

def buscar_por_texto(texto: str) -> list[dict]:
    """Busca textual como fallback quando embedding não retorna resultados bons."""
    palavras = [p for p in texto.lower().split() if len(p) > 3][:5]
    query = " | ".join(palavras)
    resp = supabase.table("documents") \
        .select("id, content, metadata, similarity:id") \
        .ilike("content", f"%{palavras[0]}%") \
        .order("id", desc=True) \
        .limit(10) \
        .execute()
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

def buscar_estudos_inteligente(pergunta: str, hist: list) -> list[dict]:
    """Busca combinada: embedding + fallback textual."""
    # Reconstrói contexto da pergunta com histórico recente
    contexto_pergunta = pergunta
    if hist:
        ultimas = hist[-4:]
        contexto_pergunta = " ".join([m["content"] for m in ultimas]) + " " + pergunta

    embedding = gerar_embedding(contexto_pergunta)
    docs = buscar_por_embedding(embedding, 20)

    # Se retornou poucos resultados com similaridade baixa, faz busca textual também
    if len(docs) < 3:
        docs_texto = buscar_por_texto(pergunta)
        ids_existentes = {d.get("id") for d in docs}
        for d in docs_texto:
            if d.get("id") not in ids_existentes:
                docs.append(d)

    return docs[:20]

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
    msgs.extend(hist[-10:])
    msgs.append({
        "role": "user",
        "content": (
            f"ESTUDOS DO ELI NA BASE:\n\n{contexto}\n\n"
            f"---\n"
            f"Responda usando APENAS o conteúdo dos estudos acima. Não use conhecimento externo.\n\n"
            f"PERGUNTA/MENSAGEM: {pergunta}"
        )
    })
    resp = openai_client.chat.completions.create(
        model="gpt-4o-mini", messages=msgs, temperature=0.15, max_tokens=600
    )
    return resp.choices[0].message.content

def salvar_log(user_id, username, first_name, pergunta, resposta, docs_n, tempo_ms):
    try:
        supabase.table("bot_logs").insert({
            "user_id": user_id, "username": username, "first_name": first_name,
            "pergunta": pergunta[:1000], "resposta": resposta[:2000],
            "docs_encontrados": docs_n, "tempo_resposta_ms": tempo_ms,
        }).execute()
    except Exception as e:
        log.error(f"Log error: {e}")

async def cmd_start(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    nome = update.effective_user.first_name or "amigo"
    await update.message.reply_text(
        f"Olá, {nome}! 🙏\n\n"
        "Aqui você acessa os estudos bíblicos do Eli Oliveira.\n\n"
        "Pergunte sobre qualquer tema dos estudos, sobre o testemunho do Eli ou o que está vivendo.\n\n"
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
        hoje_count      = sum(1 for d in dados if (d.get("criado_em") or "").startswith(hoje))
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
            f"🏆 *Top usuários:*\n{top5_txt}\n\n📆 *Últimos 7 dias:*\n{dias_txt}",
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
        p = pergunta.lower()

        # Pergunta sobre estudo de hoje
        if any(x in p for x in ["hoje", "estudo de hoje", "registrado hoje", "último estudo", "mais recente"]):
            docs = buscar_estudo_hoje()
            if not docs:
                # Tenta pegar o mais recente do banco
                resp = supabase.table("documents").select("id, content, metadata, criado_date") \
                    .order("id", desc=True).limit(3).execute()
                docs = resp.data or []
                if docs:
                    data = (docs[0].get("criado_date") or "data desconhecida")
                    contexto = f"[Estudo mais recente — data: {data}]\n" + montar_contexto(docs)
                    resposta = gerar_resposta(pergunta, contexto, historico[uid])
                else:
                    resposta = "Não há estudos registrados no banco ainda."
            else:
                resposta = gerar_resposta(pergunta, montar_contexto(docs), historico[uid])

        # Resposta numérica — continuação de lista
        elif pergunta.strip() in ["1","2","3","4","5"] and historico[uid]:
            docs     = buscar_estudos_inteligente(pergunta, historico[uid])
            contexto = montar_contexto(docs)
            resposta = gerar_resposta(f"O usuário escolheu a opção {pergunta} da lista anterior. Aprofunde nesse tema.", contexto, historico[uid])

        # Busca normal
        else:
            docs     = buscar_estudos_inteligente(pergunta, historico[uid])
            contexto = montar_contexto(docs)
            resposta = gerar_resposta(pergunta, contexto, historico[uid])

        historico[uid].append({"role": "user",      "content": pergunta})
        historico[uid].append({"role": "assistant",  "content": resposta})
        historico[uid] = historico[uid][-12:]

        salvar_log(uid, username, first_name, pergunta, resposta, len(docs) if 'docs' in dir() else 0, int((time.time()-inicio)*1000))

        for i in range(0, len(resposta), 4000):
            await update.message.reply_text(resposta[i:i+4000])

    except Exception as e:
        log.error(f"Erro [{uid}]: {e}")
        await update.message.reply_text("Ocorreu um erro. Tente novamente.")

def main():
    log.info("Iniciando Palavra Viva Bot v5...")
    app = ApplicationBuilder().token(TELEGRAM_TOKEN).build()
    app.add_handler(CommandHandler("start", cmd_start))
    app.add_handler(CommandHandler("novo",  cmd_novo))
    app.add_handler(CommandHandler("stats", cmd_stats))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, responder))
    log.info("Bot rodando!")
    app.run_polling(drop_pending_updates=True)

if __name__ == "__main__":
    main()
