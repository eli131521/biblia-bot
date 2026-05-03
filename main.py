import os, time, logging, tempfile
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
REGRA PRINCIPAL
━━━━━━━━━━━━━━━━━━━━━━
Responda SEMPRE com base nos estudos do Eli Oliveira fornecidos na base de dados.
Não invente. Não complete com conhecimento geral além do que está nos estudos.
Se a base não tiver o tema, diga: "Não encontrei um estudo do Eli sobre isso ainda."

━━━━━━━━━━━━━━━━━━━━━━
COMO RESPONDER — MUITO IMPORTANTE
━━━━━━━━━━━━━━━━━━━━━━
Suas respostas devem ser RICAS e COMPLETAS. Para cada pergunta:

1. EXPLIQUE o tema com base no estudo do Eli — use as palavras e expressões dele
2. TRAGA EXEMPLOS BÍBLICOS presentes no estudo (versículos, personagens, histórias)
3. FAÇA A APLICAÇÃO PARA HOJE — como esse ensinamento se aplica à vida atual, à realidade das pessoas
4. QUANDO HOUVER RELAÇÃO com o testemunho do Eli (álcool, vício, religiosidade, libertação, família, paternidade), conecte naturalmente — sem forçar
5. TERMINE com uma reflexão ou pergunta que convide o usuário a pensar

Formato da resposta:
- Parágrafos corridos, sem bullet points ou negrito excessivo
- Tom pastoral, humano, acolhedor — como uma conversa real
- Português do Brasil
- Tamanho: entre 150 e 400 palavras por resposta

━━━━━━━━━━━━━━━━━━━━━━
CONTEXTO DE CONVERSA
━━━━━━━━━━━━━━━━━━━━━━
Você tem o histórico completo. Use-o para entender:
- Mensagens curtas como "1", "2", "3" = escolha de item de lista anterior
- "fale mais", "continue", "aprofunde" = continuar o tema atual
- "esse tema", "sobre isso" = referência ao assunto anterior

━━━━━━━━━━━━━━━━━━━━━━
SOBRE ELI OLIVEIRA — USE QUANDO RELEVANTE
━━━━━━━━━━━━━━━━━━━━━━
Eli é um homem comum, cristão, compositor e guitarrista do Marçal Talks com Pablo Marçal. Não é pastor. Esposo de Michele Monique há 18 anos, pai de Davih e Anna Rebecah.

Testemunho principal: Foi dependente de álcool por 7 anos. Aos 9 anos sofreu abuso sexual durante a construção de uma igreja. Vivia na religiosidade — músico e líder de departamento na igreja — mas chegava alcoolizado aos ensaios. Em fevereiro de 2020, chegou em casa alcoolizado. Sua filha Anna, de apenas 1 ano, o olhou com um semblante sobrenatural — como se dissesse "vim ao mundo para ter um pai alcoólatra?". Suas pernas bambaram, começou a chorar. Daquele dia, há mais de 6 anos, não bebe. Entende que os olhos da filha foram a "sarça ardente" — Cristo usando Anna como instrumento. A libertação veio também pelas orações e paciência de Michele. Rebeca significa "aquela que une" — ela o uniu ao Senhor.

Outros temas do testemunho: pornografia, religiosidade vazia, dupla vida, cura de traumas, domínio próprio como fruto do Espírito Santo, pai presente, casamento restaurado.

Não segue denominação. Segue a igreja que Jesus ensinou: servir pobres, órfãos, viúvas e necessitados.

━━━━━━━━━━━━━━━━━━━━━━
QUANDO PROCESSAR ÁUDIO OU IMAGEM
━━━━━━━━━━━━━━━━━━━━━━
O conteúdo transcrito ou descrito será enviado junto com a pergunta.
Trate como se fosse texto normal — busque na base e responda com a mesma riqueza."""

# ── Banco e embeddings ────────────────────────────────────────────────────

def gerar_embedding(texto: str) -> list[float]:
    resp = openai_client.embeddings.create(model="text-embedding-ada-002", input=texto[:8000])
    return resp.data[0].embedding

def buscar_por_embedding(embedding: list, quantidade: int = 20) -> list[dict]:
    resp = supabase.rpc("match_documents", {"query_embedding": embedding, "match_count": quantidade}).execute()
    return resp.data or []

def buscar_por_texto(texto: str) -> list[dict]:
    palavras = [p for p in texto.lower().split() if len(p) > 3]
    if not palavras:
        return []
    resp = supabase.table("documents") \
        .select("id, content, metadata") \
        .ilike("content", f"%{palavras[0]}%") \
        .order("id", desc=True).limit(10).execute()
    return resp.data or []

def buscar_estudo_hoje() -> list[dict]:
    hoje = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    resp = supabase.table("documents") \
        .select("id, content, metadata, criado_date") \
        .gte("criado_date", hoje) \
        .order("id", desc=True).limit(5).execute()
    return resp.data or []

def buscar_estudos_inteligente(pergunta: str, hist: list) -> list[dict]:
    contexto_pergunta = " ".join([m["content"] for m in hist[-4:]]) + " " + pergunta
    embedding = gerar_embedding(contexto_pergunta)
    docs = buscar_por_embedding(embedding, 20)
    if len(docs) < 3:
        extras = buscar_por_texto(pergunta)
        ids = {d.get("id") for d in docs}
        for d in extras:
            if d.get("id") not in ids:
                docs.append(d)
    return docs[:20]

def montar_contexto(docs: list[dict]) -> str:
    if not docs:
        return "Nenhum estudo encontrado na base para essa pergunta."
    partes = []
    for i, doc in enumerate(docs, 1):
        meta   = doc.get("metadata") or {}
        titulo = meta.get("titulo") or meta.get("source") or f"Estudo {i}"
        texto  = (doc.get("content") or "")[:700]
        partes.append(f"[Estudo {i} — {titulo}]\n{texto}")
    return "\n\n---\n\n".join(partes)

def gerar_resposta(pergunta: str, contexto: str, hist: list) -> str:
    msgs = [{"role": "system", "content": SYSTEM_PROMPT}]
    msgs.extend(hist[-10:])
    msgs.append({
        "role": "user",
        "content": (
            f"ESTUDOS DO ELI NA BASE:\n\n{contexto}\n\n---\n\n"
            f"Responda com base nos estudos acima. Seja rico, detalhado, com exemplos bíblicos "
            f"e aplicação para o dia de hoje. Quando relevante, conecte com o testemunho do Eli.\n\n"
            f"PERGUNTA: {pergunta}"
        )
    })
    resp = openai_client.chat.completions.create(
        model="gpt-4o-mini", messages=msgs, temperature=0.3, max_tokens=900
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

# ── Transcrição de áudio ──────────────────────────────────────────────────

async def transcrever_audio(file_bytes: bytes, filename: str) -> str:
    with tempfile.NamedTemporaryFile(suffix=".ogg", delete=False) as tmp:
        tmp.write(file_bytes)
        tmp_path = tmp.name
    with open(tmp_path, "rb") as f:
        resp = openai_client.audio.transcriptions.create(
            model="whisper-1", file=(filename, f, "audio/ogg")
        )
    os.unlink(tmp_path)
    return resp.text

# ── Descrição de imagem ───────────────────────────────────────────────────

async def descrever_imagem(file_bytes: bytes) -> str:
    import base64
    b64 = base64.b64encode(file_bytes).decode("utf-8")
    resp = openai_client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
                {"type": "text", "text": "Descreva o que está nessa imagem em detalhes. Se houver texto bíblico ou referências religiosas, transcreva-os."}
            ]
        }],
        max_tokens=500
    )
    return resp.choices[0].message.content

# ── Handlers ──────────────────────────────────────────────────────────────

async def cmd_start(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    nome = update.effective_user.first_name or "amigo"
    await update.message.reply_text(
        f"Olá, {nome}! 🙏\n\n"
        "Aqui você acessa os estudos bíblicos do Eli Oliveira.\n\n"
        "Pode perguntar sobre qualquer tema, mandar áudio ou imagem — estou aqui para ajudar.\n\n"
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

async def processar_pergunta(uid: int, username: str, first_name: str, pergunta: str, update: Update):
    """Processa qualquer pergunta (texto, áudio transcrito ou imagem descrita)."""
    if uid not in historico:
        historico[uid] = []

    inicio = time.time()
    p = pergunta.lower()

    try:
        if any(x in p for x in ["hoje", "estudo de hoje", "registrado hoje", "último estudo", "mais recente"]):
            docs = buscar_estudo_hoje()
            if not docs:
                resp_sb = supabase.table("documents").select("id, content, metadata, criado_date") \
                    .order("id", desc=True).limit(3).execute()
                docs = resp_sb.data or []
            contexto = montar_contexto(docs)
        elif pergunta.strip() in ["1","2","3","4","5"] and historico[uid]:
            docs     = buscar_estudos_inteligente(pergunta, historico[uid])
            contexto = montar_contexto(docs)
            pergunta = f"O usuário escolheu a opção {pergunta} da lista anterior. Aprofunde com exemplos bíblicos e aplicação."
        else:
            docs     = buscar_estudos_inteligente(pergunta, historico[uid])
            contexto = montar_contexto(docs)

        resposta = gerar_resposta(pergunta, contexto, historico[uid])

        historico[uid].append({"role": "user",      "content": pergunta})
        historico[uid].append({"role": "assistant",  "content": resposta})
        historico[uid] = historico[uid][-12:]

        salvar_log(uid, username, first_name, pergunta, resposta, len(docs), int((time.time()-inicio)*1000))

        for i in range(0, len(resposta), 4000):
            await update.message.reply_text(resposta[i:i+4000])

    except Exception as e:
        log.error(f"Erro [{uid}]: {e}")
        await update.message.reply_text("Ocorreu um erro. Tente novamente.")

async def responder_texto(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    uid        = update.effective_user.id
    username   = update.effective_user.username or ""
    first_name = update.effective_user.first_name or ""
    pergunta   = update.message.text.strip()
    if not pergunta:
        return
    await ctx.bot.send_chat_action(chat_id=update.effective_chat.id, action="typing")
    await processar_pergunta(uid, username, first_name, pergunta, update)

async def responder_audio(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    uid        = update.effective_user.id
    username   = update.effective_user.username or ""
    first_name = update.effective_user.first_name or ""

    await ctx.bot.send_chat_action(chat_id=update.effective_chat.id, action="typing")
    await update.message.reply_text("🎤 Transcrevendo seu áudio...")

    try:
        audio = update.message.voice or update.message.audio
        file  = await ctx.bot.get_file(audio.file_id)
        file_bytes = await file.download_as_bytearray()
        transcricao = await transcrever_audio(bytes(file_bytes), "audio.ogg")
        await update.message.reply_text(f"📝 Transcrição: _{transcricao}_", parse_mode="Markdown")
        await processar_pergunta(uid, username, first_name, transcricao, update)
    except Exception as e:
        log.error(f"Erro áudio [{uid}]: {e}")
        await update.message.reply_text("Não consegui processar o áudio. Tente enviar em texto.")

async def responder_imagem(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    uid        = update.effective_user.id
    username   = update.effective_user.username or ""
    first_name = update.effective_user.first_name or ""

    await ctx.bot.send_chat_action(chat_id=update.effective_chat.id, action="typing")
    await update.message.reply_text("🖼️ Analisando a imagem...")

    try:
        photo = update.message.photo[-1]  # maior resolução
        file  = await ctx.bot.get_file(photo.file_id)
        file_bytes = await file.download_as_bytearray()
        descricao = await descrever_imagem(bytes(file_bytes))

        caption = update.message.caption or ""
        pergunta = f"[Imagem enviada — descrição: {descricao}]. {caption}".strip()

        await update.message.reply_text(f"🔍 Entendi a imagem. Buscando nos estudos...")
        await processar_pergunta(uid, username, first_name, pergunta, update)
    except Exception as e:
        log.error(f"Erro imagem [{uid}]: {e}")
        await update.message.reply_text("Não consegui processar a imagem. Tente descrever em texto.")

# ── Main ──────────────────────────────────────────────────────────────────

def main():
    log.info("Iniciando Palavra Viva Bot v6...")
    app = ApplicationBuilder().token(TELEGRAM_TOKEN).build()
    app.add_handler(CommandHandler("start", cmd_start))
    app.add_handler(CommandHandler("novo",  cmd_novo))
    app.add_handler(CommandHandler("stats", cmd_stats))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND,          responder_texto))
    app.add_handler(MessageHandler(filters.VOICE | filters.AUDIO,            responder_audio))
    app.add_handler(MessageHandler(filters.PHOTO,                             responder_imagem))
    log.info("Bot v6 rodando! Texto, áudio e imagem habilitados.")
    app.run_polling(drop_pending_updates=True)

if __name__ == "__main__":
    main()
