import os
import time
import logging
from datetime import datetime, timezone
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

SYSTEM_PROMPT = """Você é o assistente bíblico pessoal do Eli Oliveira, um homem comum transformado por Cristo.

━━━━━━━━━━━━━━━━━━━━━━
SOBRE ELI OLIVEIRA
━━━━━━━━━━━━━━━━━━━━━━
Eli Oliveira é um homem comum, cristão, compositor e guitarrista — inclusive do Marçal Talks com Pablo Marçal. Não é pastor, não tem título religioso. É esposo de Michele Monique há mais de 18 anos, pai de Davih e Anna Rebecah.

Nasceu e cresceu dentro da igreja, mas viveu anos preso na religiosidade sem vida real com Deus. Foi músico e líder de departamento na Assembleia de Deus Ministério Perus, em Osasco/SP, mas vivia uma dupla vida: cumpria agendas da igreja e ao mesmo tempo era dependente de álcool por cerca de 7 anos.

Aos 9 anos foi abusado sexualmente durante a construção de uma igreja, trauma que trouxe distorções comportamentais e emocionais por décadas.

━━━━━━━━━━━━━━━━━━━━━━
TESTEMUNHO DE LIBERTAÇÃO
━━━━━━━━━━━━━━━━━━━━━━
Em fevereiro de 2020, Eli saía do trabalho às 16h30 e chegava em casa à meia-noite, passando horas nos bares bebendo vodka com Coca-Cola. Usava sertralina e Rivotril. Orava pedindo a Deus para tirar a vontade de beber, mas não conseguia vencer.

Uma noite chegou em casa alcoolizado. Sua esposa Michele foi tomar banho e pediu que, se a bebê Anna chorasse, ele a pegasse. Quando Anna chorou, Eli a pegou no colo. Ao olhar nos olhos da filha — que tinha apenas 1 ano — viu um semblante sobrenatural, como de uma mulher adulta, com olhar profundo. Sentiu como se ela dissesse: "Eu vim ao mundo para ter um pai alcoólatra?"

Aquilo o quebrou por dentro. Chorou, suas pernas ficaram bambas, foi tomado por vergonha e tristeza. Daquele dia em diante, há mais de 6 anos, não coloca álcool na boca.

Eli entende que os olhos da filha foram a "sarça ardente" — o Cristo usando Anna para transmitir aquela mensagem. A libertação veio também pelas orações e paciência de Michele. O nome Rebeca significa "aquela que une" — ela o uniu ao Senhor.

━━━━━━━━━━━━━━━━━━━━━━
LINHA TEOLÓGICA
━━━━━━━━━━━━━━━━━━━━━━
Eli não segue denominação religiosa. Segue a igreja que Jesus congregou e ensinou: servir aos pobres, órfãos, viúvas e necessitados. Acredita na transformação real pela presença do Espírito Santo, não em religiosidade de fachada.

━━━━━━━━━━━━━━━━━━━━━━
COMO VOCÊ DEVE RESPONDER
━━━━━━━━━━━━━━━━━━━━━━
- Respostas CURTAS e DIRETAS — vá ao ponto
- Use os estudos da base como referência principal
- Cite versículos quando relevante, mas com naturalidade
- Seja acolhedor, humano e sem religiosidade vazia
- Se perguntarem sobre Eli, seu testemunho ou sua história, responda com base nas informações acima e na base de estudos
- Nunca invente informações sobre Eli que não estejam na base
- Fale sempre em português do Brasil
- Quando a pergunta for pessoal (solidão, vício, família, dor), conecte com o testemunho de Eli quando fizer sentido
- Não use linguagem religiosa vazia ("que Deus abençoe", "glória a Deus" a todo momento)
"""

def gerar_embedding(texto: str) -> list[float]:
    resp = openai_client.embeddings.create(model="text-embedding-ada-002", input=texto[:8000])
    return resp.data[0].embedding

def buscar_estudos(embedding: list[float], quantidade: int = 20) -> list[dict]:
    resp = supabase.rpc("match_documents", {"query_embedding": embedding, "match_count": quantidade}).execute()
    return resp.data or []

def montar_contexto(docs: list[dict]) -> str:
    partes = []
    for i, doc in enumerate(docs, 1):
        meta  = doc.get("metadata") or {}
        titulo = meta.get("titulo") or meta.get("source") or f"Estudo {i}"
        texto  = (doc.get("content") or "")[:500]
        partes.append(f"[{i}] {titulo}\n{texto}")
    return "\n\n---\n\n".join(partes)

def gerar_resposta(pergunta: str, contexto: str, hist: list) -> str:
    msgs = [{"role": "system", "content": SYSTEM_PROMPT}]
    msgs.extend(hist[-8:])
    msgs.append({"role": "user", "content": f"ESTUDOS RELEVANTES DA BASE:\n{contexto}\n\nPERGUNTA: {pergunta}"})
    resp = openai_client.chat.completions.create(
        model="gpt-4o-mini", messages=msgs, temperature=0.4, max_tokens=800
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

async def cmd_start(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    nome = update.effective_user.first_name or "amigo"
    await update.message.reply_text(
        f"Olá, {nome}! 🙏\n\n"
        "Aqui você encontra os estudos bíblicos do Eli Oliveira.\n\n"
        "Pode perguntar sobre qualquer tema da Bíblia, sobre o testemunho do Eli, "
        "ou compartilhar o que está vivendo. Estou aqui para ajudar.\n\n"
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
        logs = supabase.table("bot_logs").select("user_id, first_name, username, criado_em").execute()
        dados = logs.data or []
        total = len(dados)
        hoje = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        hoje_count = sum(1 for d in dados if (d.get("criado_em") or "").startswith(hoje))
        usuarios_unicos = len(set(d["user_id"] for d in dados))
        contagem = {}
        for d in dados:
            nome = d.get("first_name") or d.get("username") or str(d["user_id"])
            contagem[nome] = contagem.get(nome, 0) + 1
        top5 = sorted(contagem.items(), key=lambda x: x[1], reverse=True)[:5]
        top5_txt = "\n".join([f"  {i+1}. {n}: {c}" for i, (n, c) in enumerate(top5)])
        from collections import Counter
        dias = Counter((d.get("criado_em") or "")[:10] for d in dados if d.get("criado_em"))
        dias_txt = "\n".join([f"  {k}: {v}" for k, v in sorted(dias.items())[-7:]])
        await update.message.reply_text(
            f"📊 *Estatísticas*\n\n"
            f"📨 Total: *{total}*\n"
            f"👥 Únicos: *{usuarios_unicos}*\n"
            f"📅 Hoje: *{hoje_count}*\n\n"
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
        embedding = gerar_embedding(pergunta)
        docs      = buscar_estudos(embedding, 20)
        contexto  = montar_contexto(docs)
        resposta  = gerar_resposta(pergunta, contexto, historico[uid])

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
    log.info("Iniciando Palavra Viva Bot...")
    app = ApplicationBuilder().token(TELEGRAM_TOKEN).build()
    app.add_handler(CommandHandler("start", cmd_start))
    app.add_handler(CommandHandler("novo",  cmd_novo))
    app.add_handler(CommandHandler("stats", cmd_stats))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, responder))
    log.info("Bot rodando!")
    app.run_polling(drop_pending_updates=True)

if __name__ == "__main__":
    main()
