import streamlit as st
import pandas as pd
import random
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split

import time  # Add this

st.write("✅ App started...")


def explicacao_heuristica(msg, label):
    msg_lower = msg.lower()
    explicacoes_spam = []
    explicacoes_ham = []

    destaques = []

    # SPAM
    if any(word in msg_lower for word in ["ganha", "grátis", "oferta", "dinheiro", "prémio", "urgente"]):
        palavras = [w for w in ["ganha", "grátis", "oferta", "dinheiro", "prémio", "urgente"] if w in msg_lower]
        explicacoes_spam.append(f"🟡 Usa linguagem promocional: **{', '.join(palavras)}**.")
        destaques.extend(palavras)

    if "http" in msg_lower or "www" in msg_lower or "link" in msg_lower:
        explicacoes_spam.append("🔗 Contém um **link** suspeito (ex: phishing ou redirecionamento).")
        destaques.append("link")

    if any(char.isdigit() for char in msg_lower) and any(p in msg_lower for p in ["envie", "sms", "123", "número"]):
        explicacoes_spam.append("📲 Contém instruções para ações com números, típico de campanhas automatizadas.")
        destaques.extend(["envie", "sms", "123", "número"])

    if "conta" in msg_lower or "suspensa" in msg_lower:
        explicacoes_spam.append("🚨 Linguagem alarmista associada a tentativas de fraude.")
        destaques.extend(["conta", "suspensa"])

    # HAM
    if any(word in msg_lower for word in ["aula", "slides", "aniversário", "almoço", "viagem", "combinamos", "amanhã"]):
        palavras = [w for w in ["aula", "slides", "aniversário", "almoço", "viagem", "combinamos", "amanhã"] if w in msg_lower]
        explicacoes_ham.append(f"💬 Linguagem informal ou pessoal: **{', '.join(palavras)}**.")
        destaques.extend(palavras)

    if "vemos" in msg_lower or "levar" in msg_lower:
        explicacoes_ham.append("🧍 Refere-se a planos ou rotina diária.")
        destaques.extend(["vemos", "levar"])

    # Construção da explicação final
    if label == "spam" and explicacoes_spam:
        return "📌 Razões para SPAM:\n" + "\n".join(["- " + e for e in explicacoes_spam])
    elif label == "ham" and explicacoes_ham:
        return "📌 Razões para NÃO SPAM:\n" + "\n".join(["- " + e for e in explicacoes_ham])
    else:
        return "ℹ️ Classificação baseada apenas em padrões aprendidos pelo modelo."


try:
    df = pd.read_csv("messages.csv")
except FileNotFoundError:
    st.error("❌ O ficheiro messages.csv não foi encontrado. Garante que está na raíz do repositório.")
    st.stop()

@st.cache_resource
def train_models(data):
    vectorizer = CountVectorizer()
    X = vectorizer.fit_transform(data["message"])
    y = data["label"]

    models = {
        "Naive Bayes": MultinomialNB(),
        "SVM": LinearSVC(),
        "Logistic Regression": LogisticRegression(max_iter=1000)
    }

    for model in models.values():
        model.fit(X, y)

    return vectorizer, models

vectorizer, models = train_models(df)


tab1, tab2 = st.tabs(["🎯 Jogo: Tu vs Máquina", "📝 Testar Mensagem"])

# ----------- ABA 1: JOGO INTERATIVO -------------
with tab1:
    st.title("Tu vs A Máquina: Jogo de Classificação de Spam")
    st.markdown("Tenta classificar as mensagens como **Spam** ou **Não Spam** e vê como te comparas com o modelo de aprendizagem automática!")

    st.sidebar.title("⚙️ Configurações do Modelo")
    selected_model_name = st.sidebar.selectbox("Seleciona o modelo de ML para o jogo", list(models.keys()))
    model = models[selected_model_name]

    # Garantir que as mesmas mensagens são usadas até ao reset
    if "sample_df" not in st.session_state:
        st.session_state.sample_df = df.sample(5, random_state=random.randint(1, 10000)).reset_index(drop=True)

    sample_df = st.session_state.sample_df

    if len(sample_df) < 5:
        st.error("Erro: Foram selecionadas menos de 5 mensagens para o jogo. Verifica o conteúdo do ficheiro.")
        st.stop()

    user_guesses = []
    st.subheader("🔍 Classifica as Mensagens:")

    for i, row in sample_df.iterrows():
        st.markdown(f"**Mensagem {i+1}:** {row['message']}")
        guess = st.radio(f"É Spam?", ["ham", "spam"], key=i)
        user_guesses.append(guess)

    if st.button("Submeter as tuas respostas"):
        X_sample = vectorizer.transform(sample_df["message"])
        model_preds = model.predict(X_sample)

        user_correct = 0
        model_correct = 0

        for i in range(len(sample_df)):
            true_label = sample_df.loc[i, "label"]
            user = user_guesses[i]
            machine = model_preds[i]
            msg = sample_df.loc[i, 'message']

            st.markdown(f"**Mensagem {i+1}:** {msg}")
            st.markdown(f"- ✅ Rótulo Correto: `{true_label}`")
            st.markdown(f"- 👤 A tua resposta: `{user}` {'✅' if user == true_label else '❌'}")
            st.markdown(f"- 🤖 Previsão da Máquina: `{machine}` {'✅' if machine == true_label else '❌'}")

            explicacao = explicacao_heuristica(msg, true_label)
            st.markdown(f"- 📚 Explicação: {explicacao}")

            if user == true_label:
                user_correct += 1
            if machine == true_label:
                model_correct += 1
            st.markdown("---")

        st.success(f"👤 Acertaste **{user_correct}/5**.")
        st.info(f"🤖 A Máquina acertou **{model_correct}/5**.")

        # Insights
        st.subheader("🧠 Insights de Aprendizagem")
        agreements = sum([user_guesses[i] == model_preds[i] for i in range(len(sample_df))])
        concord_percent = round((agreements / len(sample_df)) * 100)
        st.markdown(f"- 💬 Tu e a Máquina concordaram em **{concord_percent}%** das mensagens.")
        num_spam = user_guesses.count("spam")
        num_ham = user_guesses.count("ham")
        if num_spam > num_ham:
            estilo = "mais conservador — classificaste mais mensagens como spam."
        elif num_ham > num_spam:
            estilo = "mais permissivo — consideraste mais mensagens como não spam."
        else:
            estilo = "equilibrado — classificaste igualmente spam e não spam."
        st.markdown(f"- 🔍 Tendência humana: és **{estilo}**")
        st.markdown(f"- 🧠 Modelo utilizado: **{selected_model_name}**")

        if user_correct > model_correct:
            st.balloons()
            st.markdown("🎉 Ganhaste à máquina!")
        elif user_correct == model_correct:
            st.markdown("🤝 Empate!")
        else:
            st.markdown("💻 A máquina ganhou desta vez!")




    

# ----------- ABA 2: TESTAR MENSAGEM -------------
with tab2:
    st.title("Testa a tua própria mensagem")

    st.markdown("Escreve uma ou várias mensagens (uma por linha) para testar se são **SPAM** ou **NÃO SPAM**:")

    with st.form("message_form"):
        user_input = st.text_area("Mensagens:", height=200, placeholder="Ex: Carregue aqui no link para ganhar o prémio.")
        submitted = st.form_submit_button("Submeter")

    if submitted:
        messages = [line.strip() for line in user_input.split("\n") if line.strip()]

        if not messages:
            st.warning("Por favor, escreve pelo menos uma mensagem.")
        else:
            st.subheader("🔍 Resultados da Classificação")

            for i, msg in enumerate(messages, 1):
                st.markdown(f"**Mensagem {i}:** {msg}")
                input_vec = vectorizer.transform([msg])

                for name, mdl in models.items():
                    prediction = mdl.predict(input_vec)[0]
                    st.markdown(f"- **{name}** → `{prediction}`")
                    st.markdown(explicacao_heuristica(msg, prediction))

                st.markdown("---")
