import streamlit as st
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.schema import HumanMessage
import base64
import time

# Page Configuration

st.set_page_config(page_title="🤖 LLM App: ", layout="wide")
st.title("💬 Multi-Model Chat App ")


#  API Keys 

OPENROUTER_KEY = "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"
GEMINI_KEY = "AIzaSyDPZwFzkBNAyyoxBcvgkQQC09XEl6wOJ"


# Sidebar Configuration

st.sidebar.header("⚙️ Settings")

provider = st.sidebar.selectbox("🌐 Choose Provider", ["OpenRouter", "Gemini"])
temperature = st.sidebar.slider("🎛️ Creativity (Temperature)", 0.0, 1.0, 0.7)
max_tokens = st.sidebar.slider("🧠 Max Tokens", 100, 2000, 512)
chat_mode = st.sidebar.radio("💬 Chat Mode", ["Text Only", "Text + Image"])


# Generate AI Response

def generate_response(prompt, image_file=None):
    try:
        if provider == "OpenRouter":
            llm = ChatOpenAI(
                model="openai/gpt-4o-mini",
                openai_api_key=OPENROUTER_KEY,
                base_url="https://openrouter.ai/api/v1",
                temperature=temperature,
                max_tokens=max_tokens,
            )

            if image_file:
                image_bytes = image_file.read()
                image_b64 = base64.b64encode(image_bytes).decode("utf-8")
                message = [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {"type": "image_url", "image_url": f"data:image/png;base64,{image_b64}"}
                        ],
                    }
                ]
                response = llm.invoke(message)
            else:
                response = llm.invoke([HumanMessage(content=prompt)])

            return response.content

        elif provider == "Gemini":
            llm = ChatGoogleGenerativeAI(
                model="gemini-1.5-flash",
                google_api_key=GEMINI_KEY,
                temperature=temperature,
                max_output_tokens=max_tokens,
            )

            if image_file:
                image_data = image_file.read()
                message = [
                    HumanMessage(content=[
                        {"type": "text", "text": prompt},
                        {"type": "image", "data": image_data},
                    ])
                ]
            else:
                message = [HumanMessage(content=prompt)]

            response = llm.invoke(message)
            return response.content

    except Exception as e:
        return f" Error: {str(e)}"

# Main Interface

st.markdown("### 💡 Ask anything — text or image-based reasoning supported!")

if "messages" not in st.session_state:
    st.session_state.messages = []

with st.form("chat_form"):
    user_input = st.text_area("📝 Enter your message:", placeholder="Type your question here...")
    image_file = None
    if chat_mode == "Text + Image":
        image_file = st.file_uploader("📷 Upload an image (optional)", type=["png", "jpg", "jpeg"])

    submitted = st.form_submit_button("🚀 Send")

    if submitted and user_input.strip():
        with st.spinner("🤔 Thinking..."):
            ai_output = generate_response(user_input, image_file)
            st.session_state.messages.append({"role": "user", "content": user_input})
            st.session_state.messages.append({"role": "assistant", "content": ai_output})
            time.sleep(0.4)
        st.rerun()  


# Display Chat History

st.markdown("### 🗨️ Chat History")

for msg in st.session_state.messages:
    if msg["role"] == "user":
        st.markdown(f"**👤 You:** {msg['content']}")
    else:
        st.markdown(f"**🤖 AI:** {msg['content']}")

# Footer

st.sidebar.markdown("---")
st.sidebar.info("Built with ❤️ using Streamlit + LangChain + OpenRouter + Gemini")

