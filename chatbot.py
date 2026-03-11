"""
╔══════════════════════════════════════════════════════════════════╗
║              HydraVision — Skin Chatbot Module                  ║
║                                                                  ║
║  Standalone chatbot powered by Groq API (llama-3.3-70b-versatile)        ║
║  Skin-only assistant with post-analysis recommendation flow     ║
║                                                                  ║
║  INTEGRATION (for your frontend):                               ║
║    from chatbot import HydraBot                                  ║
║    bot = HydraBot(api_key="gsk_...", mode="general")            ║
║    response = bot.chat("What moisturizer should I use?")        ║
║    bot.reset()   ← clear history                                ║
║                                                                  ║
║  MODES:                                                          ║
║    "general"    → general skin Q&A                              ║
║    "dehydrated" → post-analysis flow for dehydrated skin        ║
║    "hydrated"   → post-analysis flow for hydrated skin          ║
╚══════════════════════════════════════════════════════════════════╝
"""

import os
import requests
from typing import Optional, List


# ──────────────────────────────────────────────────────────────────
#  SYMPTOM OPTIONS  (use these in your UI multiselect / checkboxes)
# ──────────────────────────────────────────────────────────────────

DEHYDRATION_SYMPTOMS: List[str] = [
    "Tight / stretched feeling",
    "Dry and rough texture",
    "Flaky or peeling patches",
    "Oily / shiny (skin over-compensating)",
    "Dull, lackluster tone",
    "Fine lines / creasing",
    "Itchy or irritated",
    "Sensitive to products",
]


# ──────────────────────────────────────────────────────────────────
#  SYSTEM PROMPTS
# ──────────────────────────────────────────────────────────────────

_PROMPTS = {

    "general": """\
You are HydraBot, an expert skin health and dermatology assistant inside HydraVision.

YOUR ONLY ROLE: Answer questions strictly about skin health, skincare, and dermatology.

TOPICS YOU HANDLE:
- Skin hydration, dehydration (causes, signs, treatments)
- Skincare routines (morning/evening, product layering)
- Skin types (oily, dry, combination, sensitive, normal)
- Skin conditions (acne, eczema, rosacea, psoriasis, pigmentation, etc.)
- Skincare ingredients (hyaluronic acid, retinol, niacinamide, ceramides, SPF, etc.)
- Diet, water intake and lifestyle effects on skin
- Product recommendations (cleansers, moisturizers, serums, toners, sunscreen)
- Skin analysis interpretation

RULES:
1. NEVER answer questions outside skin/skincare/dermatology. Say:
   "I'm a skin health specialist — I can only help with skin questions! What's on your mind about your skin?"
2. Be warm, empathetic, encouraging, non-judgmental.
3. Give practical, science-backed advice in plain language.
4. Always recommend a dermatologist for serious/medical concerns.
5. Be concise but complete. Avoid filler words and repetition.
6. Do NOT diagnose — suggest possibilities and recommend professional consultation.
""",

    "dehydrated": """\
You are HydraBot, a compassionate skin health specialist inside HydraVision.
The user's skin was JUST ANALYZED by AI and found to be DEHYDRATED.

YOUR MISSION:
1. Warmly acknowledge the finding and reassure the user
2. Ask about symptoms in the highlighted red zones
3. Based on symptoms, give targeted personalized recommendations
4. Be empathetic, practical, and science-backed

SYMPTOM → RECOMMENDATION GUIDE (apply this knowledge):
- Tight/stretched       → barrier damage → occlusive moisturizer (petrolatum, shea butter, squalane)
- Dry/rough             → missing humectants → hyaluronic acid serum + emollient moisturizer
- Flaky/peeling         → needs gentle exfoliation (lactic acid 5%) + intense hydration
- Oily/shiny            → paradoxical dehydration → water-based hydration (NOT heavy oils), niacinamide
- Dull/lackluster        → antioxidants (Vit C morning) + more water intake + humectants
- Fine lines/creasing   → peptides + layered hyaluronic acid + occlusives at night
- Itchy/irritated       → barrier repair: ceramides, colloidal oatmeal, centella asiatica
- Sensitive to products → fragrance-free, minimal ingredients (CeraVe, La Roche-Posay, Vanicream)

RULES:
1. Start by warmly acknowledging dehydration result
2. Ask about symptoms first, then give recommendations based on what they say
3. NEVER answer non-skin questions — redirect politely
""",

    "hydrated": """\
You are HydraBot, a skin health specialist inside HydraVision.
The user's skin was JUST ANALYZED and found to be HYDRATED — excellent result!

YOUR MISSION:
1. Celebrate their great skin with warmth and positivity
2. Offer to give personalized maintenance tips
3. Help them keep their skin healthy long-term
4. Answer any skin questions they have

MAINTENANCE TOPICS TO COVER:
- How to maintain skin hydration through the seasons
- Daily water intake, sleep, and stress management
- SPF protection and its role in hydration
- Morning vs. evening routine optimization
- Early warning signs of dehydration to watch for

RULES: Only answer skin-related questions. Redirect anything else politely.
"""
}


# ──────────────────────────────────────────────────────────────────
#  HYDRABOT CLASS
# ──────────────────────────────────────────────────────────────────

class HydraBot:
    """
    Standalone HydraVision skin chatbot.

    Quick start:
        bot = HydraBot(api_key="gsk_...", mode="general")
        print(bot.get_opening_message())   # show greeting
        print(bot.chat("my skin feels tight"))
        bot.reset()  # clear history for new session

    For dehydrated post-analysis:
        bot = HydraBot(api_key="gsk_...", mode="dehydrated")
        print(bot.get_opening_message())
        # Show DEHYDRATION_SYMPTOMS checkboxes in your UI
        # User selects symptoms, then:
        print(bot.chat_with_symptoms(["Tight / stretched feeling", "Flaky or peeling patches"]))
        # Continue with follow-up questions:
        print(bot.chat("What products should I avoid?"))
    """

    GROQ_URL         = "https://api.groq.com/openai/v1/chat/completions"
    MODEL            = "llama-3.3-70b-versatile"
    MAX_TOKENS       = 600
    MAX_HISTORY_TURNS = 10

    def __init__(self, api_key: Optional[str] = None, mode: str = "general"):
        """
        Args:
            api_key : Groq API key. Falls back to GROQ_API_KEY environment variable.
            mode    : "general" | "dehydrated" | "hydrated"
        """
        self.api_key = (api_key or os.environ.get("GROQ_API_KEY", "")).strip()
        self.mode    = mode if mode in _PROMPTS else "general"
        self.history: List[dict] = []

    # ── Public methods ────────────────────────────────────────────

    def chat(self, user_message: str) -> str:
        """
        Send a user message and return the bot's response.

        Args:
            user_message: plain text from the user

        Returns:
            Bot reply string (includes error messages on failure)
        """
        if not self.api_key:
            return (
                "⚠️ No Groq API key set. "
                "Pass api_key= or set GROQ_API_KEY environment variable. "
                "Free key: https://console.groq.com/keys"
            )

        user_message = user_message.strip()
        if not user_message:
            return "Please type your skin question!"

        self.history.append({"role": "user", "content": user_message})
        messages  = self._build_clean_messages()
        response  = self._call_api(messages)
        self.history.append({"role": "assistant", "content": response})
        return response

    def chat_with_symptoms(self, symptoms: List[str]) -> str:
        """
        Shortcut for the dehydrated post-analysis symptom submission.

        Args:
            symptoms: list of selected symptoms from DEHYDRATION_SYMPTOMS

        Returns:
            Personalized recommendation response
        """
        if not symptoms:
            return "Please select at least one symptom so I can personalize your recommendations!"

        msg = (
            "My skin in the dehydrated areas feels: "
            + ", ".join(symptoms)
            + ". Please give me personalized hydration recommendations for each symptom."
        )
        return self.chat(msg)

    def get_opening_message(self) -> str:
        """Return the greeting/opening message for the current mode."""
        openings = {
            "general": (
                "Hello! 👋 I'm HydraBot, your personal skin health specialist.\n\n"
                "I'm here to help with everything skin-related: hydration, routines, "
                "ingredients, conditions, and more.\n\n"
                "What would you like to know about your skin today?"
            ),
            "dehydrated": (
                "I can see your skin is showing signs of dehydration — but don't worry, "
                "we'll work on getting it back to its best! 💧\n\n"
                "To give you the most accurate recommendations, I'd love to understand "
                "your symptoms better.\n\n"
                "Can you touch the red-highlighted areas on the analysis map and tell me "
                "how they feel? Please select all that apply:\n\n"
                + "\n".join(f"• {s}" for s in DEHYDRATION_SYMPTOMS)
                + "\n\nOnce you've selected, I'll give you a personalized plan!"
            ),
            "hydrated": (
                "Wonderful news! 🎉 Your skin analysis shows healthy hydration levels!\n\n"
                "Would you like some personalized tips to *maintain* this and keep your "
                "skin glowing long-term?\n\n"
                "Feel free to ask me anything about skincare too! 💧"
            ),
        }
        return openings.get(self.mode, openings["general"])

    def reset(self):
        """Clear the conversation history."""
        self.history = []

    def set_mode(self, mode: str):
        """Switch mode and reset history."""
        self.mode = mode if mode in _PROMPTS else "general"
        self.history = []

    def set_api_key(self, api_key: str):
        """Update the Groq API key."""
        self.api_key = api_key.strip()

    def get_history(self) -> List[dict]:
        """Return a copy of the conversation history."""
        return list(self.history)

    # ── Private helpers ───────────────────────────────────────────

    def _build_clean_messages(self) -> List[dict]:
        """
        Sanitize history for the Groq API:
        - Only user / assistant roles
        - Trim to last MAX_HISTORY_TURNS pairs
        - Must start with 'user', must end with 'user'
        """
        clean = [m for m in self.history if m.get("role") in ("user", "assistant")]

        max_msgs = self.MAX_HISTORY_TURNS * 2
        if len(clean) > max_msgs:
            clean = clean[-max_msgs:]

        while clean and clean[0]["role"] != "user":
            clean = clean[1:]

        while clean and clean[-1]["role"] != "user":
            clean = clean[:-1]

        return clean

    def _call_api(self, messages: List[dict]) -> str:
        """Call the Groq API and return the text response."""
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type":  "application/json",
        }
        payload = {
            "model":       self.MODEL,
            "messages":    [{"role": "system", "content": _PROMPTS[self.mode]}] + messages,
            "max_tokens":  self.MAX_TOKENS,
            "temperature": 0.7,
        }

        try:
            r = requests.post(self.GROQ_URL, headers=headers, json=payload, timeout=30)

            if r.status_code == 401:
                return (
                    "❌ Invalid API key. "
                    "Please check it at https://console.groq.com/keys"
                )
            if r.status_code == 429:
                return "⏳ Rate limit hit — please wait a moment and try again."
            if r.status_code == 400:
                try:
                    detail = r.json().get("error", {}).get("message", r.text)
                except Exception:
                    detail = r.text
                return f"❌ Request error: {detail}"

            r.raise_for_status()
            return r.json()["choices"][0]["message"]["content"]

        except requests.exceptions.Timeout:
            return "⏱️ Request timed out — please try again."
        except requests.exceptions.ConnectionError:
            return "🌐 Connection error — please check your internet."
        except Exception as e:
            return f"❌ Unexpected error: {str(e)}"


# ──────────────────────────────────────────────────────────────────
#  STREAMLIT DEMO  ← runs only when: python -m streamlit run chatbot.py
# ──────────────────────────────────────────────────────────────────

def _run_streamlit_demo():
    import streamlit as st

    st.set_page_config(page_title="HydraBot", page_icon="💧", layout="centered")

    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;700;800&family=DM+Mono:wght@400;500&display=swap');
    html,body,[class*="css"]{font-family:'Syne',sans-serif;background:#0a0f1e;color:#e8eaf6;}
    .main{background:#0a0f1e;}
    .chat-wrap{background:#080d1a;border:1px solid #1e3a5f;border-radius:16px;
               padding:1.5rem;max-height:520px;overflow-y:auto;margin-bottom:1rem;}
    .msg-user{background:linear-gradient(135deg,#1565c0,#0d47a1);
              border-radius:16px 16px 4px 16px;padding:.75rem 1.1rem;
              margin:.5rem 0 .5rem 22%;color:#fff;font-size:.9rem;line-height:1.55;
              animation:pop .18s ease;}
    .msg-bot{background:#0d1b3e;border:1px solid #1e3a5f;
             border-radius:16px 16px 16px 4px;padding:.75rem 1.1rem;
             margin:.5rem 22% .5rem 0;color:#e8eaf6;font-size:.9rem;line-height:1.55;
             animation:pop .18s ease;}
    .role{font-family:'DM Mono',monospace;font-size:.68rem;color:#546e8a;margin-bottom:.2rem;}
    @keyframes pop{from{opacity:0;transform:translateY(5px)}to{opacity:1;transform:translateY(0)}}
    .stButton>button{background:linear-gradient(135deg,#1565c0,#0d47a1)!important;
                     color:#fff!important;border:none;border-radius:8px;
                     font-family:'Syne',sans-serif;font-weight:700;width:100%;}
    section[data-testid="stSidebar"]{background:#080d1a!important;border-right:1px solid #1e3a5f;}
    section[data-testid="stSidebar"] *{color:#b0c4de!important;}
    hr{border-color:#1e3a5f;}
    input[type="text"],input[type="password"]{
        background:#0d1b3e!important;color:#e8eaf6!important;
        border:1px solid #1e3a5f!important;border-radius:8px!important;}
    </style>
    """, unsafe_allow_html=True)

    # Session defaults
    for k, v in [("bot", None), ("msgs", []), ("mode", "general"), ("started", False)]:
        if k not in st.session_state:
            st.session_state[k] = v

    # ── Sidebar ──
    with st.sidebar:
        st.markdown("### 💧 HydraBot")
        st.markdown("---")

        api_key = st.text_input("Groq API Key", type="password", placeholder="gsk_...")
        st.markdown("[🔗 Get free key at console.groq.com](https://console.groq.com/keys)")

        st.markdown("---")
        mode = st.radio(
            "Mode",
            ["general", "dehydrated", "hydrated"],
            format_func=lambda x: {
                "general":    "💬 General Skin Q&A",
                "dehydrated": "⚠️ Post-Analysis: Dehydrated",
                "hydrated":   "✅ Post-Analysis: Hydrated",
            }[x],
        )

        if st.button("🚀 Start Chatbot"):
            bot = HydraBot(api_key=api_key, mode=mode)
            st.session_state.bot     = bot
            st.session_state.mode    = mode
            st.session_state.started = True
            st.session_state.msgs    = [
                {"role": "assistant", "content": bot.get_opening_message()}
            ]
            st.rerun()

        st.markdown("---")
        if st.button("🗑️ Clear Chat"):
            if st.session_state.bot:
                st.session_state.bot.reset()
            st.session_state.msgs = []
            st.rerun()

        st.markdown("""<div style="font-family:'DM Mono',monospace;font-size:.72rem;
        color:#546e8a;margin-top:1rem;line-height:1.8;">
        📦 <b>Integration:</b><br>
        <span style="color:#4fc3f7">from chatbot import HydraBot</span><br>
        <span style="color:#4fc3f7">bot = HydraBot(key, mode)</span><br>
        <span style="color:#4fc3f7">reply = bot.chat("text")</span>
        </div>""", unsafe_allow_html=True)

    # ── Header ──
    st.markdown("""
    <div style="background:linear-gradient(135deg,#0d1b3e,#0a2a4a);border:1px solid #1e3a5f;
    border-radius:16px;padding:1.5rem 2rem;margin-bottom:1.5rem;text-align:center;">
    <h1 style="font-size:2rem;font-weight:800;
    background:linear-gradient(90deg,#4fc3f7,#b3e5fc);
    -webkit-background-clip:text;-webkit-text-fill-color:transparent;margin:0 0 .3rem;">
    💧 HydraBot</h1>
    <p style="color:#90caf9;font-family:'DM Mono',monospace;font-size:.82rem;margin:0;">
    Skin Health AI · Groq llama3-8b · Skin questions only</p>
    </div>
    """, unsafe_allow_html=True)

    if not st.session_state.started:
        st.info("👈 Enter your Groq API key in the sidebar and click **Start Chatbot**.")
        st.stop()

    # ── Chat display ──
    if st.session_state.msgs:
        html = '<div class="chat-wrap">'
        for m in st.session_state.msgs:
            content = (
                m["content"]
                .replace("\n", "<br>")
                .replace("**", "<b>", 1)
            )
            if m["role"] == "user":
                html += f'<div class="msg-user"><div class="role">You</div>{content}</div>'
            else:
                html += f'<div class="msg-bot"><div class="role">💧 HydraBot</div>{content}</div>'
        html += '</div>'
        st.markdown(html, unsafe_allow_html=True)

    # ── Symptom multiselect (dehydrated mode, first turn) ──
    user_turns = [m for m in st.session_state.msgs if m["role"] == "user"]
    is_dehy_first = st.session_state.mode == "dehydrated" and len(user_turns) == 0

    if is_dehy_first:
        st.markdown("**Select all symptoms that apply to the red-highlighted areas:**")
        selected = st.multiselect("", DEHYDRATION_SYMPTOMS, label_visibility="collapsed")
        if st.button("Get My Personalized Plan →"):
            if selected:
                with st.spinner("Preparing your personalized recommendations..."):
                    resp = st.session_state.bot.chat_with_symptoms(selected)
                st.session_state.msgs.append({"role": "user",      "content": "My skin feels: " + ", ".join(selected)})
                st.session_state.msgs.append({"role": "assistant", "content": resp})
                st.rerun()
            else:
                st.warning("Please select at least one symptom.")
    else:
        # ── Regular text input ──
        col_i, col_b = st.columns([5, 1])
        with col_i:
            user_input = st.text_input(
                "msg", key="user_input",
                label_visibility="collapsed",
                placeholder="Ask about your skin...",
                value=st.session_state.get("chat_input_val", "")
            )
        with col_b:
            send = st.button("Send ›")

        if send and user_input.strip():
            msg = user_input.strip()
            st.session_state["chat_input_val"] = ""
            with st.spinner(""):
                resp = st.session_state.bot.chat(msg)
            st.session_state.msgs.append({"role": "user",      "content": msg})
            st.session_state.msgs.append({"role": "assistant", "content": resp})
            st.rerun()


if __name__ == "__main__":
    _run_streamlit_demo()