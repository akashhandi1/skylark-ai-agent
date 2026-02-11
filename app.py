import os
import requests
import pandas as pd
import numpy as np
from datetime import datetime
from dateutil import parser
from fastapi import FastAPI
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel

# -----------------------------
# APP SETUP
# -----------------------------
app = FastAPI()

MONDAY_API_KEY = os.getenv("MONDAY_API_KEY")
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
DEALS_BOARD_ID = int(os.getenv("DEALS_BOARD_ID"))
WORK_BOARD_ID = int(os.getenv("WORK_BOARD_ID"))

MONDAY_URL = "https://api.monday.com/v2"

# -----------------------------
# MODELS
# -----------------------------
class Query(BaseModel):
    question: str

# -----------------------------
# MONDAY API
# -----------------------------
def fetch_board_data(board_id):
    query = f"""
    {{
      boards(ids: {board_id}) {{
        items_page(limit: 500) {{
          items {{
            name
            column_values {{
              id
              text
            }}
          }}
        }}
      }}
    }}
    """
    headers = {"Authorization": MONDAY_API_KEY}
    response = requests.post(MONDAY_URL, json={"query": query}, headers=headers)

    if response.status_code != 200:
        raise Exception(f"Monday API Error: {response.text}")

    return response.json()

def convert_board_to_dataframe(raw_data):
    if not raw_data.get("data") or not raw_data["data"]["boards"]:
        return pd.DataFrame()

    items = raw_data["data"]["boards"][0]["items_page"]["items"]
    rows = []

    for item in items:
        row = {"Item Name": item["name"]}
        for col in item["column_values"]:
            row[col["id"]] = col["text"]
        rows.append(row)

    df = pd.DataFrame(rows)
    return df.replace("", np.nan)

# -----------------------------
# UTILITIES
# -----------------------------
def normalize_probability(prob):
    mapping = {"High": 0.8, "Medium": 0.5, "Low": 0.2}
    return mapping.get(str(prob).strip(), 0.3)

def safe_parse_date(date_str):
    try:
        return parser.parse(date_str)
    except:
        return None

# -----------------------------
# BUSINESS LOGIC
# -----------------------------
def compute_deal_metrics(df):
    if df.empty:
        return {"error": "No deal data available."}

    df["value"] = pd.to_numeric(df["numeric_mm0f6tc2"], errors="coerce")
    df["prob"] = df["color_mm0f30w0"].apply(normalize_probability)
    df["weighted"] = df["value"] * df["prob"]

    # Only open deals
    df = df[df["color_mm0fqvp6"] == "Open"]

    return {
        "total_pipeline": float(df["value"].sum(skipna=True)),
        "weighted_pipeline": float(df["weighted"].sum(skipna=True)),
        "sector_breakdown": df.groupby("color_mm0fe66m")["value"].sum().to_dict(),
        "stage_distribution": df["color_mm0f24qr"].value_counts().to_dict()
    }

def compute_work_order_metrics(df):
    if df.empty:
        return {"error": "No work order data available."}

    df["planned_date"] = df["date_mm0fpdes"].apply(safe_parse_date)
    df["actual_date"] = df["date_mm0fz6jw"].apply(safe_parse_date)
    df["delay_days"] = (df["actual_date"] - df["planned_date"]).dt.days

    return {
        "average_delay_days": float(df["delay_days"].mean()) if df["delay_days"].notna().any() else 0,
        "sector_delay": df.groupby("color_mm0f5e45")["delay_days"].mean().dropna().to_dict(),
        "execution_status_distribution": df["color_mm0fcx9e"].value_counts().to_dict()
    }

# -----------------------------
# OPENROUTER SUMMARY
# -----------------------------
def generate_summary(structured_data, question):
    url = "https://openrouter.ai/api/v1/chat/completions"

    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json"
    }

    payload = {
        "model": "openai/gpt-4o-mini",
        "messages": [
            {
                "role": "user",
                "content": f"""
You are a founder-level business intelligence assistant.

User Question:
{question}

Structured Data:
{structured_data}

Provide:
1. Executive Summary
2. Key Risks
3. Opportunities
4. Data Caveats

Be concise and strategic.
"""
            }
        ],
        "temperature": 0.3
    }

    response = requests.post(url, headers=headers, json=payload)

    if response.status_code != 200:
        return "Error generating summary from OpenRouter."

    return response.json()["choices"][0]["message"]["content"]

# -----------------------------
# HTML CONTENT
# -----------------------------
HTML_CONTENT = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Skylark BI Agent</title>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }

        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            display: flex;
            align-items: center;
            justify-content: center;
            padding: 20px;
        }

        #chatbox {
            width: 100%;
            max-width: 900px;
            height: 90vh;
            background: white;
            border-radius: 20px;
            box-shadow: 0 20px 60px rgba(0, 0, 0, 0.3);
            display: flex;
            flex-direction: column;
            overflow: hidden;
        }

        .header {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 24px 30px;
            display: flex;
            align-items: center;
            gap: 12px;
        }

        .header h2 {
            font-size: 24px;
            font-weight: 600;
            display: flex;
            align-items: center;
            gap: 10px;
        }

        .logo {
            font-size: 32px;
            animation: float 3s ease-in-out infinite;
        }

        @keyframes float {
            0%, 100% { transform: translateY(0px); }
            50% { transform: translateY(-10px); }
        }

        .subtitle {
            font-size: 14px;
            opacity: 0.9;
            margin-top: 4px;
        }

        #messages {
            flex: 1;
            overflow-y: auto;
            padding: 30px;
            background: #f8f9fa;
            display: flex;
            flex-direction: column;
            gap: 16px;
        }

        #messages::-webkit-scrollbar {
            width: 8px;
        }

        #messages::-webkit-scrollbar-track {
            background: #f1f1f1;
        }

        #messages::-webkit-scrollbar-thumb {
            background: #c1c1c1;
            border-radius: 4px;
        }

        #messages::-webkit-scrollbar-thumb:hover {
            background: #a8a8a8;
        }

        .message {
            display: flex;
            align-items: flex-start;
            gap: 12px;
            animation: slideIn 0.3s ease-out;
        }

        @keyframes slideIn {
            from {
                opacity: 0;
                transform: translateY(10px);
            }
            to {
                opacity: 1;
                transform: translateY(0);
            }
        }

        .message.user {
            flex-direction: row-reverse;
        }

        .avatar {
            width: 36px;
            height: 36px;
            border-radius: 50%;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 18px;
            flex-shrink: 0;
        }

        .user .avatar {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        }

        .bot .avatar {
            background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        }

        .message-content {
            max-width: 70%;
            padding: 14px 18px;
            border-radius: 18px;
            word-wrap: break-word;
            line-height: 1.6;
        }

        .user .message-content {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            border-bottom-right-radius: 4px;
        }

        .bot .message-content {
            background: white;
            color: #2d3748;
            border: 1px solid #e2e8f0;
            border-bottom-left-radius: 4px;
            box-shadow: 0 2px 8px rgba(0, 0, 0, 0.05);
        }

        .message-content h1,
        .message-content h2,
        .message-content h3 {
            margin-top: 12px;
            margin-bottom: 8px;
            font-weight: 600;
        }

        .message-content h1 { font-size: 1.5em; }
        .message-content h2 { font-size: 1.3em; }
        .message-content h3 { font-size: 1.1em; }

        .message-content p {
            margin-bottom: 10px;
        }

        .message-content ul,
        .message-content ol {
            margin-left: 20px;
            margin-bottom: 10px;
        }

        .message-content li {
            margin-bottom: 4px;
        }

        .message-content code {
            background: rgba(0, 0, 0, 0.05);
            padding: 2px 6px;
            border-radius: 4px;
            font-family: 'Courier New', monospace;
            font-size: 0.9em;
        }

        .user .message-content code {
            background: rgba(255, 255, 255, 0.2);
        }

        .message-content pre {
            background: #2d3748;
            color: #e2e8f0;
            padding: 12px;
            border-radius: 8px;
            overflow-x: auto;
            margin: 10px 0;
        }

        .message-content pre code {
            background: none;
            padding: 0;
            color: inherit;
        }

        .message-content strong {
            font-weight: 600;
        }

        .message-content em {
            font-style: italic;
        }

        .message-content a {
            color: #667eea;
            text-decoration: none;
            border-bottom: 1px solid #667eea;
        }

        .user .message-content a {
            color: white;
            border-bottom-color: white;
        }

        .input-area {
            padding: 20px 30px;
            background: white;
            border-top: 1px solid #e2e8f0;
            display: flex;
            gap: 12px;
            align-items: center;
        }

        #question {
            flex: 1;
            padding: 14px 20px;
            border: 2px solid #e2e8f0;
            border-radius: 24px;
            font-size: 15px;
            outline: none;
            transition: all 0.3s ease;
            font-family: inherit;
        }

        #question:focus {
            border-color: #667eea;
            box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.1);
        }

        button {
            padding: 14px 28px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            border: none;
            border-radius: 24px;
            font-size: 15px;
            font-weight: 600;
            cursor: pointer;
            transition: all 0.3s ease;
            box-shadow: 0 4px 12px rgba(102, 126, 234, 0.3);
        }

        button:hover {
            transform: translateY(-2px);
            box-shadow: 0 6px 16px rgba(102, 126, 234, 0.4);
        }

        button:active {
            transform: translateY(0);
        }

        button:disabled {
            opacity: 0.6;
            cursor: not-allowed;
            transform: none;
        }

        .typing-indicator {
            display: flex;
            gap: 4px;
            padding: 14px 18px;
        }

        .typing-indicator span {
            width: 8px;
            height: 8px;
            background: #cbd5e0;
            border-radius: 50%;
            animation: typing 1.4s infinite;
        }

        .typing-indicator span:nth-child(2) {
            animation-delay: 0.2s;
        }

        .typing-indicator span:nth-child(3) {
            animation-delay: 0.4s;
        }

        @keyframes typing {
            0%, 60%, 100% {
                transform: translateY(0);
                opacity: 0.7;
            }
            30% {
                transform: translateY(-10px);
                opacity: 1;
            }
        }

        .welcome-message {
            text-align: center;
            padding: 40px 20px;
            color: #718096;
        }

        .welcome-message h3 {
            font-size: 20px;
            margin-bottom: 12px;
            color: #2d3748;
        }

        .welcome-message p {
            font-size: 14px;
            line-height: 1.6;
        }

        .suggestions {
            display: flex;
            flex-wrap: wrap;
            gap: 10px;
            justify-content: center;
            margin-top: 20px;
        }

        .suggestion {
            padding: 10px 16px;
            background: white;
            border: 1px solid #e2e8f0;
            border-radius: 16px;
            cursor: pointer;
            font-size: 13px;
            transition: all 0.2s ease;
        }

        .suggestion:hover {
            background: #f7fafc;
            border-color: #667eea;
            color: #667eea;
        }

        @media (max-width: 768px) {
            #chatbox {
                height: 100vh;
                border-radius: 0;
            }

            .message-content {
                max-width: 85%;
            }

            .header {
                padding: 20px;
            }

            #messages {
                padding: 20px;
            }

            .input-area {
                padding: 15px 20px;
            }
        }
    </style>
</head>
<body>
<div id="chatbox">
    <div class="header">
        <div>
            <h2>
                <span class="logo">🚀</span>
                Skylark Drones BI Agent
            </h2>
            <div class="subtitle">Your intelligent business analytics assistant</div>
        </div>
    </div>

    <div id="messages">
        <div class="welcome-message">
            <h3>Welcome to Skylark BI Agent!</h3>
            <p>Ask me anything about your business data, analytics, or insights.</p>
            <div class="suggestions">
                <div class="suggestion" onclick="sendSuggestion('What are our top-performing products?')">Top products</div>
                <div class="suggestion" onclick="sendSuggestion('Show me sales trends')">Sales trends</div>
                <div class="suggestion" onclick="sendSuggestion('Customer analytics')">Customer analytics</div>
            </div>
        </div>
    </div>

    <div class="input-area">
        <input type="text" id="question" placeholder="Ask a business question..." onkeypress="handleKeyPress(event)" />
        <button onclick="send()" id="sendBtn">Send</button>
    </div>
</div>

<script>
// Markdown-like text processing
function formatText(text) {
    if (!text) return '';
    
    // Escape HTML first
    let formatted = text
        .replace(/&/g, '&amp;')
        .replace(/</g, '&lt;')
        .replace(/>/g, '&gt;');
    
    // Code blocks (```code```)
    formatted = formatted.replace(/```([\\s\\S]*?)```/g, '<pre><code>$1</code></pre>');
    
    // Inline code (`code`)
    formatted = formatted.replace(/`([^`]+)`/g, '<code>$1</code>');
    
    // Bold (**text** or __text__)
    formatted = formatted.replace(/\\*\\*(.+?)\\*\\*/g, '<strong>$1</strong>');
    formatted = formatted.replace(/__(.+?)__/g, '<strong>$1</strong>');
    
    // Italic (*text* or _text_)
    formatted = formatted.replace(/\\*(.+?)\\*/g, '<em>$1</em>');
    formatted = formatted.replace(/_(.+?)_/g, '<em>$1</em>');
    
    // Links [text](url)
    formatted = formatted.replace(/\\[([^\\]]+)\\]\\(([^)]+)\\)/g, '<a href="$2" target="_blank">$1</a>');
    
    // Headers
    formatted = formatted.replace(/^### (.+)$/gm, '<h3>$1</h3>');
    formatted = formatted.replace(/^## (.+)$/gm, '<h2>$1</h2>');
    formatted = formatted.replace(/^# (.+)$/gm, '<h1>$1</h1>');
    
    // Unordered lists
    formatted = formatted.replace(/^\\* (.+)$/gm, '<li>$1</li>');
    formatted = formatted.replace(/^- (.+)$/gm, '<li>$1</li>');
    formatted = formatted.replace(/(<li>.*<\\/li>\\n?)+/g, '<ul>$&</ul>');
    
    // Ordered lists
    formatted = formatted.replace(/^\\d+\\. (.+)$/gm, '<li>$1</li>');
    
    // Line breaks
    formatted = formatted.replace(/\\n\\n/g, '</p><p>');
    formatted = '<p>' + formatted + '</p>';
    
    // Clean up empty paragraphs
    formatted = formatted.replace(/<p><\\/p>/g, '');
    formatted = formatted.replace(/<p>(<[huo][^>]*>)/g, '$1');
    formatted = formatted.replace(/(<\\/[huo][^>]*>)<\\/p>/g, '$1');
    
    return formatted;
}

function addMessage(content, isUser) {
    const messages = document.getElementById("messages");
    
    // Remove welcome message if it exists
    const welcome = messages.querySelector('.welcome-message');
    if (welcome) {
        welcome.remove();
    }
    
    const messageDiv = document.createElement('div');
    messageDiv.className = `message ${isUser ? 'user' : 'bot'}`;
    
    const avatar = document.createElement('div');
    avatar.className = 'avatar';
    avatar.textContent = isUser ? '👤' : '🤖';
    
    const contentDiv = document.createElement('div');
    contentDiv.className = 'message-content';
    contentDiv.innerHTML = isUser ? content.replace(/\\n/g, '<br>') : formatText(content);
    
    messageDiv.appendChild(avatar);
    messageDiv.appendChild(contentDiv);
    
    messages.appendChild(messageDiv);
    messages.scrollTop = messages.scrollHeight;
}

function showTypingIndicator() {
    const messages = document.getElementById("messages");
    const typingDiv = document.createElement('div');
    typingDiv.className = 'message bot';
    typingDiv.id = 'typing-indicator';
    
    typingDiv.innerHTML = `
        <div class="avatar">🤖</div>
        <div class="message-content">
            <div class="typing-indicator">
                <span></span>
                <span></span>
                <span></span>
            </div>
        </div>
    `;
    
    messages.appendChild(typingDiv);
    messages.scrollTop = messages.scrollHeight;
}

function removeTypingIndicator() {
    const typing = document.getElementById('typing-indicator');
    if (typing) {
        typing.remove();
    }
}

async function send() {
    const input = document.getElementById("question");
    const sendBtn = document.getElementById("sendBtn");
    
    if (!input.value.trim()) return;
    
    const question = input.value.trim();
    input.value = "";
    
    // Disable input while processing
    input.disabled = true;
    sendBtn.disabled = true;
    
    // Add user message
    addMessage(question, true);
    
    // Show typing indicator
    showTypingIndicator();
    
    try {
        const response = await fetch("/ask", {
            method: "POST",
            headers: {"Content-Type": "application/json"},
            body: JSON.stringify({question: question})
        });
        
        if (!response.ok) {
            throw new Error('Network response was not ok');
        }
        
        const data = await response.json();
        
        // Remove typing indicator
        removeTypingIndicator();
        
        // Add bot response
        addMessage(data.response || "I'm sorry, I couldn't process that request.", false);
        
    } catch (error) {
        removeTypingIndicator();
        addMessage("Sorry, there was an error processing your request. Please try again.", false);
        console.error('Error:', error);
    } finally {
        // Re-enable input
        input.disabled = false;
        sendBtn.disabled = false;
        input.focus();
    }
}

function sendSuggestion(text) {
    const input = document.getElementById("question");
    input.value = text;
    send();
}

function handleKeyPress(event) {
    if (event.key === 'Enter') {
        send();
    }
}

// Auto-focus input on load
window.onload = function() {
    document.getElementById("question").focus();
};
</script>
</body>
</html>
"""

# -----------------------------
# ROUTES
# -----------------------------
@app.get("/", response_class=HTMLResponse)
def home():
    return HTMLResponse(content=HTML_CONTENT)

@app.post("/ask")
def ask(query: Query):
    try:
        deals_raw = fetch_board_data(DEALS_BOARD_ID)
        work_raw = fetch_board_data(WORK_BOARD_ID)

        deals_df = convert_board_to_dataframe(deals_raw)
        work_df = convert_board_to_dataframe(work_raw)

        deal_metrics = compute_deal_metrics(deals_df)
        work_metrics = compute_work_order_metrics(work_df)

        structured_data = {
            "deal_metrics": deal_metrics,
            "work_order_metrics": work_metrics
        }

        answer = generate_summary(structured_data, query.question)

        return {"response": answer}

    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": str(e)}
        )
