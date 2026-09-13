const API_BASE = "/api";
const HISTORY_KEY = "pyqa_chat_history";

const chatWindow = document.getElementById("chat-window");
const chatForm = document.getElementById("chat-form");
const chatInput = document.getElementById("chat-input");
const sendBtn = document.getElementById("send-btn");
const clearBtn = document.getElementById("clear-btn");
const suggestionsEl = document.getElementById("suggestions");
const errorBanner = document.getElementById("error-banner");

const SUGGESTED_QUESTIONS = [
  "What is a variable in Python?",
  "What is the difference between a list and a tuple?",
  "What are *args and **kwargs?",
  "How do you handle exceptions with try-except?",
  "What is a lambda function?",
  "What is the difference between deep copy and shallow copy?",
];

let history = [];

function setInputDisabled(disabled) {
  chatInput.disabled = disabled;
  sendBtn.disabled = disabled;
  suggestionsEl.querySelectorAll(".suggestion-chip").forEach((chip) => {
    chip.disabled = disabled;
  });
}

function renderMessage({ role, text, codeExample, category, isError }) {
  const wrapper = document.createElement("div");
  wrapper.className = `message ${role}${isError ? " error" : ""}`;

  const bubble = document.createElement("div");
  bubble.className = "bubble";

  if (category) {
    const tag = document.createElement("span");
    tag.className = "category-tag";
    tag.textContent = category;
    bubble.appendChild(tag);
  }

  const textEl = document.createElement("div");
  textEl.className = "message-text";
  textEl.textContent = text;
  bubble.appendChild(textEl);

  if (codeExample) {
    const codeWrap = document.createElement("div");
    codeWrap.className = "code-block";

    const pre = document.createElement("pre");
    const code = document.createElement("code");
    code.textContent = codeExample;
    pre.appendChild(code);

    const copyBtn = document.createElement("button");
    copyBtn.type = "button";
    copyBtn.className = "copy-btn";
    copyBtn.textContent = "Copy";
    copyBtn.addEventListener("click", () => {
      navigator.clipboard.writeText(codeExample).then(() => {
        copyBtn.textContent = "Copied!";
        setTimeout(() => {
          copyBtn.textContent = "Copy";
        }, 1500);
      });
    });

    codeWrap.appendChild(pre);
    codeWrap.appendChild(copyBtn);
    bubble.appendChild(codeWrap);
  }

  wrapper.appendChild(bubble);
  chatWindow.appendChild(wrapper);
  chatWindow.scrollTop = chatWindow.scrollHeight;
}

function showTypingIndicator() {
  const wrapper = document.createElement("div");
  wrapper.className = "message bot typing-indicator";
  wrapper.id = "typing-indicator";
  wrapper.innerHTML =
    '<div class="bubble"><span class="dot"></span><span class="dot"></span><span class="dot"></span></div>';
  chatWindow.appendChild(wrapper);
  chatWindow.scrollTop = chatWindow.scrollHeight;
}

function hideTypingIndicator() {
  const el = document.getElementById("typing-indicator");
  if (el) el.remove();
}

function showError(message) {
  errorBanner.textContent = message;
  errorBanner.hidden = false;
  clearTimeout(showError._timer);
  showError._timer = setTimeout(() => {
    errorBanner.hidden = true;
  }, 5000);
}

function saveHistory() {
  try {
    sessionStorage.setItem(HISTORY_KEY, JSON.stringify(history));
  } catch {
    // sessionStorage may be unavailable (private browsing) — chat still
    // works for the rest of this page load, it just won't survive a reload.
  }
}

function addToHistory(entry) {
  history.push(entry);
  saveHistory();
}

function renderWelcomeMessage() {
  renderMessage({
    role: "bot",
    text:
      "Hi! I'm a Python Q&A chatbot. Ask me about variables, data types, loops, " +
      "functions, OOP, exceptions, file handling, and more — or try one of the " +
      "suggestions below.",
  });
}

async function sendMessage(message) {
  chatInput.value = "";
  setInputDisabled(true);

  renderMessage({ role: "user", text: message });
  addToHistory({ role: "user", text: message });

  showTypingIndicator();

  try {
    const response = await fetch(`${API_BASE}/chat`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ message }),
    });

    hideTypingIndicator();

    if (!response.ok) {
      const body = await response.json().catch(() => null);
      const detail = body?.detail?.[0]?.msg || "That question couldn't be processed.";
      showError(detail);
      const text = "Sorry, I couldn't process that question. Please try rephrasing it.";
      renderMessage({ role: "bot", text, isError: true });
      addToHistory({ role: "bot", text, isError: true });
      return;
    }

    const data = await response.json();
    renderMessage({
      role: "bot",
      text: data.answer,
      codeExample: data.code_example,
      category: data.category,
    });
    addToHistory({
      role: "bot",
      text: data.answer,
      codeExample: data.code_example,
      category: data.category,
    });
  } catch {
    hideTypingIndicator();
    showError("Network error — is the server running?");
    const text = "I couldn't reach the server. Please check your connection and try again.";
    renderMessage({ role: "bot", text, isError: true });
    addToHistory({ role: "bot", text, isError: true });
  } finally {
    setInputDisabled(false);
    chatInput.focus();
  }
}

function renderSuggestions() {
  suggestionsEl.innerHTML = "";
  SUGGESTED_QUESTIONS.forEach((question) => {
    const chip = document.createElement("button");
    chip.type = "button";
    chip.className = "suggestion-chip";
    chip.textContent = question;
    chip.addEventListener("click", () => sendMessage(question));
    suggestionsEl.appendChild(chip);
  });
}

function restoreHistory() {
  try {
    const saved = sessionStorage.getItem(HISTORY_KEY);
    history = saved ? JSON.parse(saved) : [];
  } catch {
    history = [];
  }

  if (history.length === 0) {
    renderWelcomeMessage();
    return;
  }

  history.forEach((entry) => renderMessage(entry));
}

chatForm.addEventListener("submit", (event) => {
  event.preventDefault();
  const message = chatInput.value.trim();
  if (!message) return;
  sendMessage(message);
});

chatInput.addEventListener("keydown", (event) => {
  if (event.key === "Enter") {
    event.preventDefault();
    chatForm.requestSubmit();
  }
});

clearBtn.addEventListener("click", () => {
  history = [];
  saveHistory();
  chatWindow.innerHTML = "";
  renderWelcomeMessage();
});

renderSuggestions();
restoreHistory();
chatInput.focus();
