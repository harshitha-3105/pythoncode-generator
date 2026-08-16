import sys
import io
import traceback
from typing import TypedDict, List, Optional

from flask import Flask, request, render_template_string, jsonify
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_core.tools import tool
from langgraph.graph import StateGraph, START, END
from langchain_google_genai import ChatGoogleGenerativeAI
import os

app = Flask(__name__)

# ==========================================
# 1. LLM INITIALIZATION
# ==========================================

api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    raise ValueError("GEMINI_API_KEY environment variable is not set.")

llm_flash = ChatGoogleGenerativeAI(
    model=os.getenv("GEMINI_MODEL", "gemini-3.1-flash-lite-preview"),
    google_api_key=api_key,
    temperature=0
)

llm = llm_flash

# ==========================================
# 2. STATE DEFINITION
# ==========================================

class CrewState(TypedDict, total=False):
    messages: List[BaseMessage]
    next_step: Optional[str]
    code: Optional[str]
    report: Optional[str]

# ==========================================
# 3. TOOLS
# ==========================================

@tool
def run_python_code(code: str) -> str:
    """Execute Python code and return the output or error."""

    if not isinstance(code, str):
        code = str(code)

    clean_code = (
        code.replace("```python", "")
        .replace("```", "")
        .strip()
    )

    old_stdout = sys.stdout
    new_stdout = io.StringIO()
    sys.stdout = new_stdout

    try:
        local_scope = {}
        exec(clean_code, {}, local_scope)
        result = new_stdout.getvalue()
    except Exception:
        result = f"Execution Error:\n{traceback.format_exc()}"
    finally:
        sys.stdout = old_stdout

    return result.strip() if result.strip() else "Success (no terminal output)"


@tool
def generate_test_cases(task_description: str) -> str:
    """Generate 3 to 5 specific test scenarios."""

    prompt = f"""
You are a Senior QA Engineer.

Generate 3 to 5 highly specific test scenarios
for this Python coding task:

{task_description}

Include:
- Normal cases
- Edge cases
- Invalid input cases where appropriate

Return only a numbered list.
"""

    response = llm_flash.invoke(prompt)
    content = response.content

    if isinstance(content, list):
        text_parts = []
        for item in content:
            if isinstance(item, dict):
                text_parts.append(item.get("text", ""))
            else:
                text_parts.append(str(item))
        return "\n".join(text_parts).strip()

    return str(content).strip()

# ==========================================
# 4. GRAPH NODES
# ==========================================

def real_time_developer(state: CrewState):
    task = state["messages"][-1].content

    dev_prompt = f"""
Write a clean Python script to solve this task:

{task}

Rules:
- Return ONLY Python code.
- Do not use Markdown.
- Do not use triple backticks.
- Keep the code simple and executable.
"""

    response = llm_flash.invoke(dev_prompt)
    content = response.content

    if isinstance(content, list):
        text_parts = []
        for item in content:
            if isinstance(item, dict):
                text_parts.append(item.get("text", ""))
            else:
                text_parts.append(str(item))
        code_str = "\n".join(text_parts).strip()
    else:
        code_str = str(content).strip()

    code_str = (
        code_str.replace("```python", "")
        .replace("```", "")
        .strip()
    )

    if not code_str:
        raise ValueError("Gemini returned empty code.")

    return {"code": code_str}


def real_time_tester(state: CrewState):
    task = state["messages"][-1].content

    test_cases = generate_test_cases.invoke(task)
    execution_result = run_python_code.invoke({"code": state["code"]})

    report = (
        "### EXECUTION OUTPUT:\n"
        f"{execution_result}\n\n"
        "### TEST SCENARIOS EVALUATED:\n"
        f"{test_cases}"
    )

    return {"report": report}


# ==========================================
# 5. GRAPH CONSTRUCTION
# ==========================================

rt_workflow = StateGraph(CrewState)

rt_workflow.add_node("developer", real_time_developer)
rt_workflow.add_node("tester", real_time_tester)

rt_workflow.add_edge(START, "developer")
rt_workflow.add_edge("developer", "tester")
rt_workflow.add_edge("tester", END)

rt_app = rt_workflow.compile()

# ==========================================
# 6. WEB INTERFACE
# ==========================================

HTML = """
<!DOCTYPE html>
<html>
<head>
    <title>AI Coding Crew</title>
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <style>
        body {
            font-family: Arial, sans-serif;
            max-width: 900px;
            margin: 40px auto;
            padding: 20px;
            background: #f5f7fb;
        }
        h1 { text-align: center; }
        textarea {
            width: 100%;
            min-height: 120px;
            padding: 12px;
            box-sizing: border-box;
            border-radius: 8px;
            border: 1px solid #ccc;
            font-size: 16px;
        }
        button {
            margin-top: 12px;
            padding: 12px 22px;
            border: 0;
            border-radius: 8px;
            background: #222;
            color: white;
            cursor: pointer;
            font-size: 16px;
        }
        .card {
            background: white;
            padding: 20px;
            margin-top: 20px;
            border-radius: 10px;
            box-shadow: 0 2px 10px rgba(0,0,0,.08);
        }
        pre {
            white-space: pre-wrap;
            background: #111;
            color: #eee;
            padding: 15px;
            border-radius: 8px;
            overflow-x: auto;
        }
        .loading { display: none; margin-top: 15px; }
    </style>
</head>
<body>
    <h1>🤖 AI Coding Crew</h1>

    <div class="card">
        <p>Enter a Python coding task:</p>
        <textarea id="task" placeholder="Example: Write a Python program to generate Fibonacci series up to 10 terms"></textarea>
        <button onclick="runAgent()">Generate & Test</button>
        <div id="loading" class="loading">⏳ Developer and Tester are working...</div>
    </div>

    <div id="result"></div>

<script>
async function runAgent() {
    const task = document.getElementById("task").value.trim();
    const result = document.getElementById("result");
    const loading = document.getElementById("loading");

    if (!task) {
        alert("Please enter a coding task.");
        return;
    }

    loading.style.display = "block";
    result.innerHTML = "";

    try {
        const response = await fetch("/run", {
            method: "POST",
            headers: {"Content-Type": "application/json"},
            body: JSON.stringify({task: task})
        });

        const data = await response.json();

        if (!response.ok) {
            throw new Error(data.error || "Request failed");
        }

        result.innerHTML = `
            <div class="card">
                <h2>👨‍💻 Generated Code</h2>
                <pre>${escapeHtml(data.code)}</pre>
            </div>
            <div class="card">
                <h2>🧪 Test Report</h2>
                <pre>${escapeHtml(data.report)}</pre>
            </div>
        `;
    } catch (error) {
        result.innerHTML = `
            <div class="card">
                <h2>❌ Error</h2>
                <pre>${escapeHtml(error.message)}</pre>
            </div>
        `;
    } finally {
        loading.style.display = "none";
    }
}

function escapeHtml(text) {
    return String(text)
        .replaceAll("&", "&amp;")
        .replaceAll("<", "&lt;")
        .replaceAll(">", "&gt;")
        .replaceAll('"', "&quot;")
        .replaceAll("'", "&#039;");
}
</script>
</body>
</html>
"""

@app.route("/", methods=["GET"])
def home():
    return render_template_string(HTML)

@app.route("/run", methods=["POST"])
def run_task():
    try:
        data = request.get_json(silent=True) or {}
        task = str(data.get("task", "")).strip()

        if not task:
            return jsonify({"error": "Please provide a coding task."}), 400

        state = {
            "messages": [HumanMessage(content=task)]
        }

        result = rt_app.invoke(
            state,
            config={"recursion_limit": 20}
        )

        return jsonify({
            "code": result.get("code", ""),
            "report": result.get("report", "")
        })

    except Exception as e:
        return jsonify({
            "error": str(e)
        }), 500

if __name__ == "__main__":
    port = int(os.getenv("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
