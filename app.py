import sys
import io
import traceback
import re
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
# 3. PYTHON-ONLY GUARDRAIL
# ==========================================

PYTHON_ONLY_MESSAGE = (
    "I can't answer this. I can only generate Python code."
)


def is_python_coding_task(task: str) -> bool:
    """
    Allow programming/code requests and force them to Python.
    Reject explicitly requested non-Python languages and
    reject non-programming/general questions.
    """

    text = task.strip().lower()

    if not text:
        return False

    # --------------------------------------
    # Explicitly requested non-Python languages
    # --------------------------------------

    non_python_patterns = [
        r"\bc\+\+\b",
        r"\bcpp\b",
        r"\bc plus plus\b",
        r"\bc language\b",
        r"\bin c\b",
        r"\busing c\b",
        r"\bjava\b",
        r"\bjavascript\b",
        r"\btypescript\b",
        r"\brust\b",
        r"\bgolang\b",
        r"\bgo language\b",
        r"\bkotlin\b",
        r"\bswift\b",
        r"\bphp\b",
        r"\bruby\b",
        r"\bc#\b",
        r"\bc sharp\b",
        r"\bmatlab\b",
        r"\bscala\b",
        r"\bperl\b",
        r"\bdart\b",
        r"\blua\b",
        r"\bfortran\b",
    ]

    for pattern in non_python_patterns:
        if re.search(pattern, text):
            return False

    # --------------------------------------
    # Explicit Python request
    # --------------------------------------

    if re.search(r"\bpython\b", text):
        return True

    # --------------------------------------
    # General programming intent
    # --------------------------------------

    programming_words = [
        "code",
        "program",
        "script",
        "function",
        "class",
        "algorithm",
        "implement",
        "implementation",
        "programming",
        "coding",
        "compile",
        "debug",
        "debugging",
        "syntax",
        "loop",
        "variable",
        "input",
        "output",
        "calculator",
        "game",
        "app",
        "application",
        "automation",
        "sort",
        "sorting",
        "search",
        "searching",
        "array",
        "list",
        "dictionary",
        "tuple",
        "stack",
        "queue",
        "matrix",
        "recursion",
        "factorial",
        "fibonacci",
        "palindrome",
        "prime number",
        "prime numbers",
    ]

    if any(word in text for word in programming_words):
        return True

    # --------------------------------------
    # Natural-language coding requests
    # --------------------------------------

    coding_phrases = [
        r"\bwrite\b.*\bprogram\b",
        r"\bwrite\b.*\bcode\b",
        r"\bwrite\b.*\bscript\b",
        r"\bcreate\b.*\bprogram\b",
        r"\bcreate\b.*\bcode\b",
        r"\bcreate\b.*\bscript\b",
        r"\bmake\b.*\bprogram\b",
        r"\bmake\b.*\bcode\b",
        r"\bmake\b.*\bscript\b",
        r"\bgenerate\b.*\bprogram\b",
        r"\bgenerate\b.*\bcode\b",
        r"\bgenerate\b.*\bscript\b",
        r"\bwrite\b.*\bseries\b",
        r"\bgenerate\b.*\bseries\b",
    ]

    return any(
        re.search(pattern, text)
        for pattern in coding_phrases
    )


def guardrail_response(task: str) -> Optional[str]:
    if not is_python_coding_task(task):
        return PYTHON_ONLY_MESSAGE

    return None


# ==========================================
# 4. TOOLS
# ==========================================

@tool
def run_python_code(code: str) -> str:
    """Execute Python code and return output or error."""

    if not isinstance(code, str):
        code = str(code)

    clean_code = (
        code.replace("```python", "")
        .replace("```", "")
        .strip()
    )

    # Never execute the rejection message
    if clean_code == PYTHON_ONLY_MESSAGE:
        return PYTHON_ONLY_MESSAGE

    old_stdout = sys.stdout
    new_stdout = io.StringIO()
    sys.stdout = new_stdout

    try:
        local_scope = {}

        exec(
            clean_code,
            {},
            local_scope
        )

        result = new_stdout.getvalue()

    except Exception:
        result = (
            "Execution Error:\n"
            + traceback.format_exc()
        )

    finally:
        sys.stdout = old_stdout

    return (
        result.strip()
        if result.strip()
        else "Success (no terminal output)"
    )


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
                text_parts.append(
                    item.get("text", "")
                )
            else:
                text_parts.append(str(item))

        return "\n".join(text_parts).strip()

    return str(content).strip()


# ==========================================
# 5. GRAPH NODES
# ==========================================

def real_time_developer(state: CrewState):

    task = state["messages"][-1].content

    # HARD GUARDRAIL
    rejection = guardrail_response(task)

    if rejection:
        return {
            "code": rejection,
            "report": rejection
        }

    # --------------------------------------
    # Python-only developer prompt
    # --------------------------------------

    dev_prompt = f"""
You are a STRICT Python-only code generator.

User request:
{task}

Rules:

1. Generate ONLY Python code.
2. Never generate C.
3. Never generate C++.
4. Never generate Java.
5. Never generate JavaScript.
6. Never generate TypeScript.
7. Never generate Rust.
8. Never generate Go.
9. Never generate Kotlin.
10. Never generate Swift.
11. Never generate PHP.
12. Never generate Ruby.
13. Never generate C#.
14. Never generate MATLAB.
15. Never answer general knowledge questions.
16. Never answer conversational questions.
17. Never answer questions unrelated to programming.
18. Do not explain the answer.
19. Do not use Markdown.
20. Do not use triple backticks.
21. Return ONLY valid Python code.
"""

    response = llm_flash.invoke(dev_prompt)

    content = response.content

    if isinstance(content, list):

        text_parts = []

        for item in content:

            if isinstance(item, dict):
                text_parts.append(
                    item.get("text", "")
                )
            else:
                text_parts.append(str(item))

        code_str = "\n".join(text_parts).strip()

    else:
        code_str = str(content).strip()

    code_str = (
        code_str
        .replace("```python", "")
        .replace("```", "")
        .strip()
    )

    if not code_str:
        raise ValueError(
            "Gemini returned empty code."
        )

    return {
        "code": code_str
    }


def real_time_tester(state: CrewState):

    task = state["messages"][-1].content

    # Do not test rejected requests
    rejection = guardrail_response(task)

    if rejection:
        return {
            "report": rejection
        }

    test_cases = generate_test_cases.invoke(
        task
    )

    execution_result = run_python_code.invoke(
        {
            "code": state["code"]
        }
    )

    report = (
        "### EXECUTION OUTPUT:\n"
        f"{execution_result}\n\n"
        "### TEST SCENARIOS EVALUATED:\n"
        f"{test_cases}"
    )

    return {
        "report": report
    }


# ==========================================
# 6. GRAPH CONSTRUCTION
# ==========================================

rt_workflow = StateGraph(CrewState)

rt_workflow.add_node(
    "developer",
    real_time_developer
)

rt_workflow.add_node(
    "tester",
    real_time_tester
)

rt_workflow.add_edge(
    START,
    "developer"
)

rt_workflow.add_edge(
    "developer",
    "tester"
)

rt_workflow.add_edge(
    "tester",
    END
)

rt_app = rt_workflow.compile()


# ==========================================
# 7. WEB INTERFACE
# ==========================================

HTML = """
<!DOCTYPE html>

<html>

<head>

    <title>AI Coding Crew</title>

    <meta
        name="viewport"
        content="width=device-width, initial-scale=1"
    >

    <style>

        body {
            font-family: Arial, sans-serif;
            max-width: 900px;
            margin: 40px auto;
            padding: 20px;
            background: #f5f7fb;
        }

        h1 {
            text-align: center;
        }

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

        .loading {
            display: none;
            margin-top: 15px;
        }

    </style>

</head>

<body>

    <h1>🤖 AI Coding Crew</h1>

    <div class="card">

        <p>
            Enter a Python coding task only:
        </p>

        <textarea
            id="task"
            placeholder="Example: Write a Fibonacci series"
        ></textarea>

        <button onclick="runAgent()">
            Generate & Test
        </button>

        <div
            id="loading"
            class="loading"
        >
            ⏳ Developer and Tester are working...
        </div>

    </div>

    <div id="result"></div>


<script>

async function runAgent() {

    const task =
        document
        .getElementById("task")
        .value
        .trim();

    const result =
        document
        .getElementById("result");

    const loading =
        document
        .getElementById("loading");


    if (!task) {

        alert(
            "Please enter a Python coding task."
        );

        return;
    }


    loading.style.display = "block";

    result.innerHTML = "";


    try {

        const response =
            await fetch(
                "/run",
                {
                    method: "POST",

                    headers: {
                        "Content-Type":
                            "application/json"
                    },

                    body: JSON.stringify({
                        task: task
                    })
                }
            );


        const data =
            await response.json();


        if (!response.ok) {

            throw new Error(
                data.error ||
                "Request failed"
            );

        }


        result.innerHTML = `

            <div class="card">

                <h2>
                    👨‍💻 Generated Code
                </h2>

                <pre>
${escapeHtml(data.code)}
                </pre>

            </div>


            <div class="card">

                <h2>
                    🧪 Test Report
                </h2>

                <pre>
${escapeHtml(data.report)}
                </pre>

            </div>

        `;


    } catch (error) {

        result.innerHTML = `

            <div class="card">

                <h2>
                    ❌ Error
                </h2>

                <pre>
${escapeHtml(error.message)}
                </pre>

            </div>

        `;

    } finally {

        loading.style.display = "none";

    }

}


function escapeHtml(text) {

    return String(text)

        .replaceAll(
            "&",
            "&amp;"
        )

        .replaceAll(
            "<",
            "&lt;"
        )

        .replaceAll(
            ">",
            "&gt;"
        )

        .replaceAll(
            '"',
            "&quot;"
        )

        .replaceAll(
            "'",
            "&#039;"
        );

}

</script>

</body>

</html>
"""


# ==========================================
# 8. FLASK ROUTES
# ==========================================

@app.route(
    "/",
    methods=["GET"]
)
def home():

    return render_template_string(
        HTML
    )


@app.route(
    "/run",
    methods=["POST"]
)
def run_task():

    try:

        data = (
            request
            .get_json(
                silent=True
            )
            or {}
        )

        task = str(
            data.get(
                "task",
                ""
            )
        ).strip()


        if not task:

            return jsonify(
                {
                    "error":
                    "Please provide a coding task."
                }
            ), 400


        # ==================================
        # HARD API-LEVEL GUARDRAIL
        # ==================================

        rejection = guardrail_response(
            task
        )

        if rejection:

            return jsonify(
                {
                    "code": rejection,
                    "report": rejection
                }
            )


        state = {

            "messages": [
                HumanMessage(
                    content=task
                )
            ]

        }


        result = rt_app.invoke(

            state,

            config={
                "recursion_limit": 20
            }

        )


        return jsonify(

            {
                "code":
                    result.get(
                        "code",
                        ""
                    ),

                "report":
                    result.get(
                        "report",
                        ""
                    )
            }

        )


    except Exception as e:

        return jsonify(

            {
                "error":
                    str(e)
            }

        ), 500


# ==========================================
# 9. START SERVER
# ==========================================

if __name__ == "__main__":

    port = int(
        os.getenv(
            "PORT",
            5000
        )
    )

    app.run(
        host="0.0.0.0",
        port=port
    )
