# AI-Virtual-Assistant
from __future__ import annotations
import os
import re
import sys
import json
import time
import math
import uuid
import psutil
import queue
import atexit
import numexpr as ne
import requests
import webbrowser
import threading
import datetime as dt
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

# --- Third‑party libs (optional voice) ---
try:
    import speech_recognition as sr  # for voice input
except Exception:
    sr = None
try:
    import pyttsx3  # offline TTS
except Exception:
    pyttsx3 = None

from duckduckgo_search import DDGS
import wikipedia
from dotenv import load_dotenv
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.prompt import Prompt
from rich.markdown import Markdown

# OpenAI new SDK (>=1.0)
try:
    from openai import OpenAI
except Exception:
    OpenAI = None

console = Console()
load_dotenv()

# ---------------------- Utility & Memory ----------------------
APP_NAME = "Advanced VAI"
MEM_PATH = os.path.join(os.path.dirname(__file__), "assistant_memory.json")


def _now_iso():
    return dt.datetime.now().isoformat(timespec="seconds")


class Memory:
    def __init__(self, path: str):
        self.path = path
        self.data = {"facts": [], "dialog": []}
        self._load()

    def _load(self):
        if os.path.exists(self.path):
            try:
                with open(self.path, "r", encoding="utf-8") as f:
                    self.data = json.load(f)
            except Exception:
                console.print("[yellow]Memory file unreadable, starting fresh.[/]")
                self.data = {"facts": [], "dialog": []}

    def save(self):
        try:
            with open(self.path, "w", encoding="utf-8") as f:
                json.dump(self.data, f, ensure_ascii=False, indent=2)
        except Exception as e:
            console.print(f"[red]Failed to save memory: {e}[/]")

    def add_fact(self, fact: str):
        if fact and fact not in self.data["facts"]:
            self.data["facts"].append(fact)
            self.save()

    def add_dialog(self, role: str, content: str):
        self.data["dialog"].append({"role": role, "content": content, "ts": _now_iso()})
        # Cap dialog memory to last 100 turns
        self.data["dialog"] = self.data["dialog"][-200:]
        self.save()


memory = Memory(MEM_PATH)

# ---------------------- Tooling System -----------------------
ToolFn = Callable[[str], str]

@dataclass
class Tool:
    name: str
    description: str
    usage: str
    fn: ToolFn


class ToolRegistry:
    def __init__(self):
        self.tools: Dict[str, Tool] = {}

    def register(self, name: str, description: str, usage: str):
        def decorator(func: ToolFn):
            self.tools[name] = Tool(name, description, usage, func)
            return func
        return decorator

    def call(self, name: str, arg: str) -> str:
        if name not in self.tools:
            return f"Tool '{name}' not found."
        try:
            return self.tools[name].fn(arg)
        except Exception as e:
            return f"[tool:{name}] error: {e}"

    def help_text(self) -> str:
        lines = []
        for t in self.tools.values():
            lines.append(f"- {t.name}: {t.description}\n  usage: {t.usage}")
        return "\n".join(lines)


tools = ToolRegistry()

# ---------------------- LLM Wrapper --------------------------
class LLM:
    def __init__(self):
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            console.print("[yellow]OPENAI_API_KEY not set; LLM responses disabled. Use tools only.[/]")
            self.client = None
        else:
            if OpenAI is None:
                console.print("[red]openai>=1 package missing. Install with `pip install -U openai`." )
                self.client = None
            else:
                self.client = OpenAI()

    def chat(self, prompt: str, sys_prompt: Optional[str] = None, history: Optional[List[Dict[str, str]]] = None) -> str:
        if not self.client:
            return "(LLM unavailable) " + prompt
        messages = []
        if sys_prompt:
            messages.append({"role": "system", "content": sys_prompt})
        if history:
            messages.extend(history)
        messages.append({"role": "user", "content": prompt})
        try:
            resp = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=messages,
                temperature=0.3,
                max_tokens=600,
            )
            return resp.choices[0].message.content.strip()
        except Exception as e:
            return f"(LLM error) {e}"


llm = LLM()

# ---------------------- Helper functions ---------------------

def strip_code_fences(text: str) -> str:
    return re.sub(r"^```[a-zA-Z]*\n|\n```$", "", text.strip())


def require_internet() -> None:
    try:
        requests.get("https://duckduckgo.com", timeout=5)
    except Exception:
        raise RuntimeError("No internet connection for this tool.")


# ---------------------- Built-in Tools -----------------------
@tools.register(
    name="search",
    description="Web search via DuckDuckGo; returns top snippets.",
    usage="search: your query"
)
def tool_search(q: str) -> str:
    require_internet()
    q = q.strip() or ""
    if not q:
        return "Provide a query after 'search:'"
    with DDGS() as ddgs:
        results = ddgs.text(q, max_results=5)
    if not results:
        return "No results."
    lines = []
    for r in results:
        title = r.get("title")
        href = r.get("href")
        body = r.get("body")
        lines.append(f"• {title}\n  {body}\n  {href}")
    return "\n".join(lines)


@tools.register(
    name="wiki",
    description="Get a concise summary from Wikipedia.",
    usage="wiki: topic"
)
def tool_wiki(topic: str) -> str:
    topic = topic.strip()
    if not topic:
        return "Provide a topic after 'wiki:'"
    try:
        wikipedia.set_lang("en")
        summary = wikipedia.summary(topic, sentences=5, auto_suggest=True, redirect=True)
        return summary
    except Exception as e:
        return f"Wikipedia error: {e}"


@tools.register(
    name="calc",
    description="Safe numerical calculator using numexpr (supports + - * / ** sqrt sin cos etc).",
    usage="calc: expression"
)
def tool_calc(expr: str) -> str:
    expr = expr.strip()
    if not expr:
        return "Provide an expression after 'calc:'"
    try:
        # Guard: only allow numbers and safe symbols
        if not re.fullmatch(r"[0-9eEpi\.\+\-\*/\(\)\^\s,]+", expr.replace("**", "^")):
            # allow a subset of names
            allowed_names = {"pi": math.pi, "e": math.e}
            return str(eval(expr, {"__builtins__": {}}, allowed_names))
        expr = expr.replace("^", "**")
        val = ne.evaluate(expr)
        return str(val)
    except Exception as e:
        return f"Calc error: {e}"


@tools.register(
    name="weather",
    description="Current weather by city using Open‑Meteo + Nominatim geocoding.",
    usage="weather: City, Country"
)
def tool_weather(place: str) -> str:
    require_internet()
    place = place.strip()
    if not place:
        return "Provide a place after 'weather:'"
    try:
        geo = requests.get(
            "https://nominatim.openstreetmap.org/search",
            params={"q": place, "format": "json", "limit": 1},
            headers={"User-Agent": APP_NAME},
            timeout=10,
        ).json()
        if not geo:
            return "Location not found."
        lat = float(geo[0]["lat"]); lon = float(geo[0]["lon"])
        wx = requests.get(
            "https://api.open-meteo.com/v1/forecast",
            params={"latitude": lat, "longitude": lon, "current_weather": True},
            timeout=10,
        ).json()
        cur = wx.get("current_weather", {})
        if not cur:
            return "Weather unavailable."
        return (
            f"Weather for {place} (lat {lat:.2f}, lon {lon:.2f})\n"
            f"• Temp: {cur.get('temperature')}°C\n"
            f"• Wind: {cur.get('windspeed')} km/h\n"
            f"• Code: {cur.get('weathercode')} at {cur.get('time')}"
        )
    except Exception as e:
        return f"Weather error: {e}"


COMMON_APPS = {
    "chrome": "chrome",
    "edge": "msedge" if sys.platform.startswith("win") else "",
    "notepad": "notepad" if sys.platform.startswith("win") else "",
    "calculator": "calc" if sys.platform.startswith("win") else "",
    "vscode": "code",
}


@tools.register(
    name="open",
    description="Open a URL in default browser or launch a common app (chrome, vscode, notepad, calculator).",
    usage="open: https://example.com  OR  open: vscode"
)
def tool_open(target: str) -> str:
    t = target.strip()
    if not t:
        return "Provide a URL or app name after 'open:'"
    if re.match(r"^https?://", t):
        webbrowser.open(t)
        return f"Opened URL: {t}"
    app = COMMON_APPS.get(t.lower())
    if app:
        try:
            if sys.platform.startswith("win"):
                os.startfile(app)  # type: ignore
            else:
                os.system(f"open {app}" if sys.platform == "darwin" else f"{app} &")
            return f"Launched app: {t}"
        except Exception as e:
            return f"Failed to launch {t}: {e}"
    return "Unknown app. Try a URL or one of: " + ", ".join(COMMON_APPS)


@tools.register(
    name="files",
    description="Search files by name pattern in a directory (default: current).",
    usage="files: *.pdf in C:/Users/You/Documents"
)
def tool_files(arg: str) -> str:
    arg = arg.strip()
    if not arg:
        pattern, root = "*", os.getcwd()
    else:
        m = re.match(r"(.+?)\s+in\s+(.+)$", arg)
        if m:
            pattern, root = m.group(1).strip(), os.path.expanduser(m.group(2).strip())
        else:
            pattern, root = arg, os.getcwd()
    import fnmatch
    hits = []
    for base, _, files in os.walk(root):
        for f in files:
            if fnmatch.fnmatch(f, pattern):
                hits.append(os.path.join(base, f))
                if len(hits) >= 50:
                    break
    if not hits:
        return "No matching files."
    return "\n".join(hits[:50])


@tools.register(
    name="sysinfo",
    description="Show CPU, RAM, and battery status.",
    usage="sysinfo:"
)
def tool_sysinfo(_: str) -> str:
    cpu = psutil.cpu_percent(interval=0.5)
    ram = psutil.virtual_memory().percent
    batt = psutil.sensors_battery()
    btxt = f"Battery: {batt.percent}%" if batt else "Battery: n/a"
    return f"CPU: {cpu}%\nRAM: {ram}%\n{btxt}"


# ---------------------- TTS & STT ----------------------------
class Speaker:
    def __init__(self):
        self.engine = None
        if pyttsx3:
            try:
                self.engine = pyttsx3.init()
            except Exception:
                self.engine = None

    def say(self, text: str):
        if not text:
            return
        if self.engine:
            self.engine.say(text)
            self.engine.runAndWait()
        else:
            # fallback: print only
            pass


class Listener:
    def __init__(self):
        self.available = sr is not None
        self.recognizer = sr.Recognizer() if self.available else None

    def listen_once(self) -> Optional[str]:
        if not self.available:
            return None
        try:
            with sr.Microphone() as source:
                console.print("[dim]Listening... (Ctrl+C to stop)[/]")
                self.recognizer.adjust_for_ambient_noise(source, duration=0.5)
                audio = self.recognizer.listen(source, timeout=5, phrase_time_limit=15)
            try:
                text = self.recognizer.recognize_google(audio)
                return text
            except sr.UnknownValueError:
                return None
        except Exception:
            return None


speaker = Speaker()
listener = Listener()

# ---------------------- Intent Router ------------------------
COMMAND_PATTERNS = [
    (r"^help$", "help"),
    (r"^tools$", "tools"),
    (r"^search:(.+)$", "search"),
    (r"^wiki:(.+)$", "wiki"),
    (r"^calc:(.+)$", "calc"),
    (r"^weather:(.+)$", "weather"),
    (r"^open:(.+)$", "open"),
    (r"^files:(.*)$", "files"),
    (r"^sysinfo:?$", "sysinfo"),
]

SYSTEM_PROMPT = (
    "You are a concise, helpful AI assistant. If the user asks for actions that match a tool,\n"
    "produce a short answer and, where appropriate, suggest a matching command (e.g., search:, calc:).\n"
    "When answering questions, be accurate, cite sources if provided by tools, and keep replies under 10 sentences unless asked."
)


def route_intent(text: str) -> Tuple[str, Optional[str]]:
    t = text.strip()
    for pat, name in COMMAND_PATTERNS:
        m = re.match(pat, t, flags=re.IGNORECASE)
        if m:
            arg = m.group(1).strip() if m.groups() else ""
            return name, arg
    # Natural language: try quick heuristics to map to a tool
    low = t.lower()
    if low.startswith("open "):
        return "open", t.split(" ", 1)[1]
    if any(k in low for k in ["search", "look up", "find on web"]):
        return "search", t
    if any(k in low for k in ["weather", "temperature", "forecast"]):
        return "weather", t
    if re.search(r"\d\s*[\+\-\*/^]", t):
        return "calc", t
    if any(k in low for k in ["who is", "what is", "wikipedia"]):
        return "wiki", t
    return "chat", None


# ---------------------- Main Assistant Loop ------------------
class Assistant:
    def __init__(self, use_voice: bool = False):
        self.use_voice = use_voice and listener.available
        if use_voice and not listener.available:
            console.print("[yellow]Voice libraries not available. Falling back to text mode.[/]")
        self.running = True

    def banner(self):
        table = Table.grid(expand=True)
        table.add_column(justify="left")
        table.add_row(f"[bold cyan]{APP_NAME}[/] — type [bold]help[/] for commands.  [dim]{_now_iso()}[/]")
        console.print(Panel.fit(table, border_style="cyan"))

    def help(self):
        md = Markdown(
            """
**Commands**
- `search: query` — Web search (top snippets)
- `wiki: topic` — Wikipedia summary
- `calc: expr` — Calculator (e.g., `2*(3+4)`)
- `weather: City, Country` — Current weather
- `open: URL|app` — Open a website or common app
- `files: pattern in DIR` — Find files
- `sysinfo` — CPU/RAM/Battery
- `tools` — List all tools
- Any other text — Chat with LLM (if configured)
            """
        )
        console.print(md)

    def handle(self, user_text: str) -> str:
        intent, arg = route_intent(user_text)
        if intent == "help":
            self.help()
            return ""
        if intent == "tools":
            return tools.help_text()
        if intent == "chat":
            history = memory.data.get("dialog", [])[-10:]
            history_fmt = [{"role": m["role"], "content": m["content"]} for m in history]
            reply = llm.chat(user_text, SYSTEM_PROMPT, history_fmt)
            return reply
        # Tool call
        res = tools.call(intent, arg or user_text)
        return res

    def speak_if_needed(self, text: str):
        if self.use_voice and text:
            speaker.say(text)

    def loop(self):
        self.banner()
        while self.running:
            try:
                if self.use_voice:
                    heard = listener.listen_once()
                    if not heard:
                        continue
                    user_text = heard
                    console.print(f"[bold]>[/] {user_text}")
                else:
                    user_text = Prompt.ask("[bold green]You[/]")
                if user_text.lower() in {"exit", "quit", ":q", "bye"}:
                    console.print("[dim]Goodbye![/]")
                    break
                memory.add_dialog("user", user_text)
                reply = self.handle(user_text)
                if reply:
                    console.print(Panel.fit(reply, border_style="magenta"))
                    memory.add_dialog("assistant", reply)
                    self.speak_if_needed(reply)
            except KeyboardInterrupt:
                console.print("\n[dim]Interrupted. Type 'quit' to exit.[/]")
            except Exception as e:
                console.print(f"[red]Error: {e}[/]")


# ---------------------- Entry Point --------------------------
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Advanced Virtual AI Assistant")
    parser.add_argument("--voice", action="store_true", help="Enable voice mode (requires microphone & PyAudio)")
    args = parser.parse_args()

    assistant = Assistant(use_voice=args.voice)
    atexit.register(memory.save)
    assistant.loop()
