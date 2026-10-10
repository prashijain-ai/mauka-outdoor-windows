# Turns the predictions into a short, friendly message.
#
# A small Gemma model running on your own laptop (through Ollama) writes it, but it
# only gets numbers we already worked out. If Ollama isn't running, we fall back to
# a simple template, and the app tells you which one wrote the text.

import os
import re

import requests

DEFAULT_MODEL = os.getenv("MAUKA_MODEL", "gemma3:1b")
LANGUAGES = ["English", "Hinglish", "Hindi"]


def ollama_url():
    host = os.getenv("OLLAMA_HOST", "http://localhost:11434")
    return host if host.startswith("http") else "http://" + host


def show_hour(h):
    return f"{h % 24:02d}:00"


def find_windows(hours, probs, threshold=0.5, top=3):
    # a "window" is a run of hours in a row that are all above the threshold
    runs = []
    current = []
    for h, p in zip(hours, probs):
        if p >= threshold and (not current or h == current[-1][0] + 1):
            current.append((int(h), float(p)))
        else:
            if current:
                runs.append(current)
            current = [(int(h), float(p))] if p >= threshold else []
    if current:
        runs.append(current)

    windows = []
    for run in runs:
        windows.append({
            "start": run[0][0],
            "end": run[-1][0] + 1,
            "hours": len(run),
            "mean_p": sum(p for _, p in run) / len(run),
        })
    # prefer windows of at least 2 hours (a single lucky hour isn't much of a window),
    # then put the most confident one first
    longer = [w for w in windows if w["hours"] >= 2]
    pool = longer if longer else windows
    pool.sort(key=lambda w: (w["mean_p"], w["hours"]), reverse=True)
    return pool[:top]


def build_facts(city, windows, aqi_today, feels_like_max_today):
    return {
        "city": city,
        "windows": [{"from": show_hour(w["start"]), "to": show_hour(w["end"]),
                     "chance_percent": round(w["mean_p"] * 100)} for w in windows],
        "aqi_today_avg": round(aqi_today),
        "feels_like_max_today_c": round(feels_like_max_today),
    }


def fallback_nudge(facts, language="English"):
    windows = facts["windows"]

    if not windows:
        messages = {
            "English": "Tomorrow looks hard for being outside all day: air quality, heat or rain "
                       "keeps getting in the way. Keep it short, or plan something indoors.",
            "Hinglish": "Kal poore din bahar ka mahaul theek nahi lag raha: hawa, garmi ya baarish "
                        "dikkat degi. Bahar kam time bitao ya kuch indoor plan karo.",
            "Hindi": "कल पूरे दिन बाहर का माहौल ठीक नहीं दिख रहा: हवा, गर्मी या बारिश दिक्कत देगी। "
                     "बाहर कम समय बिताएँ या कुछ घर के अंदर करें।",
        }
        return messages[language]

    best = windows[0]
    city = facts["city"]
    time_range = f"{best['from']} to {best['to']}"
    chance = best["chance_percent"]
    backup = ""
    if len(windows) > 1:
        backup = f" Backup: {windows[1]['from']} to {windows[1]['to']}."

    messages = {
        "English": f"Your best window tomorrow in {city} is {time_range} "
                   f"(about {chance}% chance it stays clear and comfortable).{backup} "
                   "Step outside for ten minutes.",
        "Hinglish": f"Kal {city} mein bahar jaane ka sabse accha time {best['from']} se {best['to']} hai "
                    f"(lagbhag {chance}% chance ki hawa saaf aur mausam theek rahega).{backup} "
                    "Das minute bahar nikalke dekho.",
        "Hindi": f"कल {city} में बाहर निकलने का सबसे अच्छा समय {best['from']} से {best['to']} है "
                 f"(लगभग {chance}% संभावना कि हवा साफ़ और मौसम ठीक रहेगा)।{backup} "
                 "दस मिनट बाहर निकलकर देखें।",
    }
    return messages[language]


def ollama_status(model=DEFAULT_MODEL):
    # returns (is Ollama running, is the model downloaded)
    try:
        r = requests.get(ollama_url() + "/api/tags", timeout=2)
        r.raise_for_status()
        installed = [m.get("name", "") for m in r.json().get("models", [])]
    except Exception:
        return False, False

    wanted = model if ":" in model else model + ":latest"
    return True, wanted in installed


def make_prompt(facts, language):
    style = {
        "English": "English",
        "Hinglish": "Hinglish (Hindi written in English letters, casual and friendly)",
        "Hindi": "Hindi (Devanagari script)",
    }[language]
    best = facts["windows"][0]
    others = [f"{w['from']} to {w['to']}" for w in facts["windows"][1:]]
    return (
        "Write a short, friendly nudge of exactly 2 sentences that encourages someone to spend "
        "a few minutes outside TOMORROW.\n"
        f"Write it in {style}.\n"
        "Rules:\n"
        f"- Name the best window using exactly these times: {best['from']} to {best['to']} "
        "(keep the 24-hour format with the colon).\n"
        f"- Say there is about a {best['chance_percent']}% chance it will be a good time.\n"
        "- Say 'tomorrow', never 'today'.\n"
        "- Do not mention the sun, sunshine, mood, health, weather or air quality.\n"
        "- Do not add any facts that are not listed here.\n\n"
        f"City: {facts['city']}\n"
        f"Best window: {best['from']} to {best['to']} ({best['chance_percent']}% chance)\n"
        f"Other windows: {', '.join(others) if others else 'none'}\n\n"
        "Nudge:"
    )


# words a small model likes to invent, which we never gave it
BANNED = re.compile(r"\b(sun|sunny|sunshine|mood|health|healthy|medical|doctor)\b", re.IGNORECASE)


def looks_grounded(text, facts):
    # the text must use the exact best window times and avoid made-up claims
    if not facts["windows"]:
        return False
    best = facts["windows"][0]
    if best["from"] not in text or best["to"] not in text:
        return False
    return not BANNED.search(text)


def ask_ollama(prompt, model, temperature):
    try:
        r = requests.post(ollama_url() + "/api/generate", json={
            "model": model,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": temperature, "num_predict": 160},
        }, timeout=120)
        r.raise_for_status()
        return r.json().get("response", "").strip().strip('"')
    except Exception:
        return ""


def generate_nudge(facts, language="English", model=DEFAULT_MODEL):
    # returns (text, who wrote it)
    if not facts["windows"]:
        return fallback_nudge(facts, language), "template (no good window to write about)"

    running, installed = ollama_status(model)
    if not (running and installed):
        return fallback_nudge(facts, language), "template (local model not running)"

    prompt = make_prompt(facts, language)
    for temperature in (0.4, 0.2):  # one retry with a calmer setting
        text = ask_ollama(prompt, model, temperature)
        if 20 <= len(text) <= 700 and looks_grounded(text, facts):
            return text, f"{model} via Ollama (local), checked against the numbers"
    return fallback_nudge(facts, language), "template (the local model's text didn't match the numbers)"
