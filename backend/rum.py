"""Monitoramento de desempenho com usuários reais (RUM): validação, resumo e página HTML.

Sem dependências externas (só biblioteca padrão), para ser fácil de testar.
Os relatórios vêm do app (src/rum.ts): Web Vitals, tempo de troca de tela e latência da API.
Nenhum dado pessoal é coletado: só tempos, rota normalizada e dados do aparelho (largura, rede, instalado ou não).
"""
import html
import json
import math
import re
from typing import Any, Dict, List, Optional

MAX_BODY = 12_000
_PATH_OK = re.compile(r"[^A-Za-z0-9/_:\-.]")

# limites do Google (Core Web Vitals): (bom, ruim). Acima de "ruim" = ruim; até "bom" = bom.
THRESHOLDS = {"lcp": (2500, 4000), "fcp": (1800, 3000), "cls": (0.1, 0.25), "inp": (200, 500), "ttfb": (800, 1800)}
LABELS = {"lcp": "LCP — maior elemento visível", "fcp": "FCP — primeiro conteúdo", "cls": "CLS — estabilidade do layout",
          "inp": "INP — resposta a toques", "ttfb": "TTFB — resposta do servidor"}


def _num(x: Any, lo: float, hi: float) -> Optional[float]:
    if isinstance(x, bool) or not isinstance(x, (int, float)) or math.isnan(x) or math.isinf(x):
        return None
    return min(max(float(x), lo), hi)


def _path(x: Any) -> str:
    return _PATH_OK.sub("?", str(x or "").split("?")[0])[:80] or "/"


def _str(x: Any, n: int) -> str:
    return str(x if x is not None else "")[:n]


def parse_payload(raw: bytes) -> Optional[Dict[str, Any]]:
    """Valida e limpa o relatório. Devolve None se for inválido, grande demais ou vazio."""
    if not raw or len(raw) > MAX_BODY:
        return None
    try:
        d = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        return None
    if not isinstance(d, dict):
        return None

    doc: Dict[str, Any] = {"v": _str(d.get("v"), 20), "sid": _str(d.get("sid"), 16), "path": _path(d.get("path")), "nav": _str(d.get("nav"), 20)}

    vit = d.get("vit")
    if isinstance(vit, dict):
        limits = {"fcp": 120_000, "lcp": 120_000, "inp": 60_000, "ttfb": 60_000, "dcl": 120_000, "load": 300_000, "cls": 10}
        clean = {k: _num(vit.get(k), 0, hi) for k, hi in limits.items()}
        clean = {k: v for k, v in clean.items() if v is not None}
        if clean:
            doc["vit"] = clean

    ctx = d.get("ctx")
    if isinstance(ctx, dict):
        doc["ctx"] = {
            "sa": 1 if ctx.get("sa") == 1 else 0,
            "mob": 1 if ctx.get("mob") == 1 else 0,
            "net": _str(ctx.get("net"), 10),
            "w": int(_num(ctx.get("w"), 0, 10_000) or 0),
            "mem": _num(ctx.get("mem"), 0, 64) or 0,
        }

    routes = []
    for r in (d.get("routes") or [])[:40] if isinstance(d.get("routes"), list) else []:
        ms = _num(r.get("ms"), 0, 60_000) if isinstance(r, dict) else None
        if ms is not None:
            routes.append({"r": _path(r.get("r")), "ms": ms})
    if routes:
        doc["routes"] = routes

    apis = []
    for a in (d.get("api") or [])[:40] if isinstance(d.get("api"), list) else []:
        if not isinstance(a, dict):
            continue
        ms, st = _num(a.get("ms"), 0, 300_000), _num(a.get("s"), 0, 999)
        if ms is None or st is None:
            continue
        item = {"p": _path(a.get("p")), "m": _str(a.get("m"), 8).upper(), "ms": ms, "s": int(st)}
        ttft = _num(a.get("ttft"), 0, 300_000)
        if ttft is not None:
            item["ttft"] = ttft
        apis.append(item)
    if apis:
        doc["api"] = apis

    return doc if ("vit" in doc or "routes" in doc or "api" in doc) else None


def percentile(values: List[float], q: float) -> Optional[float]:
    """Percentil por interpolação linear (q entre 0 e 100)."""
    if not values:
        return None
    s = sorted(values)
    k = (len(s) - 1) * q / 100
    lo, hi = math.floor(k), math.ceil(k)
    return s[lo] if lo == hi else s[lo] + (s[hi] - s[lo]) * (k - lo)


def rating(metric: str, value: Optional[float]) -> str:
    if value is None or metric not in THRESHOLDS:
        return ""
    good, poor = THRESHOLDS[metric]
    return "bom" if value <= good else ("ruim" if value > poor else "regular")


def _stats(values: List[float], digits: int = 0) -> Dict[str, Any]:
    r = lambda x: None if x is None else round(x, digits)
    return {"n": len(values), "p50": r(percentile(values, 50)), "p75": r(percentile(values, 75)), "p95": r(percentile(values, 95))}


def summarize(docs: List[Dict[str, Any]]) -> Dict[str, Any]:
    vit: Dict[str, List[float]] = {}
    routes: Dict[str, List[float]] = {}
    api: Dict[str, Dict[str, List[float]]] = {}
    sessions, installed, mobile, total_api_err = set(), 0, 0, 0
    for d in docs:
        if d.get("sid"):
            sessions.add(d["sid"])
        for k, v in (d.get("vit") or {}).items():
            vit.setdefault(k, []).append(v)
        ctx = d.get("ctx") or {}
        if "vit" in d:  # conta aparelho uma vez por carregamento (o relatório com Web Vitals)
            installed += 1 if ctx.get("sa") == 1 else 0
            mobile += 1 if ctx.get("mob") == 1 else 0
        for r in d.get("routes") or []:
            routes.setdefault(r["r"], []).append(r["ms"])
        for a in d.get("api") or []:
            e = api.setdefault(f'{a["m"]} {a["p"]}', {"ms": [], "ttft": [], "err": []})
            e["ms"].append(a["ms"])
            e["err"].append(1.0 if (a["s"] >= 400 or a["s"] == 0) else 0.0)
            if "ttft" in a:
                e["ttft"].append(a["ttft"])
    loads = len(vit.get("ttfb", [])) or max((len(v) for v in vit.values()), default=0)
    out: Dict[str, Any] = {"reports": len(docs), "sessions": len(sessions), "loads": loads, "installed": installed, "mobile": mobile, "vitals": {}, "routes": [], "api": []}
    for m in ("lcp", "fcp", "cls", "inp", "ttfb"):
        if m in vit:
            st = _stats(vit[m], 3 if m == "cls" else 0)
            st["rating"] = rating(m, st["p75"])
            out["vitals"][m] = st
    for name, vals in sorted(routes.items(), key=lambda kv: -len(kv[1])):
        out["routes"].append({"route": name, **_stats(vals)})
    for name, e in sorted(api.items(), key=lambda kv: -len(kv[1]["ms"])):
        row = {"call": name, **_stats(e["ms"]), "err_pct": round(100 * sum(e["err"]) / len(e["err"]), 1)}
        if e["ttft"]:
            row["ttft_p75"] = round(percentile(e["ttft"], 75) or 0)
        out["api"].append(row)
    return out


_COLORS = {"bom": "#2e7d32", "regular": "#b26a00", "ruim": "#c62828", "": "#555"}


def render_html(s: Dict[str, Any], days: int) -> str:
    e = html.escape
    def n(x: Any, suf: str = "") -> str:
        return "—" if x is None else f"{x}{suf}"
    rows_v = "".join(
        f'<tr><td>{e(LABELS[m])}</td><td>{n(v["p50"])}</td><td><b style="color:{_COLORS[v["rating"]]}">{n(v["p75"])}</b></td>'
        f'<td>{n(v["p95"])}</td><td style="color:{_COLORS[v["rating"]]}">{e(v["rating"])}</td><td>{v["n"]}</td></tr>'
        for m, v in s["vitals"].items()) or '<tr><td colspan="6">Sem dados ainda.</td></tr>'
    rows_r = "".join(f'<tr><td>{e(r["route"])}</td><td>{r["n"]}</td><td>{n(r["p50"])}</td><td>{n(r["p75"])}</td></tr>' for r in s["routes"]) or '<tr><td colspan="4">Sem dados ainda.</td></tr>'
    rows_a = "".join(
        f'<tr><td>{e(a["call"])}</td><td>{a["n"]}</td><td>{n(a["p50"])}</td><td>{n(a["p75"])}</td><td>{n(a["p95"])}</td>'
        f'<td>{a["err_pct"]}%</td><td>{n(a.get("ttft_p75"))}</td></tr>' for a in s["api"]) or '<tr><td colspan="7">Sem dados ainda.</td></tr>'
    return f"""<!doctype html><html lang="pt-BR"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<meta name="robots" content="noindex"><title>Desempenho — Nosso Álbum</title>
<style>body{{font:15px/1.4 system-ui,sans-serif;margin:16px;color:#222;background:#fdfbf7}}h1{{font-size:20px}}h2{{font-size:16px;margin-top:24px}}
table{{border-collapse:collapse;width:100%;background:#fff}}th,td{{border:1px solid #e5e0d8;padding:6px 8px;text-align:right}}th:first-child,td:first-child{{text-align:left}}
th{{background:#f5f2eb}}.s{{color:#666;font-size:13px}}.wrap{{overflow-x:auto}}</style></head><body>
<h1>Desempenho com usuários reais</h1>
<p class="s">Últimos {days} dia(s) · {s["loads"]} carregamento(s) · {s["sessions"]} sessão(ões) · {s["installed"]} com app instalado · {s["mobile"]} no celular · {s["reports"]} relatório(s). Tempos em ms (CLS sem unidade). p75 = 3 de cada 4 acessos são melhores que isso.</p>
<h2>Web Vitals</h2><div class="wrap"><table><tr><th>Métrica</th><th>p50</th><th>p75</th><th>p95</th><th>nota</th><th>n</th></tr>{rows_v}</table></div>
<h2>Troca de tela (tempo até desenhar)</h2><div class="wrap"><table><tr><th>Tela</th><th>n</th><th>p50</th><th>p75</th></tr>{rows_r}</table></div>
<h2>Chamadas à API (vistas pelo usuário)</h2><div class="wrap"><table><tr><th>Chamada</th><th>n</th><th>p50</th><th>p75</th><th>p95</th><th>erros</th><th>1º token p75</th></tr>{rows_a}</table></div>
<p class="s">"1º token" = tempo até aparecer a primeira palavra da IA.</p></body></html>"""
