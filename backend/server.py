import base64
import hashlib
import json
import logging
import os
import secrets
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Annotated, Any, List, Optional

import bcrypt
import firebase_admin
import httpx
from firebase_admin import credentials, firestore_async
import jwt
from bson import ObjectId
from dotenv import load_dotenv
from fastapi import APIRouter, Depends, FastAPI, File, HTTPException, UploadFile
from fastapi.responses import StreamingResponse
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from motor.motor_asyncio import AsyncIOMotorClient
from pydantic import BaseModel, BeforeValidator, ConfigDict, Field
from starlette.middleware.cors import CORSMiddleware

from emergentintegrations.llm.chat import LlmChat, StreamDone, TextDelta, UserMessage

ROOT_DIR = Path(__file__).parent
load_dotenv(ROOT_DIR / ".env")

client = AsyncIOMotorClient(os.environ["MONGO_URL"])
db = client[os.environ["DB_NAME"]]
JWT_SECRET = os.environ["JWT_SECRET"]
CLOUD_NAME = os.environ["CLOUDINARY_CLOUD_NAME"]
UPLOAD_PRESET = os.environ["CLOUDINARY_UPLOAD_PRESET"]
LLM_KEY = os.environ["EMERGENT_LLM_KEY"]

# E-mail (códigos de recuperação de senha) enviado pela API do Brevo, que usa a porta web comum (443)
BREVO_API_KEY = os.environ.get("BREVO_API_KEY", "")
BREVO_SENDER = os.environ.get("BREVO_SENDER_EMAIL", "")

# Firebase: the album lives in Firestore collection "groups" (shared with the original website)
firebase_admin.initialize_app(credentials.Certificate(json.loads(base64.b64decode(os.environ["FIREBASE_SERVICE_ACCOUNT_B64"]))))
groups = firestore_async.client().collection("groups")

logger = logging.getLogger("album")
logging.basicConfig(level=logging.INFO)

app = FastAPI()
api = APIRouter(prefix="/api")
bearer = HTTPBearer(auto_error=False)

MODELS = {
    "claude-sonnet": ("anthropic", "claude-sonnet-5-5"),
    "claude-opus": ("anthropic", "claude-opus-5-5"),
    "gpt": ("openai", "gpt-5.6-terra"),
}

# ---------------------------------------------------------------- models
PyObjectId = Annotated[str, BeforeValidator(lambda v: str(v) if isinstance(v, ObjectId) else v)]


class BaseDocument(BaseModel):
    model_config = ConfigDict(populate_by_name=True)
    id: Optional[PyObjectId] = Field(default=None, validation_alias="_id")

    @classmethod
    def from_mongo(cls, doc: dict):
        return cls.model_validate(doc)

    def to_mongo(self) -> dict:
        data = self.model_dump(exclude_none=True)
        data.pop("id", None)
        return data


class User(BaseDocument):
    email: str
    name: str
    password_hash: str = ""
    avatar_url: str = ""


class UserOut(BaseModel):
    id: str
    email: str
    name: str
    avatar_url: Optional[str] = None
    partner_name: Optional[str] = None
    partner_avatar_url: Optional[str] = None


class Memory(BaseModel):
    """One album entry (Firestore doc in "groups"): several photos + caption, or a free-standing note."""
    id: str
    images: List[str] = []
    legenda: str = ""
    data: str = ""  # YYYY-MM-DD
    texto: str = ""
    music: str = ""
    note: Optional[dict] = None
    is_free_note: bool = False
    frame_bg: str = "#f7faf7"
    frame_border: str = "#dce8dc"
    frame_text_color: str = "#2d1b1b"
    created_at: str = ""
    author_name: Optional[str] = None
    updated_by_name: Optional[str] = None
    updated_at: Optional[str] = None
    photo_notes: Optional[dict] = None


# app field -> Firestore field (names kept compatible with the website's script)
FS_FIELDS = {
    "images": "images", "legenda": "text", "data": "date", "texto": "story", "music": "music", "note": "note",
    "is_free_note": "isFreeNote", "frame_bg": "frameBg", "frame_border": "frameBorder", "frame_text_color": "frameTextColor",
    "author_name": "authorName", "updated_by_name": "updatedByName", "updated_at": "updatedAt",
}


def first_present(d: dict, *keys):
    for k in keys:
        if d.get(k) not in (None, ""):
            return d[k]
    return None


def normalize_images(raw) -> List[str]:
    """Accepts a list (or single value) of URL strings / Cloudinary public ids / dicts with url|secure_url."""
    items = raw if isinstance(raw, list) else ([raw] if raw else [])
    out = []
    for it in items:
        url = it.get("secure_url") or it.get("url") if isinstance(it, dict) else it
        if not isinstance(url, str) or not url.strip():
            continue
        url = url.strip()
        if not url.startswith("http"):
            url = f"https://res.cloudinary.com/{CLOUD_NAME}/image/upload/{url}"
        out.append(url)
    return out


def memory_from_snap(snap) -> Memory:
    d = snap.to_dict() or {}
    created = d.get("createdAt")
    fields = {k: d[fk] for k, fk in FS_FIELDS.items() if d.get(fk) is not None}
    # Tolerate older/alternative field names (read-only fallbacks; the canonical names above win)
    fields["images"] = normalize_images(first_present(d, "images", "photos", "fotos", "urls", "imageUrl", "image", "foto"))
    fields.setdefault("legenda", first_present(d, "caption", "legenda", "title") or "")
    fields.setdefault("author_name", first_present(d, "author"))
    created_iso = created.isoformat() if hasattr(created, "isoformat") else str(created or "")
    fields.setdefault("data", first_present(d, "data") or created_iso[:10])
    if isinstance(d.get("photoNotes"), dict):
        fields["photo_notes"] = {str(k): v for k, v in d["photoNotes"].items() if isinstance(v, dict)}
    return Memory(id=snap.id, created_at=created_iso, **fields)


class Notification(BaseDocument):
    recipient_id: str
    actor_name: str
    kind: str  # added | edited | deleted
    message: str
    photo_id: Optional[str] = None
    read: bool = False
    created_at: str


class ChatMessage(BaseDocument):
    user_id: str
    role: str
    text: str
    model: str
    created_at: str


class LoginIn(BaseModel):
    email: str
    password: str
    remember: bool = True


class ForgotIn(BaseModel):
    email: str


class ResetIn(BaseModel):
    email: str
    code: str
    password: str = Field(min_length=8)


class MemoryIn(BaseModel):
    images: List[str] = []
    legenda: str = ""
    data: str = ""
    texto: str = ""
    music: str = ""
    note: Optional[dict] = None
    is_free_note: bool = False


class MemoryUpdate(BaseModel):
    images: Optional[List[str]] = None
    legenda: Optional[str] = None
    data: Optional[str] = None
    texto: Optional[str] = None
    music: Optional[str] = None
    note: Optional[dict] = None
    is_free_note: Optional[bool] = None


class CaptionIn(BaseModel):
    hint: str = ""
    data: str = ""
    model: str = "claude-sonnet"


class ChatIn(BaseModel):
    text: str
    model: str = "claude-sonnet"


class LetterIn(BaseModel):
    model: str = "claude-opus"
    tone: str = "romântica"


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


# ---------------------------------------------------------------- e-mail e limite de tentativas
async def send_email(to: str, subject: str, body: str) -> None:
    if not (BREVO_API_KEY and BREVO_SENDER):
        logger.error("Brevo não configurado (BREVO_API_KEY / BREVO_SENDER_EMAIL): e-mail não enviado")
        return
    try:
        async with httpx.AsyncClient(timeout=20) as hc:
            r = await hc.post(
                "https://api.brevo.com/v3/smtp/email",
                headers={"api-key": BREVO_API_KEY, "accept": "application/json"},
                json={
                    "sender": {"name": "Nosso Álbum", "email": BREVO_SENDER},
                    "to": [{"email": to}],
                    "subject": subject,
                    "textContent": body,
                },
            )
        if r.status_code >= 300:
            logger.error("Brevo erro %s: %s", r.status_code, r.text[:200])
    except Exception:
        logger.exception("Falha ao enviar e-mail")


async def too_many(key: str, limit: int, window_min: int) -> bool:
    since = (datetime.now(timezone.utc) - timedelta(minutes=window_min)).isoformat()
    return await db.attempts.count_documents({"key": key, "at": {"$gte": since}}) >= limit


async def register_attempt(key: str) -> None:
    await db.attempts.insert_one({"key": key, "at": now_iso()})


# ---------------------------------------------------------------- auth
def hash_pw(pw: str) -> str:
    return bcrypt.hashpw(pw.encode(), bcrypt.gensalt()).decode()


def make_token(uid: str, remember: bool) -> str:
    exp = datetime.now(timezone.utc) + (timedelta(days=60) if remember else timedelta(hours=12))
    return jwt.encode({"sub": uid, "exp": exp}, JWT_SECRET, algorithm="HS256")


async def current_user(creds: Annotated[Optional[HTTPAuthorizationCredentials], Depends(bearer)]) -> User:
    if not creds:
        raise HTTPException(401, "Não autenticado")
    try:
        payload = jwt.decode(creds.credentials, JWT_SECRET, algorithms=["HS256"])
        doc = await db.users.find_one({"_id": ObjectId(payload["sub"])})
    except Exception:
        raise HTTPException(401, "Sessão expirada, entre novamente")
    if not doc:
        raise HTTPException(401, "Sessão expirada, entre novamente")
    return User.from_mongo(doc)


CurrentUser = Annotated[User, Depends(current_user)]


async def user_out(u: User) -> UserOut:
    partner = await db.users.find_one({"_id": {"$ne": ObjectId(u.id)}})
    return UserOut(
        id=u.id, email=u.email, name=u.name, avatar_url=u.avatar_url or None,
        partner_name=partner["name"] if partner else None,
        partner_avatar_url=(partner.get("avatar_url") or None) if partner else None,
    )


@api.post("/auth/login")
async def login(body: LoginIn):
    email = body.email.strip().lower()
    key = f"login:{email}"
    if await too_many(key, 5, 15):
        raise HTTPException(429, "Muitas tentativas. Espere 15 minutos e tente de novo.")
    doc = await db.users.find_one({"email": email})
    if not doc or not bcrypt.checkpw(body.password.encode(), doc["password_hash"].encode()):
        await register_attempt(key)
        raise HTTPException(401, "E-mail ou senha incorretos")
    await db.attempts.delete_many({"key": key})
    u = User.from_mongo(doc)
    return {"token": make_token(u.id, body.remember), "user": await user_out(u)}


@api.get("/auth/me", response_model=UserOut)
async def me(user: CurrentUser):
    return await user_out(user)


class PartnerIn(BaseModel):
    name: str = Field(min_length=1)
    email: str
    password: str = Field(min_length=8)


@api.post("/auth/partner", response_model=UserOut)
async def add_partner(body: PartnerIn, user: CurrentUser):
    """Private album: the logged-in user can register exactly one partner (max 2 accounts)."""
    if await db.users.count_documents({}) >= 2:
        raise HTTPException(403, "O álbum já tem as duas contas do casal")
    email = body.email.strip().lower()
    if "@" not in email or "." not in email.split("@")[-1]:
        raise HTTPException(400, "E-mail inválido")
    if await db.users.find_one({"email": email}):
        raise HTTPException(400, "Este e-mail já está cadastrado")
    await db.users.insert_one(User(email=email, name=body.name.strip(), password_hash=hash_pw(body.password)).to_mongo())
    return await user_out(user)


class AvatarIn(BaseModel):
    url: str = ""


@api.put("/auth/avatar", response_model=UserOut)
async def set_avatar(body: AvatarIn, user: CurrentUser):
    """Foto de perfil da própria conta. Só aceita fotos enviadas pelo nosso Cloudinary; url vazia remove a foto."""
    url = body.url.strip()
    if url and not url.startswith(f"https://res.cloudinary.com/{CLOUD_NAME}/"):
        raise HTTPException(400, "Endereço de foto inválido")
    await db.users.update_one({"_id": ObjectId(user.id)}, {"$set": {"avatar_url": url}})
    user.avatar_url = url
    return await user_out(user)


@api.post("/auth/forgot")
async def forgot(body: ForgotIn):
    email = body.email.strip().lower()
    resp = {"message": "Se o e-mail estiver cadastrado, enviamos um código. Olhe também o spam."}
    key = f"forgot:{email}"
    if await too_many(key, 3, 60):
        return resp
    await register_attempt(key)
    doc = await db.users.find_one({"email": email})
    if doc:
        code = f"{secrets.randbelow(1_000_000):06d}"
        await db.reset_codes.delete_many({"email": email})
        await db.reset_codes.insert_one({
            "email": email,
            "code_hash": hashlib.sha256(code.encode()).hexdigest(),
            "expires_at": (datetime.now(timezone.utc) + timedelta(minutes=15)).isoformat(),
            "tries": 0,
        })
        await send_email(
            email,
            "Seu código para redefinir a senha",
            f"Seu código do Nosso Álbum é: {code}\n\nEle vale por 15 minutos. "
            "Se não foi você que pediu, pode ignorar este e-mail.",
        )
    return resp


@api.post("/auth/reset")
async def reset(body: ResetIn):
    email = body.email.strip().lower()
    doc = await db.reset_codes.find_one({"email": email})
    if not doc or doc["expires_at"] < now_iso() or doc.get("tries", 0) >= 5:
        raise HTTPException(400, "Código inválido ou expirado")
    if not secrets.compare_digest(doc["code_hash"], hashlib.sha256(body.code.strip().encode()).hexdigest()):
        await db.reset_codes.update_one({"_id": doc["_id"]}, {"$inc": {"tries": 1}})
        raise HTTPException(400, "Código inválido ou expirado")
    await db.users.update_one({"email": email}, {"$set": {"password_hash": hash_pw(body.password)}})
    await db.reset_codes.delete_many({"email": email})
    await db.attempts.delete_many({"key": f"login:{email}"})
    return {"message": "Senha alterada com sucesso"}


# ---------------------------------------------------------------- notifications helper
async def notify_partner(actor: User, kind: str, message: str, photo_id: Optional[str]):
    async for p in db.users.find({"_id": {"$ne": ObjectId(actor.id)}}):
        n = Notification(recipient_id=str(p["_id"]), actor_name=actor.name, kind=kind, message=message,
                         photo_id=photo_id, created_at=now_iso())
        await db.notifications.insert_one(n.to_mongo())


# ---------------------------------------------------------------- photos
@api.post("/upload")
async def upload(user: CurrentUser, file: UploadFile = File(...)):
    content = await file.read()
    if len(content) > 10 * 1024 * 1024:
        raise HTTPException(413, "Foto muito grande (máx. 10 MB)")
    async with httpx.AsyncClient(timeout=90) as hc:
        r = await hc.post(
            f"https://api.cloudinary.com/v1_1/{CLOUD_NAME}/image/upload",
            data={"upload_preset": UPLOAD_PRESET, "folder": "nosso-album-app"},
            files={"file": (file.filename or "foto.jpg", content, file.content_type or "image/jpeg")},
        )
    if r.status_code != 200:
        logger.error("Cloudinary error %s %s", r.status_code, r.text)
        raise HTTPException(502, "Não foi possível enviar a foto")
    return {"url": r.json()["secure_url"]}


async def list_memories() -> List[Memory]:
    snaps = await groups.get()
    return sorted((memory_from_snap(s) for s in snaps), key=lambda m: (m.data, m.created_at), reverse=True)


@api.get("/photos", response_model=List[Memory])
async def list_photos(user: CurrentUser):
    return await list_memories()


async def get_memory_or_404(pid: str) -> Memory:
    snap = await groups.document(pid).get()
    if not snap.exists:
        raise HTTPException(404, "Memória não encontrada")
    return memory_from_snap(snap)


def memory_title(m: Memory) -> str:
    return m.legenda or (m.note or {}).get("text", "").split("\n")[0] or "uma memória"


@api.get("/photos/{pid}", response_model=Memory)
async def get_photo(pid: str, user: CurrentUser):
    return await get_memory_or_404(pid)


@api.post("/photos", response_model=Memory)
async def create_photo(body: MemoryIn, user: CurrentUser):
    if not body.images and not body.is_free_note:
        raise HTTPException(400, "Escolha pelo menos uma foto")
    doc = {FS_FIELDS[k]: v for k, v in body.model_dump().items()}
    doc.update(createdAt=datetime.now(timezone.utc), authorName=user.name,
               frameBg="#f7faf7", frameBorder="#dce8dc", frameTextColor="#2d1b1b")
    ref = groups.document()
    await ref.set(doc)
    m = await get_memory_or_404(ref.id)
    what = "um bilhete novo 💌" if m.is_free_note else "uma memória nova 💕"
    await notify_partner(user, "added", f"{user.name} adicionou {what}", m.id)
    return m


FIELD_LABELS = {"legenda": "a legenda", "data": "a data", "texto": "o texto", "images": "as fotos", "music": "a música", "note": "o bilhete"}


@api.put("/photos/{pid}", response_model=Memory)
async def update_photo(pid: str, body: MemoryUpdate, user: CurrentUser):
    old = await get_memory_or_404(pid)
    changes = {k: v for k, v in body.model_dump(exclude_none=True).items() if getattr(old, k) != v}
    if changes:
        ts = now_iso()
        await groups.document(pid).update({FS_FIELDS[k]: v for k, v in changes.items()} | {"updatedAt": ts, "updatedByName": user.name})
        labels = [FIELD_LABELS[k] for k in FIELD_LABELS if k in changes]
        what = labels[0] if len(labels) == 1 else ", ".join(labels[:-1]) + " e " + labels[-1]
        await notify_partner(user, "edited", f"{user.name} alterou {what} de \"{memory_title(old)}\" ✏️", pid)
    return await get_memory_or_404(pid)


@api.delete("/photos/{pid}")
async def delete_photo(pid: str, user: CurrentUser):
    old = await get_memory_or_404(pid)
    await groups.document(pid).delete()
    await notify_partner(user, "deleted", f"{user.name} removeu a memória \"{memory_title(old)}\"", None)
    return {"ok": True}


# ---------------------------------------------------------------- notifications
@api.get("/notifications", response_model=List[Notification])
async def list_notifications(user: CurrentUser):
    docs = await db.notifications.find({"recipient_id": user.id}).sort("created_at", -1).to_list(200)
    return [Notification.from_mongo(d) for d in docs]


@api.get("/notifications/unread-count")
async def unread_count(user: CurrentUser):
    return {"count": await db.notifications.count_documents({"recipient_id": user.id, "read": False})}


@api.post("/notifications/read-all")
async def read_all(user: CurrentUser):
    await db.notifications.update_many({"recipient_id": user.id, "read": False}, {"$set": {"read": True}})
    return {"ok": True}


# ---------------------------------------------------------------- AI
MESES = ["janeiro", "fevereiro", "março", "abril", "maio", "junho", "julho", "agosto", "setembro", "outubro", "novembro", "dezembro"]
DIAS = ["segunda-feira", "terça-feira", "quarta-feira", "quinta-feira", "sexta-feira", "sábado", "domingo"]
BR_TZ = timezone(timedelta(hours=-3))  # horário de Brasília/Fortaleza
TONES = {
    "romântica": "romântico, delicado e sincero",
    "divertida": "leve, bem-humorado e brincalhão, sem perder o carinho",
    "nostálgica": "nostálgico e saudoso, lembrando o caminho que vocês fizeram juntos",
}


def make_chat(model_key: str, system: str, session: str) -> LlmChat:
    provider, model = MODELS.get(model_key, MODELS["claude-sonnet"])
    return LlmChat(api_key=LLM_KEY, session_id=session, system_message=system).with_model(provider, model)


def data_pt(d: date) -> str:
    return f"{d.day} de {MESES[d.month - 1]} de {d.year}"


def parse_date(s: str) -> Optional[date]:
    try:
        return datetime.strptime((s or "")[:10], "%Y-%m-%d").date()
    except ValueError:
        return None


def cut(text: str, n: int) -> str:
    t = " ".join((text or "").split())
    return t if len(t) <= n else t[: n - 1].rstrip() + "…"


def next_anniversary(d: date, today: date) -> Optional[date]:
    for y in (today.year, today.year + 1):
        try:
            nxt = d.replace(year=y)
        except ValueError:  # 29 de fevereiro em ano não bissexto
            continue
        if nxt >= today:
            return nxt
    return None


def memory_line(m: Memory) -> str:
    d = parse_date(m.data)
    if m.is_free_note:
        body = f"[bilhete] {cut((m.note or {}).get('text', ''), 300)}"
    else:
        body = cut(m.legenda or "sem legenda", 100)
        if m.texto:
            body += f" — {cut(m.texto, 300)}"
        if len(m.images) > 1:
            body += f" ({len(m.images)} fotos)"
    if m.music:
        body += f" (música: {cut(m.music, 80)})"
    who = f" [por {m.author_name}]" if m.author_name else ""
    return f"- {data_pt(d) if d else 'sem data'}: {body}{who}"


async def album_context() -> str:
    """Resumo do álbum para a IA: data de hoje, números, 'neste dia', próximos aniversários e as memórias."""
    mems = await list_memories()
    today = datetime.now(BR_TZ).date()
    head = [f"Hoje é {DIAS[today.weekday()]}, {data_pt(today)}."]
    if not mems:
        return "\n".join(head + ["O álbum ainda não tem memórias."])

    notes = sum(1 for m in mems if m.is_free_note)
    head.append(f"O álbum tem {len(mems)} registros: {len(mems) - notes} memórias com foto e {notes} bilhetes.")
    dated = [(parse_date(m.data), m) for m in mems]
    dated = [(d, m) for d, m in dated if d]
    if dated:
        head.append(f"Primeira memória: {data_pt(min(d for d, _ in dated))}. Mais recente: {data_pt(max(d for d, _ in dated))}.")

    on_this_day, upcoming = [], []
    for d, m in dated:
        nxt = next_anniversary(d, today)
        if not nxt or nxt.year <= d.year:
            continue
        delta = (nxt - today).days
        years = nxt.year - d.year
        title = cut(m.legenda or (m.note or {}).get("text", "") or "uma memória", 80)
        if delta == 0:
            on_this_day.append(f"- Há {years} ano(s), em {data_pt(d)}: {title}")
        elif delta <= 14:
            upcoming.append((delta, f"- Em {delta} dia(s) faz {years} ano(s) de: {title} ({data_pt(d)})"))

    parts = ["\n".join(head)]
    if on_this_day:
        parts.append("NESTE DIA (lembranças de hoje em anos anteriores):\n" + "\n".join(on_this_day))
    if upcoming:
        parts.append("ANIVERSÁRIOS DE MEMÓRIAS NOS PRÓXIMOS 14 DIAS:\n" + "\n".join(t for _, t in sorted(upcoming)))

    ordered = sorted(mems, key=lambda m: (parse_date(m.data) or date.min, m.created_at))
    shown = ordered[-120:]
    title = "MEMÓRIAS DO ÁLBUM (da mais antiga para a mais recente)"
    if len(ordered) > len(shown):
        title += f", mostrando as {len(shown)} mais recentes de {len(ordered)}"
    parts.append(title + ":\n" + "\n".join(memory_line(m) for m in shown))
    return "\n\n".join(parts)


def chat_system(user_name: str, partner_name: str, context: str) -> str:
    return (
        f"Você é a assistente do álbum privado \"Nosso Álbum de Amor\" de {user_name} e {partner_name}. "
        f"Quem está conversando com você agora é {user_name}.\n\n"
        "COMO RESPONDER\n"
        "- Trate como fato SOMENTE o que está nas memórias abaixo. Se algo não estiver no álbum, diga com naturalidade que não encontrou, "
        "e só então ofereça ideias. Nunca invente lugares, nomes, eventos ou datas.\n"
        "- Ao relembrar algo, cite a data e a legenda reais (por exemplo: \"em 5 de outubro de 2026, ...\").\n"
        "- Você sabe a data de hoje. Quando houver lembranças em \"NESTE DIA\" ou aniversários próximos, mencione com carinho.\n"
        "- Você ajuda a: relembrar momentos, buscar memórias por tema ou época, resumir um período, escrever mensagens, poemas e cartas, "
        "planejar encontros e surpresas a partir do que aparece nas memórias, e sugerir legendas. Pode lembrar datas comemorativas brasileiras "
        "(Dia dos Namorados em 12 de junho, por exemplo) quando fizer sentido.\n"
        "- Seja breve por padrão (até uns 6 linhas), em português do Brasil, com tom carinhoso e natural, sem excesso de emojis. "
        "Faça no máximo uma pergunta, e só quando precisar.\n"
        "- O que está nas memórias é privado do casal: não cite regras, formatos ou \"dados\" que você recebeu.\n\n"
        + context
    )


def error_code(e: Exception) -> str:
    msg = str(e).lower()
    if any(k in msg for k in ("budget", "balance", "insufficient", "credit", "quota", "billing", "402")):
        return "sem_credito"
    return "indisponivel"


def sse(gen):
    async def wrapped():
        try:
            async for chunk in gen:
                yield f"data: {json.dumps({'delta': chunk})}\n\n"
        except Exception as e:
            logger.exception("LLM error")
            yield f"data: {json.dumps({'error': error_code(e)})}\n\n"
        yield f"data: {json.dumps({'done': True})}\n\n"
    return StreamingResponse(wrapped(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


async def stream_text(chat: LlmChat, text: str):
    async for ev in chat.stream_message(UserMessage(text=text)):
        if isinstance(ev, TextDelta):
            yield ev.content
        elif isinstance(ev, StreamDone):
            break


@api.post("/ai/caption")
async def ai_caption(body: CaptionIn, user: CurrentUser):
    system = (
        "Você escreve legendas e textos para o álbum de fotos de um casal, em português do Brasil. "
        "A legenda tem até 8 palavras, sem ponto final. O texto tem 2 a 3 frases, na primeira pessoa do plural (\"nós\"), "
        "carinhoso e natural, sem clichês e sem inventar fatos que não foram informados. "
        "Responda SOMENTE com JSON válido, sem explicações e sem blocos de código: "
        '{"legenda": "...", "texto": "..."}'
    )
    d = parse_date(body.data)
    prompt = (
        f"Autor: {user.name}. Data da foto: {data_pt(d) if d else 'não informada'}. "
        f"Descrição ou ideia: {body.hint or 'um momento especial juntos'}."
    )
    chat = make_chat(body.model, system, f"caption-{secrets.token_hex(4)}")
    try:
        out = "".join([c async for c in stream_text(chat, prompt)]).strip()
    except Exception:
        logger.exception("LLM error (caption)")
        raise HTTPException(503, "A IA está indisponível agora. Tente novamente.")
    out = out.replace("```json", "").replace("```", "").strip()
    start, end = out.find("{"), out.rfind("}")
    try:
        data = json.loads(out[start:end + 1])
        return {"legenda": str(data.get("legenda", "")), "texto": str(data.get("texto", ""))}
    except Exception:
        return {"legenda": "", "texto": out}


@api.get("/ai/messages", response_model=List[ChatMessage])
async def ai_messages(user: CurrentUser):
    docs = await db.ai_messages.find({"user_id": user.id}).sort("created_at", 1).to_list(200)
    return [ChatMessage.from_mongo(d) for d in docs]


@api.delete("/ai/messages")
async def ai_clear(user: CurrentUser):
    await db.ai_messages.delete_many({"user_id": user.id})
    return {"ok": True}


@api.post("/ai/chat")
async def ai_chat(body: ChatIn, user: CurrentUser):
    history = await db.ai_messages.find({"user_id": user.id}).sort("created_at", -1).to_list(30)
    history.reverse()
    partner = (await user_out(user)).partner_name or "seu amor"
    system = chat_system(user.name, partner, await album_context())
    convo = "\n".join(f"{'Usuário' if m['role'] == 'user' else 'Assistente'}: {m['text']}" for m in history)
    prompt = (f"Conversa anterior:\n{convo}\n\n" if convo else "") + f"Usuário ({user.name}): {body.text}"
    chat = make_chat(body.model, system, f"chat-{user.id}")
    asked_at = now_iso()

    async def gen():
        parts: list[str] = []
        async for c in stream_text(chat, prompt):
            parts.append(c)
            yield c
        answer = "".join(parts).strip()
        if answer:  # só guarda a conversa quando a resposta chegou inteira
            await db.ai_messages.insert_many([
                ChatMessage(user_id=user.id, role="user", text=body.text, model=body.model, created_at=asked_at).to_mongo(),
                ChatMessage(user_id=user.id, role="assistant", text=answer, model=body.model, created_at=now_iso()).to_mongo(),
            ])
    return sse(gen())


@api.post("/ai/letter")
async def ai_letter(body: LetterIn, user: CurrentUser):
    partner = (await user_out(user)).partner_name or "meu amor"
    tone = TONES.get(body.tone, TONES["romântica"])
    system = (
        "Você escreve cartas de amor em português do Brasil, em primeira pessoa, com voz humana e sincera. "
        f"O tom é {tone}. Evite clichês e exageros. Use apenas memórias reais do álbum que receber, "
        "citando momentos concretos (com a data ou a legenda). Nunca invente fatos."
    )
    prompt = (
        f"Escreva uma carta de {user.name} para {partner}, com 3 a 4 parágrafos curtos, usando 2 ou 3 memórias reais do álbum. "
        f"Se houver algo em \"NESTE DIA\" ou aniversário próximo, vale citar. Termine com a assinatura de {user.name}.\n\n"
        f"{await album_context()}"
    )
    return sse(stream_text(make_chat(body.model, system, f"letter-{secrets.token_hex(4)}"), prompt))


@api.get("/")
async def root():
    return {"message": "Nosso Álbum de Amor API"}


app.include_router(api)

# Em produção, defina ALLOWED_ORIGINS no .env com os endereços do app separados por vírgula.
# O app instalado no celular não é afetado por CORS; isso vale só para o navegador.
ALLOWED_ORIGINS = [o.strip() for o in os.environ.get("ALLOWED_ORIGINS", "*").split(",") if o.strip()]
app.add_middleware(CORSMiddleware, allow_credentials=True, allow_origins=ALLOWED_ORIGINS, allow_methods=["*"], allow_headers=["*"])


@app.on_event("startup")
async def startup():
    await db.users.create_index("email", unique=True)
    # limpa registros antigos de tentativas de login/recuperação
    old = (datetime.now(timezone.utc) - timedelta(days=1)).isoformat()
    await db.attempts.delete_many({"at": {"$lt": old}})
    email = os.environ["OWNER_EMAIL"].strip().lower()
    await db.users.update_one(
        {"email": email},
        {"$setOnInsert": User(email=email, name=os.environ["OWNER_NAME"], password_hash=hash_pw(os.environ["OWNER_PASSWORD"])).to_mongo()},
        upsert=True,
    )


@app.on_event("shutdown")
async def shutdown():
    client.close()
