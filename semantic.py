#!/usr/bin/env python3
"""
Умный поиск по смыслу для Wiki Offline.

Как это работает:
  1. Для каждой статьи берём заголовок и первый абзац.
  2. Локальная нейросеть (multilingual-e5-small, ~110 МБ) превращает этот текст
     в вектор из 384 чисел — «смысловой отпечаток» статьи.
  3. При поиске запрос тоже превращается в вектор, и находятся статьи
     с самыми близкими по смыслу отпечатками — даже если слова не совпадают.

Индексирование запускается в отдельном процессе с низким приоритетом:
    python semantic.py index --zim файл.zim --out папка_индекса
Его можно прервать в любой момент — при следующем запуске оно продолжится.
"""

from __future__ import annotations

import argparse
import html as htmllib
import json
import os
import re
import sys
import time
import urllib.request
from pathlib import Path

import numpy as np

APP_DIR = Path(__file__).resolve().parent
MODEL_NAME = "multilingual-e5-small"
MODEL_REPO = "Xenova/multilingual-e5-small"
MODEL_FILES = {                     # локальное имя -> путь в репозитории
    "tokenizer.json": "tokenizer.json",
    "model.onnx": "onnx/model_quantized.onnx",
}
MODEL_DIR = APP_DIR / "models" / MODEL_NAME
INDEX_ROOT = APP_DIR / "index"
DIM = 384
MAX_TOKENS = 96           # сколько токенов текста статьи учитывать
INTRO_CHARS = 700         # сколько символов вступления брать
INDEX_VERSION = 1


# --------------------------------------------------------------------------- #
#  Извлечение текста из HTML статьи
# --------------------------------------------------------------------------- #
_RE_DROP = re.compile(
    r"<(style|script|table|figure|sup|math|noscript)\b[^>]*>.*?</\1\s*>",
    re.S | re.I,
)
_RE_P = re.compile(r"<p\b[^>]*>(.*?)</p\s*>", re.S | re.I)
_RE_TAG = re.compile(r"<[^>]+>")
_RE_SPACE = re.compile(r"\s+")
_RE_REFS = re.compile(r"\[\s*(?:\d+|[a-zA-Zа-яА-Я]{1,3}\s*\d*|источник не указан[^\]]*)\s*\]")
_RE_PUNCT_SPACE = re.compile(r"\s+([.,;:!?)»])")
_RE_ACCENT = re.compile("[\u0300\u0301]")   # знаки ударения: «Исаа́к» -> «Исаак»
_RE_IMG = re.compile(r"<img\b[^>]*>", re.I)
_RE_ATTR = re.compile(r'(\w[\w-]*)\s*=\s*"([^"]*)"')


def _body_start(doc: str) -> int:
    i = doc.find("mw-parser-output")
    if i < 0:
        i = doc.find("<body")
    return max(i, 0)


def extract_intro(doc: str, max_chars: int = INTRO_CHARS) -> str:
    """Возвращает первые абзацы статьи обычным текстом."""
    start = _body_start(doc)
    # Вводная часть статьи — до первого заголовка раздела.
    end = doc.find("<h2", start)
    chunk = doc[start:end if end > 0 else start + 200_000]
    if len(chunk) > 300_000:
        chunk = chunk[:300_000]
    chunk = _RE_DROP.sub(" ", chunk)
    parts, total = [], 0
    for m in _RE_P.finditer(chunk):
        text = _RE_TAG.sub("", m.group(1))
        text = htmllib.unescape(text)
        text = _RE_REFS.sub("", text)
        text = _RE_SPACE.sub(" ", text).strip()
        if len(text) < 25:
            continue
        parts.append(text)
        total += len(text)
        if total >= max_chars:
            break
    if not parts:
        # Нет абзацев (например, страница-список): берём любой текст.
        text = _RE_SPACE.sub(" ", htmllib.unescape(_RE_TAG.sub(" ", chunk))).strip()
        parts = [text[:max_chars]]
    out = _RE_PUNCT_SPACE.sub(r"\1", " ".join(parts))
    if len(out) > max_chars:
        cut = out.rfind(" ", 0, max_chars)
        out = out[: cut if cut > max_chars * 0.6 else max_chars] + "…"
    return out


def extract_image(doc: str) -> str | None:
    """Первая «настоящая» картинка статьи (относительный путь) или None."""
    start = _body_start(doc)
    for m in _RE_IMG.finditer(doc, start, start + 400_000):
        attrs = dict(_RE_ATTR.findall(m.group(0)))
        src = attrs.get("src", "")
        if not src or src.startswith("data:"):
            continue
        cls = attrs.get("class", "")
        if "math" in cls or src.endswith(".svg") and "mw-file-element" not in cls:
            continue
        try:
            w = int(attrs.get("width", "100"))
            h = int(attrs.get("height", "100"))
        except ValueError:
            w = h = 100
        if w < 60 or h < 40:
            continue
        return src
    return None


# --------------------------------------------------------------------------- #
#  Модель
# --------------------------------------------------------------------------- #
def model_ready() -> bool:
    return all((MODEL_DIR / f).is_file() for f in MODEL_FILES)


def download_model(progress=None):
    """Скачивает файлы модели с huggingface.co (один раз)."""
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    sizes = {}
    for local, remote in MODEL_FILES.items():
        if (MODEL_DIR / local).is_file():
            continue
        url = f"https://huggingface.co/{MODEL_REPO}/resolve/main/{remote}"
        part = MODEL_DIR / (local + ".part")
        req = urllib.request.Request(url, headers={"User-Agent": "WikiOffline/2.0"})
        with urllib.request.urlopen(req, timeout=60) as r, open(part, "wb") as f:
            total = int(r.headers.get("Content-Length") or 0)
            sizes[local] = total
            done = 0
            while True:
                buf = r.read(1 << 20)
                if not buf:
                    break
                f.write(buf)
                done += len(buf)
                if progress:
                    progress(local, done, total)
        part.replace(MODEL_DIR / local)


class Embedder:
    """Превращает тексты в нормированные векторы (ONNX Runtime, только CPU)."""

    def __init__(self, threads: int | None = None):
        import onnxruntime as ort
        from tokenizers import Tokenizer

        self.tok = Tokenizer.from_file(str(MODEL_DIR / "tokenizer.json"))
        self.tok.enable_truncation(max_length=MAX_TOKENS)
        self.tok.no_padding()
        opts = ort.SessionOptions()
        if threads:
            opts.intra_op_num_threads = threads
        opts.inter_op_num_threads = 1
        opts.log_severity_level = 3
        self.sess = ort.InferenceSession(
            str(MODEL_DIR / "model.onnx"), opts, providers=["CPUExecutionProvider"]
        )
        self.input_names = {i.name for i in self.sess.get_inputs()}

    def _run(self, encs) -> np.ndarray:
        n = max(len(e.ids) for e in encs)
        ids = np.ones((len(encs), n), dtype=np.int64)       # 1 = <pad> в XLM-R
        mask = np.zeros((len(encs), n), dtype=np.int64)
        for i, e in enumerate(encs):
            ids[i, : len(e.ids)] = e.ids
            mask[i, : len(e.ids)] = 1
        feeds = {"input_ids": ids, "attention_mask": mask}
        if "token_type_ids" in self.input_names:
            feeds["token_type_ids"] = np.zeros_like(ids)
        hidden = self.sess.run(None, feeds)[0]               # (B, T, 384)
        m = mask[..., None].astype(np.float32)
        vec = (hidden * m).sum(1) / np.maximum(m.sum(1), 1e-9)
        vec /= np.linalg.norm(vec, axis=1, keepdims=True) + 1e-12
        return vec.astype(np.float32)

    def embed(self, texts: list[str], batch: int = 16) -> np.ndarray:
        encs = self.tok.encode_batch(texts)
        # Сортируем по длине, чтобы меньше тратить на выравнивание (padding).
        order = sorted(range(len(encs)), key=lambda i: len(encs[i].ids))
        out = np.zeros((len(texts), DIM), dtype=np.float32)
        for s in range(0, len(order), batch):
            idx = order[s : s + batch]
            out[idx] = self._run([encs[i] for i in idx])
        return out

    def embed_query(self, text: str) -> np.ndarray:
        return self.embed(["query: " + text])[0]


# --------------------------------------------------------------------------- #
#  Файлы индекса
# --------------------------------------------------------------------------- #
def index_dir_for(book_id: str, uuid: str) -> Path:
    safe = re.sub(r"[^\w.\-]+", "_", book_id)[:60]
    return INDEX_ROOT / f"{safe}_{str(uuid)[:8]}"


def read_state(d: Path) -> dict:
    try:
        return json.loads((d / "state.json").read_text("utf-8"))
    except Exception:
        return {}


def write_state(d: Path, state: dict):
    tmp = d / "state.json.tmp"
    tmp.write_text(json.dumps(state, ensure_ascii=False, indent=1), "utf-8")
    os.replace(tmp, d / "state.json")


def quantize(vecs: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """float32 -> int8 + масштаб для каждого вектора (в 4 раза меньше места)."""
    mx = np.abs(vecs).max(axis=1, keepdims=True) + 1e-12
    scale = (127.0 / mx).astype(np.float32)
    q = np.clip(np.round(vecs * scale), -127, 127).astype(np.int8)
    return q, (1.0 / scale[:, 0]).astype(np.float32)


class VectorIndex:
    """Чтение индекса во время поиска (обновляется, пока идёт индексирование)."""

    def __init__(self, d: Path):
        self.dir = d
        self.count = -1
        self.vecs = self.ids = self.inv = None
        self._mtime = 0.0

    def _reload(self):
        f = self.dir / "state.json"
        try:
            mtime = f.stat().st_mtime
        except OSError:
            self.count = 0
            return
        if mtime == self._mtime:
            return
        self._mtime = mtime
        state = read_state(self.dir)
        n = int(state.get("count", 0))
        if n <= 0 or state.get("model") != MODEL_NAME:
            self.count = 0
            return
        self.vecs = np.memmap(self.dir / "vectors.i8", dtype=np.int8, mode="r", shape=(n, DIM))
        self.ids = np.fromfile(self.dir / "ids.u32", dtype=np.uint32, count=n)
        self.inv = np.fromfile(self.dir / "scales.f32", dtype=np.float32, count=n)
        self.count = n

    def search(self, q: np.ndarray, k: int = 100) -> list[tuple[int, float]]:
        self._reload()
        if self.count <= 0:
            return []
        n, step = self.count, 32768
        scores = np.empty(n, dtype=np.float32)
        q = q.astype(np.float32)
        for s in range(0, n, step):
            e = min(s + step, n)
            scores[s:e] = (self.vecs[s:e].astype(np.float32) @ q) * self.inv[s:e]
        k = min(k, n)
        top = np.argpartition(-scores, k - 1)[:k]
        top = top[np.argsort(-scores[top])]
        return [(int(self.ids[i]), float(scores[i])) for i in top]


# --------------------------------------------------------------------------- #
#  Индексирование (отдельный процесс)
# --------------------------------------------------------------------------- #
def _acquire_lock(out: Path):
    """Не даёт запустить два индексатора для одного архива одновременно."""
    f = open(out / "lock", "a+")
    try:
        if os.name == "nt":
            import msvcrt
            f.seek(0)
            msvcrt.locking(f.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl
            fcntl.flock(f.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        f.close()
        raise SystemExit("Индексирование этого архива уже запущено")
    return f


def _watch_parent():
    """Завершиться, когда программа (родительский процесс) закрылась."""
    import threading

    def watch():
        try:
            sys.stdin.buffer.read()
        except Exception:
            pass
        os._exit(0)

    threading.Thread(target=watch, daemon=True).start()


def run_indexer(zim_path: str, out: Path, threads: int):
    from libzim.reader import Archive

    out.mkdir(parents=True, exist_ok=True)
    lock = _acquire_lock(out)  # noqa: F841 (держим открытым до конца)
    state = read_state(out)
    if state.get("model") != MODEL_NAME or state.get("version") != INDEX_VERSION:
        state = {}
    archive = Archive(zim_path)
    state.update({
        "model": MODEL_NAME, "version": INDEX_VERSION, "dim": DIM,
        "zim": zim_path, "uuid": str(archive.uuid),
        "entry_count": archive.entry_count,
    })
    state.setdefault("count", 0)
    state.setdefault("next_entry", 0)
    state.setdefault("elapsed", 0.0)
    state["pid"] = os.getpid()
    state["error"] = None

    if not model_ready():
        state["phase"] = "download"
        write_state(out, state)
        last = [0.0]

        def prog(name, done, total):
            if time.time() - last[0] > 0.5:
                last[0] = time.time()
                state["download"] = {"file": name, "done": done, "total": total}
                write_state(out, state)

        download_model(prog)
        state.pop("download", None)

    state["phase"] = "indexing"
    write_state(out, state)
    emb = Embedder(threads)

    # Обрезаем файлы до последнего сохранённого состояния (если процесс был прерван).
    n = state["count"]
    for name, size in (("vectors.i8", DIM), ("ids.u32", 4), ("scales.f32", 4)):
        p = out / name
        with open(p, "ab") as f:
            f.truncate(n * size)

    fv = open(out / "vectors.i8", "ab")
    fi = open(out / "ids.u32", "ab")
    fs = open(out / "scales.f32", "ab")

    total = archive.entry_count
    i = state["next_entry"]
    texts, ids = [], []
    t_last = time.time()
    chunk = 512

    def flush():
        nonlocal texts, ids, t_last
        if texts:
            vecs = emb.embed(texts)
            q, inv = quantize(vecs)
            fv.write(q.tobytes()); fi.write(np.asarray(ids, dtype=np.uint32).tobytes())
            fs.write(inv.tobytes())
            for f in (fv, fi, fs):
                f.flush(); os.fsync(f.fileno())
            state["count"] += len(texts)
        now = time.time()
        state["elapsed"] += now - t_last
        t_last = now
        state["next_entry"] = i
        state["updated"] = now
        write_state(out, state)
        texts, ids = [], []

    while i < total:
        try:
            entry = archive._get_entry_by_id(i)
            if not entry.is_redirect:
                item = entry.get_item()
                if item.mimetype.startswith("text/html"):
                    doc = bytes(item.content).decode("utf-8", "replace")
                    title = entry.title or entry.path
                    intro = _RE_ACCENT.sub("", extract_intro(doc))
                    texts.append(f"passage: {title}. {intro}")
                    ids.append(i)
        except Exception:
            pass
        i += 1
        if len(texts) >= chunk:
            flush()
    flush()
    state["phase"] = "done"
    state["pid"] = None
    write_state(out, state)
    for f in (fv, fi, fs):
        f.close()


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("index")
    p.add_argument("--zim", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--threads", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    p.add_argument("--watch-parent", action="store_true",
                   help="завершиться, когда закроется stdin (родительский процесс)")
    sub.add_parser("download")
    args = ap.parse_args()

    if args.cmd == "download":
        download_model(lambda n, d, t: print(f"\r{n}: {d >> 20}/{t >> 20} МБ", end=""))
        print()
        return
    if args.watch_parent:
        _watch_parent()
    out = Path(args.out)
    try:
        run_indexer(args.zim, out, args.threads)
    except SystemExit:
        raise
    except Exception as e:
        st = read_state(out)
        st.update({"phase": "error", "error": str(e), "pid": None})
        write_state(out, st)
        raise


if __name__ == "__main__":
    main()
