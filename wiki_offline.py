#!/usr/bin/env python3
"""
Wiki Offline — программа для просмотра Википедии без интернета.

Читает ZIM-архивы (формат Kiwix) и показывает их в браузере через
локальный веб-сервер (доступен только с этого компьютера).

Возможности: умный поиск по смыслу (локальная нейросеть), подсказки,
история просмотров, закладки, тёмная тема.

Запуск:
    python wiki_offline.py                   # ZIM-файлы из папки ./zim
    python wiki_offline.py путь\\к\\файлу.zim  # конкретные файлы
    python wiki_offline.py --port 9000 --no-browser
"""

from __future__ import annotations

import argparse
import atexit
import html
import json
import mimetypes
import os
import re
import shutil
import subprocess
import sys
import threading
import time
import webbrowser
from collections import OrderedDict
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, quote, unquote, urljoin, urlsplit

try:
    from libzim.reader import Archive
    from libzim.search import Query, Searcher
    from libzim.suggestion import SuggestionSearcher
except ImportError:  # pragma: no cover
    print("Не установлена библиотека libzim. Выполните:\n    pip install -r requirements.txt")
    sys.exit(1)

import semantic

APP_DIR = Path(__file__).resolve().parent
STATIC_DIR = APP_DIR / "static"
DEFAULT_ZIM_DIR = APP_DIR / "zim"
DATA_DIR = APP_DIR / "data"
CONFIG_FILE = DATA_DIR / "library.json"
USER_FILE = DATA_DIR / "user.json"

TEXT_MIMES = ("text/", "application/javascript", "application/json", "image/svg+xml")

try:
    import numpy  # noqa: F401
    import onnxruntime  # noqa: F401
    import tokenizers  # noqa: F401
    AI_AVAILABLE = True
except ImportError:
    AI_AVAILABLE = False


class LRU(OrderedDict):
    def __init__(self, size: int):
        super().__init__()
        self.size = size
        self.lock = threading.Lock()

    def get_or(self, key, factory):
        with self.lock:
            if key in self:
                self.move_to_end(key)
                return self[key]
        value = factory()
        with self.lock:
            self[key] = value
            while len(self) > self.size:
                self.popitem(last=False)
        return value


# --------------------------------------------------------------------------- #
#  Текст: раскладка, стоп-слова, нормализация
# --------------------------------------------------------------------------- #
_EN = "qwertyuiop[]asdfghjkl;'zxcvbnm,.`QWERTYUIOP{}ASDFGHJKL:\"ZXCVBNM<>~"
_RU = "йцукенгшщзхъфывапролджэячсмитьбюёЙЦУКЕНГШЩЗХЪФЫВАПРОЛДЖЭЯЧСМИТЬБЮЁ"
EN2RU = str.maketrans(_EN, _RU)

STOP_WORDS = set("""
что кто где когда как какой какая какое какие каков почему зачем откуда куда сколько чей отчего
такое такой такая такие это этот эта эти тот та те то ли же бы был была было были быть есть
и в во на с со к ко по о об обо от до из у за для про при над под без через а но или не ни
мне меня мой моя мои я ты вы он она они его её ее их нам нас вам вас
можно нужно надо ещё еще уже очень самый самая самое самые
расскажи покажи объясни скажи найди подскажи хочу знать узнать
написал написала написали изобрел изобрёл открыл создал придумал основал построил снял
называется зовется значит означает работает устроен устроена состоит находится расположен расположена
делает живут обитает родился умер началась закончилась произошла
what who where when why how which is are was were the a an of in on to for and or do does did
""".split())

QUESTION_PREFIX = re.compile(
    r"^(?:расскажи|покажи|объясни|скажи|найди|подскажи|хочу знать|хочу узнать|"
    r"что такое|что значит|что означает|кто такой|кто такая|кто такие|"
    r"кто написал|кто автор|кто создал|кто изобрел|кто изобрёл|кто открыл|"
    r"кто основал|кто построил|кто придумал|кто снял|"
    r"почему|отчего|зачем|из-за чего|от чего|"
    r"как работает|как устроен|как устроена|как устроено|как называется|как зовется|как появил\w+|"
    r"где находится|где расположен|где расположена|где живут|где обитает|"
    r"когда был\w*|когда начал\w+|когда закончил\w+|когда произош\w+|когда родил\w+|когда умер\w*|"
    r"в каком году|какой|какая|какое|какие|сколько\s+\w+)\b[\s:,-]*",
    re.I,
)

CONCEPT_HINTS = [
    (re.compile(r"сам(ая|ый|ое)\s+высок(ая|ий)\s+(гора|вершина)|высочайш\w+\s+(гора|вершина)", re.I),
     ["Джомолунгма", "Список_высочайших_вершин_Земли", "Гора"]),
    (re.compile(r"сам(ое|ый)\s+глубок(ое|ий)\s+озеро|глубочайш\w+\s+озеро", re.I),
     ["Байкал", "Озеро"]),
    (re.compile(r"сам(ая|ый)\s+длинн(ая|ый)\s+река|крупнейш\w+\s+рек", re.I),
     ["Амазонка", "Нил", "Список_рек_по_длине"]),
    (re.compile(r"сам(ая|ый)\s+больш(ая|ой)\s+планет", re.I),
     ["Юпитер", "Солнечная_система"]),
    (re.compile(r"сам(ый|ая|ое)\s+больш(ой|ая|ое)\s+(океан|материк|континент|остров|пустыня|страна|государство)", re.I),
     ["Тихий_океан", "Евразия", "Гренландия", "Сахара", "Россия"]),
    (re.compile(r"планет\w*\s+с\s+кольцами", re.I),
     ["Сатурн", "Кольца_Сатурна", "Кольца_планет"]),
    (re.compile(r"небо\s+голуб|голуб\w+\s+небо|цвет\s+неба", re.I),
     ["Диффузное_излучение_неба", "Рэлеевское_рассеяние", "Небо"]),
    (re.compile(r"не\s+сворачивается\s+кров|св[её]ртываемост\w*\s+кров", re.I),
     ["Гемофилия", "Свёртывание_крови"]),
    (re.compile(r"притяжени\w*.*яблок|яблок\w*.*ньютон|закон\w*.*тяготен", re.I),
     ["Классическая_теория_тяготения_Ньютона", "Гравитация", "Ньютон,_Исаак"]),
    (re.compile(r"частиц\w*\s+без\s+масс|безмассов\w+\s+частиц", re.I),
     ["Безмассовые_частицы", "Фотон"]),
]

_MEDIA_Query = re.compile(r"\b(фильм|кино|сериал|песня|альбом|сингл|группа|игра|опера|пьеса|саундтрек|мультфильм)\b", re.I)
_MEDIA_SNIPPET = re.compile(
    r"—\s*(?:[а-яё\-]+\s+){0,4}(?:художественный\s+|драматический\s+|музыкальный\s+|студийный\s+|компьютерная\s+|телевизионный\s+)?"
    r"(?:фильм|телесериал|сериал|песня|альбом|сингл|рок-группа|музыкальная группа|компьютерная игра|видеоигра|саундтрек|опера)\b",
    re.I,
)


def normalize(s: str) -> str:
    s = semantic._RE_ACCENT.sub("", s).casefold().replace("ё", "е")
    return re.sub(r"[\s_]+", " ", re.sub(r"[^\w\s]", " ", s)).strip()


def nominative_forms(phrase: str) -> list[str]:
    """Приводит русские слова в косвенных падежах к начальной форме для поиска по заголовкам."""
    words = phrase.split()
    out = []
    for w in words:
        lw = w.lower()
        if len(lw) > 4:
            if lw.endswith("ую"):
                out.append(w[:-2] + "ая"); continue
            if lw.endswith("юю"):
                out.append(w[:-2] + "яя"); continue
            if lw.endswith("ию"):
                out.append(w[:-2] + "ия"); continue
            if lw.endswith("ии") and not lw.endswith("нии"):
                out.append(w[:-2] + "ия"); continue
            if lw.endswith("ости"):
                out.append(w[:-1]); continue
            if lw.endswith("ой") and len(lw) > 5:
                out.append(w[:-2] + "ая"); continue
            if lw.endswith("ого"):
                out.append(w[:-3] + "ый"); continue
        if len(lw) > 3:
            if lw.endswith("у") and not lw.endswith(("му", "ну", "ру")):
                out.append(w[:-1] + "а"); continue
            if lw.endswith("ю"):
                out.append(w[:-1] + "я"); continue
        out.append(w)
    alt = " ".join(out)
    return [phrase, alt] if alt.casefold() != phrase.casefold() else [phrase]


def extract_core(q: str) -> str:
    core = q.strip()
    for _ in range(2):
        core = QUESTION_PREFIX.sub("", core).strip(" ?!.,«»\"'")
    core = re.sub(r"\b(в мире|на земле|на свете|в природе)\b", "", core, flags=re.I).strip()
    return core or q.strip()


def keywords(q: str) -> str:
    words = [w for w in re.findall(r"\w+", q.casefold()) if w not in STOP_WORDS and len(w) > 1]
    return " ".join(words) or q


def looks_like_wrong_layout(q: str) -> bool:
    letters = [c for c in q if c.isalpha()]
    return bool(letters) and all(c.isascii() for c in letters)


# --------------------------------------------------------------------------- #
#  Библиотека ZIM-файлов
# --------------------------------------------------------------------------- #
class Book:
    """Один открытый ZIM-архив."""

    def __init__(self, book_id: str, path: Path):
        self.id = book_id
        self.path = path
        self.archive = Archive(str(path))
        self.uuid = str(self.archive.uuid)
        self._searcher: Searcher | None = None
        self._suggester: SuggestionSearcher | None = None
        self._lock = threading.Lock()
        self._vindex: semantic.VectorIndex | None = None
        self.snippets = LRU(2000)
        self.results = LRU(64)

    # --- метаданные -------------------------------------------------------- #
    def meta(self, key: str, default: str = "") -> str:
        try:
            return bytes(self.archive.get_metadata(key)).decode("utf-8", "replace")
        except Exception:
            return default

    @property
    def title(self) -> str:
        return self.meta("Title") or self.path.stem

    @property
    def language(self) -> str:
        return self.meta("Language")

    def is_russian(self) -> bool:
        return any(x in self.language.lower() for x in ("ru", "rus"))

    def has_illustration(self) -> bool:
        try:
            return bool(self.archive.get_illustration_sizes())
        except Exception:
            return False

    def illustration(self) -> bytes | None:
        try:
            sizes = sorted(self.archive.get_illustration_sizes())
            if not sizes:
                return None
            return bytes(self.archive.get_illustration_item(sizes[-1]).content)
        except Exception:
            return None

    @property
    def index_dir(self) -> Path:
        return semantic.index_dir_for(self.id, self.uuid)

    @property
    def vindex(self) -> semantic.VectorIndex:
        if self._vindex is None:
            self._vindex = semantic.VectorIndex(self.index_dir)
        return self._vindex

    def info(self) -> dict:
        a = self.archive
        return {
            "id": self.id,
            "title": self.title,
            "description": self.meta("Description"),
            "language": self.language,
            "date": self.meta("Date"),
            "articles": a.article_count,
            "size": self.path.stat().st_size,
            "file": str(self.path),
            "fulltext": a.has_fulltext_index,
            "illustration": self.has_illustration(),
            "main": self.main_path(),
        }

    # --- навигация ------------------------------------------------------- #
    def main_path(self) -> str | None:
        a = self.archive
        if not a.has_main_entry:
            return None
        entry = a.main_entry
        for _ in range(10):
            if not entry.is_redirect:
                break
            entry = entry.get_redirect_entry()
        return entry.path

    def find_entry(self, path: str):
        """Ищет запись по пути, учитывая старую схему с пространствами имён."""
        a = self.archive
        candidates = [path]
        if re.match(r"^[A-Z\-]/", path):
            candidates.append(path[2:])
        else:
            candidates.append("A/" + path)
        for p in candidates:
            try:
                if a.has_entry_by_path(p):
                    return a.get_entry_by_path(p)
            except Exception:
                continue
        return None

    def resolve(self, path: str):
        entry = self.find_entry(path)
        for _ in range(10):
            if entry is None or not entry.is_redirect:
                break
            entry = entry.get_redirect_entry()
        return entry

    def random_path(self) -> str | None:
        a = self.archive
        for _ in range(30):
            try:
                entry = a.get_random_entry()
                if entry.is_redirect:
                    entry = entry.get_redirect_entry()
                if entry.get_item().mimetype.startswith("text/html"):
                    return entry.path
            except Exception:
                continue
        return None

    # --- текст статьи для результатов поиска ----------------------------- #
    def snippet(self, path: str) -> dict:
        def make():
            out = {"snippet": "", "image": None}
            try:
                entry = self.resolve(path)
                if entry is None:
                    return out
                item = entry.get_item()
                if not item.mimetype.startswith("text/html"):
                    return out
                doc = bytes(item.content).decode("utf-8", "replace")
                out["snippet"] = semantic.extract_intro(doc, 420)
                img = semantic.extract_image(doc)
                if img:
                    base = f"/content/{quote(self.id)}/{quote(entry.path)}"
                    out["image"] = urljoin(base, img)
            except Exception:
                pass
            return out
        return self.snippets.get_or(path, make)

    # --- поиск ----------------------------------------------------------- #
    def suggest(self, text: str, count: int = 10) -> list[dict]:
        with self._lock:
            if self._suggester is None:
                self._suggester = SuggestionSearcher(self.archive)
            suggestion = self._suggester.suggest(text)
        return self._entries(suggestion.getResults(0, count))

    def fulltext(self, text: str, count: int = 40) -> list[dict]:
        if not self.archive.has_fulltext_index:
            return []
        with self._lock:
            if self._searcher is None:
                self._searcher = Searcher(self.archive)
            search = self._searcher.search(Query().set_query(text))
        return self._entries(search.getResults(0, count))

    def _entries(self, paths) -> list[dict]:
        out = []
        for path in paths:
            try:
                entry = self.archive.get_entry_by_path(path)
                if entry.is_redirect:
                    entry = entry.get_redirect_entry()
                out.append({"title": entry.title or path, "path": entry.path})
            except Exception:
                out.append({"title": path, "path": path})
        return out


class Library:
    """Набор ZIM-файлов, доступных в программе."""

    def __init__(self, zim_dir: Path, extra_files: list[Path]):
        self.zim_dir = zim_dir
        self.extra_files = list(extra_files)
        self.books: dict[str, Book] = {}
        self.errors: list[str] = []
        self._lock = threading.Lock()
        self._load_config()
        self.refresh()

    def _load_config(self):
        try:
            data = json.loads(CONFIG_FILE.read_text("utf-8"))
            for f in data.get("files", []):
                p = Path(f)
                if p not in self.extra_files:
                    self.extra_files.append(p)
        except Exception:
            pass

    def _save_config(self):
        try:
            DATA_DIR.mkdir(exist_ok=True)
            CONFIG_FILE.write_text(
                json.dumps({"files": [str(p) for p in self.extra_files]},
                           ensure_ascii=False, indent=2), "utf-8")
        except Exception:
            pass

    def _candidates(self) -> list[Path]:
        files: list[Path] = []
        if self.zim_dir.is_dir():
            files += sorted(self.zim_dir.rglob("*.zim"))
        files += [p for p in self.extra_files if p.is_file()]
        seen, unique = set(), []
        for f in files:
            key = os.path.normcase(str(f.resolve()))
            if key not in seen:
                seen.add(key)
                unique.append(f.resolve())
        return unique

    @staticmethod
    def _make_id(path: Path, taken: set[str]) -> str:
        base = re.sub(r"[^\w.\-]+", "_", path.stem, flags=re.UNICODE).strip("_") or "book"
        book_id, n = base, 2
        while book_id in taken:
            book_id, n = f"{base}_{n}", n + 1
        return book_id

    def refresh(self):
        with self._lock:
            known = {os.path.normcase(str(b.path)): b for b in self.books.values()}
            new_books: dict[str, Book] = {}
            self.errors = []
            for path in self._candidates():
                key = os.path.normcase(str(path))
                book = known.get(key)
                if book is None:
                    try:
                        book = Book(self._make_id(path, set(new_books)), path)
                    except Exception as e:
                        self.errors.append(f"{path.name}: {e}")
                        continue
                new_books[book.id] = book
            self.books = new_books

    def add_file(self, path: Path) -> Book | None:
        path = path.resolve()
        if path not in self.extra_files:
            self.extra_files.append(path)
            self._save_config()
        self.refresh()
        for b in self.books.values():
            if os.path.normcase(str(b.path)) == os.path.normcase(str(path)):
                return b
        return None

    def remove_file(self, book_id: str) -> bool:
        book = self.books.get(book_id)
        if not book:
            return False
        before = len(self.extra_files)
        self.extra_files = [
            p for p in self.extra_files
            if os.path.normcase(str(p.resolve())) != os.path.normcase(str(book.path))
        ]
        if len(self.extra_files) != before:
            self._save_config()
            self.refresh()
            return True
        return False

    def get(self, book_id: str) -> Book | None:
        return self.books.get(book_id)


# --------------------------------------------------------------------------- #
#  Умный поиск (ИИ)
# --------------------------------------------------------------------------- #
class AIManager:
    """Запускает индексирование в фоне и выполняет поиск по смыслу."""

    def __init__(self):
        self.procs: dict[str, subprocess.Popen] = {}
        self._embedder = None
        self._emb_lock = threading.Lock()
        self.qcache = LRU(256)
        atexit.register(self.stop_all)

    # --- индексирование -------------------------------------------------- #
    def running(self, book: Book) -> bool:
        p = self.procs.get(book.id)
        return p is not None and p.poll() is None

    def start(self, book: Book):
        if not AI_AVAILABLE or self.running(book):
            return
        d = book.index_dir
        d.mkdir(parents=True, exist_ok=True)
        (d / "paused").unlink(missing_ok=True)
        threads = max(1, (os.cpu_count() or 2) - 2)
        flags = 0
        if os.name == "nt":
            flags = subprocess.BELOW_NORMAL_PRIORITY_CLASS | subprocess.CREATE_NO_WINDOW
        log = open(d / "log.txt", "ab")
        self.procs[book.id] = subprocess.Popen(
            [sys.executable, str(APP_DIR / "semantic.py"), "index",
             "--zim", str(book.path), "--out", str(d),
             "--threads", str(threads), "--watch-parent"],
            stdin=subprocess.PIPE, stdout=log, stderr=log,
            creationflags=flags, cwd=str(APP_DIR),
        )

    def stop(self, book: Book, mark_paused: bool = True):
        p = self.procs.pop(book.id, None)
        if p and p.poll() is None:
            p.terminate()
            try:
                p.wait(5)
            except Exception:
                pass
        if mark_paused and book.index_dir.is_dir():
            (book.index_dir / "paused").write_text("1")

    def reset(self, book: Book):
        self.stop(book, mark_paused=False)
        book._vindex = None
        shutil.rmtree(book.index_dir, ignore_errors=True)
        book.results.clear()

    def stop_all(self):
        for p in self.procs.values():
            if p.poll() is None:
                p.terminate()

    def autostart(self, library: Library):
        """Продолжает индексирование, прерванное закрытием программы."""
        if not AI_AVAILABLE:
            return
        for book in library.books.values():
            st = semantic.read_state(book.index_dir)
            if st and st.get("phase") not in ("done",) and not (book.index_dir / "paused").exists():
                self.start(book)

    def status(self, book: Book) -> dict:
        if not AI_AVAILABLE:
            return {"state": "unavailable"}
        st = semantic.read_state(book.index_dir)
        running = self.running(book)
        if not st:
            return {"state": "starting" if running else "none"}
        phase = st.get("phase")
        total = max(int(st.get("entry_count") or 1), 1)
        nxt = int(st.get("next_entry") or 0)
        elapsed = float(st.get("elapsed") or 0)
        out = {
            "count": int(st.get("count") or 0),
            "percent": 100.0 if phase == "done" else round(100.0 * nxt / total, 1),
            "error": st.get("error"),
        }
        if phase == "done":
            out["state"] = "done"
        elif phase == "error" and not running:
            out["state"] = "error"
        elif not running:
            out["state"] = "paused"
        elif phase == "download":
            out["state"] = "download"
            out["download"] = st.get("download")
        else:
            out["state"] = "indexing" if phase == "indexing" else "starting"
            if elapsed > 5 and nxt > 0:
                out["eta"] = int((total - nxt) / (nxt / elapsed))
                out["rate"] = round(out["count"] / elapsed, 1)
        return out

    # --- поиск ----------------------------------------------------------- #
    def embedder(self):
        with self._emb_lock:
            if self._embedder is None:
                self._embedder = semantic.Embedder(threads=4)
            return self._embedder

    def search(self, book: Book, q: str, k: int = 100) -> list[tuple[int, float]]:
        if not AI_AVAILABLE or not semantic.model_ready():
            return []
        vi = book.vindex
        vi._reload()
        if vi.count <= 0:
            return []
        vec = self.qcache.get_or(q, lambda: self.embedder().embed_query(q))
        return vi.search(vec, k)


AI = AIManager()


def smart_suggest(book: Book, query: str, count: int = 8) -> list[dict]:
    """Подсказки по заголовкам с пониманием раскладки, вопросов и падежей."""
    q = query.strip()
    if not q:
        return []
    seen, out = set(), []

    def push(items):
        for it in items:
            if it["path"] not in seen:
                seen.add(it["path"])
                out.append(it)

    for rx, paths in CONCEPT_HINTS:
        if rx.search(q):
            for p in paths:
                e = book.resolve(p)
                if e:
                    push([{"title": e.title or e.path, "path": e.path}])

    push(book.suggest(q, count))
    if book.is_russian() and looks_like_wrong_layout(q):
        alt = q.translate(EN2RU)
        push(book.suggest(alt, count))
        q = alt

    core = extract_core(q)
    for form in nominative_forms(core):
        if form.casefold() != q.casefold():
            push(book.suggest(form, count))
    return out[:count]


def smart_search(book: Book, query: str) -> dict:
    """
    Умный гибридный поиск:
      1. Исправление раскладки, разбор вопросов и приведение падежей.
      2. Сбор кандидатов из заголовков, полнотекстового индекса и векторного ИИ-индекса.
      3. Мгновенное нейросетевое ранжирование кандидатов по смыслу (multilingual-e5-small).
    """
    q = query.strip()
    info = {"query": q, "corrected": None, "suggestion": None}

    # 1. Неправильная раскладка: «vjcrdf» -> «москва».
    titles = book.suggest(q, 10)
    if book.is_russian() and looks_like_wrong_layout(q):
        alt = q.translate(EN2RU)
        alt_titles = book.suggest(alt, 10)
        if alt_titles and not titles:
            q, titles = alt, alt_titles
            info["corrected"] = alt
        elif alt_titles:
            info["suggestion"] = alt

    core = extract_core(q)
    kw = keywords(core)

    scores: dict[str, float] = {}
    meta: dict[str, dict] = {}
    pre_sims: dict[str, float] = {}
    concept_set: set[str] = set()

    def add(items, weight, k=8):
        for rank, it in enumerate(items):
            p = it["path"]
            scores[p] = scores.get(p, 0.0) + weight / (k + rank)
            meta.setdefault(p, {"title": it["title"], "path": p})

    # 2. Подсказки для частых вопросов («самая высокая гора», «почему небо голубое» и т.д.).
    for rx, paths in CONCEPT_HINTS:
        if rx.search(q):
            for rank, p in enumerate(paths):
                e = book.resolve(p)
                if e:
                    concept_set.add(e.path)
                    add([{"title": e.title or e.path, "path": e.path}], 1.2 / (1 + rank * 0.2), k=1)

    # 3. Заголовки по основному понятию и в именительном падеже («войну и мир» -> «Война и мир»).
    add(titles, 0.55)
    for form in nominative_forms(core):
        if form.casefold() != q.casefold():
            add(book.suggest(form, 8), 0.75)
    if kw.casefold() != core.casefold():
        for form in nominative_forms(kw):
            add(book.suggest(form, 6), 0.55)

    # 4. Полнотекстовый поиск по ключевым словам.
    add(book.fulltext(kw, 30), 0.75)

    # 5. Векторный ИИ-индекс (если архив уже проиндексирован полностью или частично).
    ai_ready = AI_AVAILABLE and semantic.model_ready()
    qvec = None
    if ai_ready:
        try:
            qvec = AI.qcache.get_or(q, lambda: AI.embedder().embed_query(q))
            vi = book.vindex
            vi._reload()
            if vi.count > 0:
                sem_items = []
                for idx, sim in vi.search(qvec, 60):
                    try:
                        e = book.archive._get_entry_by_id(idx)
                        p = e.path
                        pre_sims[p] = float(sim)
                        sem_items.append({"title": e.title or p, "path": p})
                    except Exception:
                        continue
                add(sem_items, 1.1)
        except Exception:
            qvec = None

    # 6. Мгновенное нейросетевое переранжирование топ-кандидатов по смыслу первого абзаца.
    used_neural = bool(pre_sims)
    if qvec is not None and meta:
        top_paths = sorted(scores, key=lambda p: -scores[p])[:18]
        to_embed_paths, passages = [], []
        for p in top_paths:
            if p in pre_sims:
                continue
            sn = book.snippet(p)["snippet"]
            if not sn:
                continue
            clean_sn = re.sub(r"\([^)]{15,200}\)", "", sn[:320])
            passages.append(f"passage: {meta[p]['title']}. {clean_sn}")
            to_embed_paths.append(p)
            if len(passages) >= 12:
                break
        if passages:
            try:
                pvecs = AI.embedder().embed(passages, batch=12)
                for p, sim in zip(to_embed_paths, pvecs @ qvec):
                    pre_sims[p] = float(sim)
                used_neural = True
            except Exception:
                pass

    # 7. Итоговый рейтинг: смысл + совпадение заголовка - нерелевантные фильмы/песни.
    nq, ncore, nkw = normalize(q), normalize(core), normalize(kw)
    is_media_query = bool(_MEDIA_Query.search(q))
    for p, m in meta.items():
        nt = normalize(m["title"])
        base_title = nt.split(" (")[0]
        if nt in (nq, ncore, nkw):
            scores[p] += 0.45
        elif base_title in (ncore, nkw):
            scores[p] += 0.12
        if p in concept_set:
            scores[p] += 0.35
        if p in pre_sims:
            scores[p] += max(0.0, pre_sims[p] - 0.65) * 2.2
        tl = m["title"].lower()
        if "(значения)" in tl:
            scores[p] -= 0.12
        elif not is_media_query:
            if "(" in tl and any(w in tl for w in ("фильм", "песня", "альбом", "сериал", "саундтрек", "пьеса", "опера", "игра")):
                scores[p] -= 0.14
            else:
                sn = book.snippets.get(p, {}).get("snippet", "")
                if sn and _MEDIA_SNIPPET.search(sn[:220]):
                    scores[p] -= 0.16

    ranked = sorted(scores, key=lambda p: -scores[p])
    info["results"] = [meta[p] for p in ranked]
    vi = book.vindex
    st = AI.status(book)
    info["mode"] = "ai" if used_neural else "keyword"
    info["ai"] = {"state": st.get("state"), "percent": st.get("percent"), "count": vi.count}
    return info


# --------------------------------------------------------------------------- #
#  История и закладки
# --------------------------------------------------------------------------- #
class UserData:
    MAX_HISTORY = 1000

    def __init__(self, path: Path):
        self.path = path
        self.lock = threading.Lock()
        try:
            self.data = json.loads(path.read_text("utf-8"))
        except Exception:
            self.data = {}
        self.data.setdefault("history", [])
        self.data.setdefault("bookmarks", [])

    def _save(self):
        self.path.parent.mkdir(exist_ok=True)
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self.data, ensure_ascii=False, indent=1), "utf-8")
        os.replace(tmp, self.path)

    def snapshot(self) -> dict:
        with self.lock:
            return json.loads(json.dumps(self.data))

    def add_history(self, item: dict):
        with self.lock:
            h = [x for x in self.data["history"]
                 if not (x["book"] == item["book"] and x["path"] == item["path"])]
            h.insert(0, item)
            self.data["history"] = h[: self.MAX_HISTORY]
            self._save()

    def remove_history(self, book: str, path: str):
        with self.lock:
            self.data["history"] = [x for x in self.data["history"]
                                    if not (x["book"] == book and x["path"] == path)]
            self._save()

    def clear_history(self):
        with self.lock:
            self.data["history"] = []
            self._save()

    def toggle_bookmark(self, item: dict) -> bool:
        with self.lock:
            b = self.data["bookmarks"]
            for x in b:
                if x["book"] == item["book"] and x["path"] == item["path"]:
                    b.remove(x)
                    self._save()
                    return False
            b.insert(0, item)
            self._save()
            return True

    def remove_bookmark(self, book: str, path: str):
        with self.lock:
            self.data["bookmarks"] = [x for x in self.data["bookmarks"]
                                      if not (x["book"] == book and x["path"] == path)]
            self._save()


# --------------------------------------------------------------------------- #
#  Выбор файла через системный диалог (в отдельном процессе — так надёжнее)
# --------------------------------------------------------------------------- #
DIALOG_CODE = r"""
import tkinter as tk
from tkinter import filedialog
root = tk.Tk()
root.withdraw()
root.attributes('-topmost', True)
p = filedialog.askopenfilename(
    title='Выберите ZIM-файл Википедии',
    filetypes=[('ZIM-архивы', '*.zim'), ('Все файлы', '*.*')])
import sys
sys.stdout.buffer.write((p or '').encode('utf-8'))
"""


def ask_zim_file() -> str:
    try:
        res = subprocess.run([sys.executable, "-c", DIALOG_CODE],
                             capture_output=True, timeout=600)
        return res.stdout.decode("utf-8", "replace").strip()
    except Exception:
        return ""


# --------------------------------------------------------------------------- #
#  HTTP-сервер
# --------------------------------------------------------------------------- #
class Handler(BaseHTTPRequestHandler):
    library: Library
    user: UserData
    server_version = "WikiOffline/2.0"
    protocol_version = "HTTP/1.1"

    def log_message(self, fmt, *args):
        pass

    def log_error(self, fmt, *args):
        sys.stderr.write("[ошибка] " + (fmt % args) + "\n")

    # --- утилиты ответа --------------------------------------------------- #
    def send_bytes(self, data: bytes, mime: str, status=HTTPStatus.OK, cache: int = 0):
        self.send_response(status)
        self.send_header("Content-Type", mime)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", f"public, max-age={cache}" if cache else "no-store")
        self.end_headers()
        if self.command != "HEAD":
            self.wfile.write(data)

    def send_json(self, obj, status=HTTPStatus.OK):
        self.send_bytes(json.dumps(obj, ensure_ascii=False).encode("utf-8"),
                        "application/json; charset=utf-8", status)

    def redirect(self, location: str):
        self.send_response(HTTPStatus.FOUND)
        self.send_header("Location", location)
        self.send_header("Content-Length", "0")
        self.end_headers()

    def not_found(self, text="Страница не найдена"):
        body = (
            "<!DOCTYPE html><html lang='ru'><head><meta charset='utf-8'>"
            "<link rel='stylesheet' href='/static/page.css'></head><body><main>"
            f"<h1>Страница не найдена</h1><p>{html.escape(text)}</p></main></body></html>"
        )
        self.send_bytes(body.encode("utf-8"), "text/html; charset=utf-8", HTTPStatus.NOT_FOUND)

    def read_json(self) -> dict:
        n = int(self.headers.get("Content-Length") or 0)
        if not n:
            return {}
        try:
            return json.loads(self.rfile.read(n).decode("utf-8"))
        except Exception:
            return {}

    # --- маршрутизация --------------------------------------------------- #
    def do_HEAD(self):
        self.do_GET()

    def do_GET(self):
        self._safe("GET")

    def do_POST(self):
        self._safe("POST")

    def _safe(self, method):
        try:
            self.route(method)
        except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
            pass
        except Exception as e:  # pragma: no cover
            self.log_error("%s: %r", self.path, e)
            try:
                self.send_json({"error": str(e)}, HTTPStatus.INTERNAL_SERVER_ERROR)
            except Exception:
                pass

    def book_param(self, qs) -> Book | None:
        return self.library.get(qs.get("book", ""))

    def route(self, method: str):
        parts = urlsplit(self.path)
        path = unquote(parts.path)
        qs = {k: v[0] for k, v in parse_qs(parts.query).items()}
        if method == "POST":
            return self.route_post(path, qs, self.read_json())

        if path in ("/", "/index.html"):
            return self.static("index.html")
        if path == "/viewer":
            return self.redirect("/")
        if path.startswith("/static/"):
            return self.static(path[len("/static/"):])
        if path.startswith("/content/"):
            return self.content(parts.path[len("/content/"):])
        if path == "/random":
            book = self.book_param(qs)
            target = book.random_path() if book else None
            if not target:
                return self.not_found("Не удалось выбрать случайную статью")
            return self.redirect(f"/content/{quote(book.id)}/{quote(target)}")
        if path == "/api/books":
            if qs.get("refresh"):
                self.library.refresh()
            books = []
            for b in self.library.books.values():
                info = b.info()
                info["ai"] = AI.status(b)
                books.append(info)
            return self.send_json({
                "books": books,
                "errors": self.library.errors,
                "zim_dir": str(self.library.zim_dir),
                "ai_available": AI_AVAILABLE,
            })
        if path == "/api/ai/status":
            return self.send_json({b.id: AI.status(b) for b in self.library.books.values()})
        if path.startswith("/api/illustration/"):
            book = self.library.get(path[len("/api/illustration/"):])
            data = book.illustration() if book else None
            if not data:
                return self.not_found()
            return self.send_bytes(data, "image/png", cache=86400)
        if path == "/api/suggest":
            book, q = self.book_param(qs), qs.get("q", "").strip()
            if not book or not q:
                return self.send_json([])
            return self.send_json(smart_suggest(book, q, 8))
        if path == "/api/search":
            return self.api_search(qs)
        if path == "/api/user":
            return self.send_json(self.user.snapshot())
        return self.not_found()

    def route_post(self, path: str, qs: dict, body: dict):
        book = self.book_param(qs) or self.library.get(body.get("book", ""))
        if path == "/api/open":
            return self.api_open()
        if path == "/api/remove":
            return self.send_json({"ok": self.library.remove_file(qs.get("book", ""))})
        if path == "/api/refresh":
            self.library.refresh()
            return self.send_json({"ok": True})
        if path in ("/api/ai/start", "/api/ai/stop", "/api/ai/reset"):
            if not book:
                return self.send_json({"ok": False, "error": "Архив не найден"})
            {"/api/ai/start": AI.start, "/api/ai/stop": AI.stop,
             "/api/ai/reset": AI.reset}[path](book)
            time.sleep(0.3)
            return self.send_json({"ok": True, "status": AI.status(book)})
        if path == "/api/history":
            if body.get("book") and body.get("path"):
                self.user.add_history({
                    "book": body["book"], "path": body["path"],
                    "title": body.get("title") or body["path"],
                    "time": int(time.time()),
                })
            return self.send_json({"ok": True})
        if path == "/api/history/remove":
            self.user.remove_history(body.get("book", ""), body.get("path", ""))
            return self.send_json({"ok": True})
        if path == "/api/history/clear":
            self.user.clear_history()
            return self.send_json({"ok": True})
        if path == "/api/bookmarks/toggle":
            on = self.user.toggle_bookmark({
                "book": body.get("book", ""), "path": body.get("path", ""),
                "title": body.get("title") or body.get("path", ""),
                "time": int(time.time()),
            })
            return self.send_json({"ok": True, "bookmarked": on})
        if path == "/api/bookmarks/remove":
            self.user.remove_bookmark(body.get("book", ""), body.get("path", ""))
            return self.send_json({"ok": True})
        return self.not_found()

    # --- обработчики ----------------------------------------------------- #
    def static(self, name: str):
        file = (STATIC_DIR / name).resolve()
        if STATIC_DIR.resolve() not in file.parents or not file.is_file():
            return self.not_found()
        mime = mimetypes.guess_type(file.name)[0] or "application/octet-stream"
        if mime.startswith(TEXT_MIMES):
            mime += "; charset=utf-8"
        self.send_bytes(file.read_bytes(), mime)

    def content(self, raw: str):
        book_raw, _, entry_raw = raw.partition("/")
        book = self.library.get(unquote(book_raw))
        if not book:
            return self.not_found("Архив не найден")
        entry_path = unquote(entry_raw)
        if not entry_path:
            main = book.main_path()
            if not main:
                return self.not_found("В архиве нет главной страницы")
            return self.redirect(f"/content/{quote(book.id)}/{quote(main)}")
        entry = book.find_entry(entry_path)
        if entry is None:
            return self.not_found(f"Статья «{entry_path}» не найдена в архиве")
        if entry.is_redirect:
            target = entry.get_redirect_entry()
            return self.redirect(f"/content/{quote(book.id)}/{quote(target.path)}")
        item = entry.get_item()
        mime = item.mimetype or "application/octet-stream"
        if mime.startswith(TEXT_MIMES) and "charset" not in mime:
            mime += "; charset=utf-8"
        self.send_bytes(bytes(item.content), mime, cache=3600)

    def api_search(self, qs: dict):
        book, q = self.book_param(qs), qs.get("q", "").strip()
        if not book or not q:
            return self.send_json({"results": [], "total": 0})
        offset = max(int(qs.get("offset", 0) or 0), 0)
        limit = min(max(int(qs.get("limit", 15) or 15), 1), 50)
        t = time.time()
        # Пока идёт индексирование, кэш живёт недолго — индекс растёт.
        key = (q, int(time.time() // 30) if AI.running(book) else 0)
        res = book.results.get_or(key, lambda: smart_search(book, q))
        page = []
        for i, r in enumerate(res["results"][offset: offset + limit]):
            item = dict(r)
            item.update(book.snippet(r["path"]))
            page.append(item)
        out = {k: v for k, v in res.items() if k != "results"}
        out.update({"results": page, "total": len(res["results"]), "offset": offset,
                    "ms": int((time.time() - t) * 1000)})
        return self.send_json(out)

    def api_open(self):
        path = ask_zim_file()
        if not path:
            return self.send_json({"ok": False, "cancelled": True})
        try:
            book = self.library.add_file(Path(path))
        except Exception as e:
            return self.send_json({"ok": False, "error": str(e)})
        if not book:
            err = "; ".join(self.library.errors) or "Не удалось открыть файл"
            return self.send_json({"ok": False, "error": err})
        return self.send_json({"ok": True, "book": book.info()})


# --------------------------------------------------------------------------- #
#  Запуск
# --------------------------------------------------------------------------- #
def make_server(host: str, port: int) -> ThreadingHTTPServer:
    last_error = None
    for p in range(port, port + 20):
        try:
            srv = ThreadingHTTPServer((host, p), Handler)
            srv.daemon_threads = True
            return srv
        except OSError as e:
            last_error = e
    raise SystemExit(f"Не удалось занять порт ({last_error})")


def migrate_old_config():
    old = APP_DIR / "library.json"
    if old.is_file() and not CONFIG_FILE.exists():
        DATA_DIR.mkdir(exist_ok=True)
        shutil.move(str(old), CONFIG_FILE)


def main():
    parser = argparse.ArgumentParser(description="Просмотр Википедии офлайн (ZIM-файлы Kiwix)")
    parser.add_argument("files", nargs="*", help="ZIM-файлы для открытия")
    parser.add_argument("--dir", default=str(DEFAULT_ZIM_DIR),
                        help="папка с ZIM-файлами (по умолчанию ./zim)")
    parser.add_argument("--port", type=int, default=8765, help="порт (по умолчанию 8765)")
    parser.add_argument("--no-browser", action="store_true", help="не открывать браузер")
    args = parser.parse_args()

    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(errors="replace")
        except Exception:
            pass

    migrate_old_config()
    zim_dir = Path(args.dir)
    zim_dir.mkdir(parents=True, exist_ok=True)
    Handler.library = Library(zim_dir, [Path(f) for f in args.files])
    Handler.user = UserData(USER_FILE)

    server = make_server("127.0.0.1", args.port)
    url = f"http://127.0.0.1:{server.server_address[1]}/"

    books = Handler.library.books
    print("=" * 60)
    print(" Wiki Offline — Википедия без интернета")
    print("=" * 60)
    print(f" Папка с ZIM-файлами: {zim_dir.resolve()}")
    if books:
        for b in books.values():
            print(f"   • {b.title}  ({b.archive.article_count:,} статей)".replace(",", " "))
    else:
        print("   (ZIM-файлы не найдены — добавьте их в папку или через кнопку в программе)")
    for err in Handler.library.errors:
        print(f"   ! {err}")
    if not AI_AVAILABLE:
        print("\n ! Умный поиск недоступен: pip install -r requirements.txt")
    print(f"\n Откройте в браузере: {url}")
    print(" Для выхода закройте это окно или нажмите Ctrl+C\n")

    AI.autostart(Handler.library)
    if not args.no_browser:
        threading.Timer(0.6, webbrowser.open, [url]).start()

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nЗавершение работы.")
    finally:
        AI.stop_all()
        server.server_close()


if __name__ == "__main__":
    main()
