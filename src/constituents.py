"""
constituents.py — Recupero dei costituenti degli indici da fonti pubbliche gratuite.

Sostituisce l'endpoint EODHD `fundamentals/{INDEX}.INDX` (filter=Components), che
richiede il piano *Fundamentals Data* e NON è più incluso nell'abbonamento in uso.
Tutto il resto della dashboard (`/api/eod/...`, sia titoli sia indici `.INDX`)
continua a funzionare con la chiave EODHD.

Principi di progetto (nati da rotture osservate, non da teoria):

  1. **Catena di fonti, non fonte unica.** Ogni indice ha una primaria e più
     fallback indipendenti. Se tutte falliscono si usa uno snapshot statico
     versionato nel repo: la dashboard non si spegne mai.
  2. **Colonne per NOME, mai per posizione.** Una fonte usa `symbol` minuscolo,
     un'altra `Symbol` maiuscolo: leggere `iloc[:, 0]` restituisce i nomi delle
     società al posto dei ticker, e la rottura è muta.
  3. **La staleness non si vede dal conteggio.** Una lista vecchia di due mesi ha
     comunque 503 righe e supera qualunque controllo strutturale. Dove la fonte
     dichiara una data (`as_of`) la si legge e la si usa per declassarla.
  4. **Meglio un'eccezione che un ticker inventato.** Un simbolo malformato che
     diventa un ticker EODHD inesistente produce un buco silenzioso nel
     denominatore della breadth.
  5. **Zero dipendenze aggiuntive.** Solo `requests` + stdlib (`re`, `csv`, `io`).
     Niente `lxml`/`beautifulsoup4`: `requirements.txt` resta invariato.

Fonti verificate con download reale il 2026-08-10.
"""

from __future__ import annotations

import csv
import io
import re
import time
from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Callable, Optional

import requests

from .config import (
    CONSTITUENTS_TIMEOUT,
    CONSTITUENTS_MAX_STALE_DAYS,
    INDEX_CONFIG,
    MAX_RETRIES,
    RETRY_DELAY,
)
from .constituents_snapshot import SNAPSHOTS, SNAPSHOT_DATE


# ═══════════════════════════════════════════════════════════════════════════════
# USER-AGENT — due valori diversi, non è un refuso
# ═══════════════════════════════════════════════════════════════════════════════

# Wikimedia applica una robot policy (phabricator T400119): senza User-Agent
# descrittivo risponde 403 su /wiki/, /w/ e /api/rest_v1/. Di conseguenza
# `pandas.read_html(url)` diretto non funziona MAI su Wikipedia: usa urllib
# con lo UA di default.
UA_WMF = "BreadthMonitor/2.1 (Kriterion Quant; https://github.com/LukeGSW)"

# Akamai, davanti ad api.nasdaq.com, resetta la connessione se lo User-Agent
# contiene una email o un URL. Per quell'host serve una stringa nuda.
UA_PLAIN = "BreadthMonitor/2.1"


# ═══════════════════════════════════════════════════════════════════════════════
# RISULTATO DI UNA FONTE
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class ComponentsResult:
    """Esito del recupero costituenti, con la provenienza sempre allegata."""

    tickers:  list[str]                       # ticker già in formato EODHD
    source:   str                             # etichetta leggibile della fonte usata
    as_of:    Optional[date] = None           # data dichiarata dalla fonte, se presente
    warnings: list[str] = field(default_factory=list)
    degraded: bool = False                    # True se la fonte è stale o è lo snapshot

    @property
    def count(self) -> int:
        return len(self.tickers)


class ComponentsError(RuntimeError):
    """Nessuna fonte utilizzabile per l'indice richiesto."""


# ═══════════════════════════════════════════════════════════════════════════════
# HTTP
# ═══════════════════════════════════════════════════════════════════════════════

def _http_get(url: str, ua: str = UA_WMF, timeout: int = CONSTITUENTS_TIMEOUT) -> requests.Response:
    """GET con retry sui soli errori transitori (429, 5xx, timeout, connessione)."""
    last_exc: Optional[Exception] = None

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            resp = requests.get(url, headers={"User-Agent": ua}, timeout=timeout)
            resp.raise_for_status()
            return resp
        except requests.exceptions.HTTPError as e:
            status = e.response.status_code if e.response is not None else 0
            if status in (429, 500, 502, 503, 504) and attempt < MAX_RETRIES:
                time.sleep(RETRY_DELAY * attempt)
                last_exc = e
                continue
            raise
        except requests.exceptions.RequestException as e:
            last_exc = e
            if attempt < MAX_RETRIES:
                time.sleep(RETRY_DELAY * attempt)
                continue
            raise

    raise RuntimeError(f"GET fallita su {url}") from last_exc


# ═══════════════════════════════════════════════════════════════════════════════
# NORMALIZZAZIONE SIMBOLO → TICKER EODHD
# ═══════════════════════════════════════════════════════════════════════════════

# Un ticker USA è 1-5 lettere, con al più una share class di UNA lettera dopo il
# punto (BRK.B, BF.B). Ammetterne due lascerebbe passare i suffissi di borsa —
# `AAPL.MI` diventerebbe `AAPL-MI.US`, un ticker inesistente creato in silenzio.
_US_RE = re.compile(r"[A-Z]{1,5}(?:\.[A-Z])?")

# Un ticker XETRA è alfanumerico e deve contenere almeno una lettera. Le cifre
# fanno parte del simbolo e NON vanno normalizzate (DB1, G1A, G24, HEN3, HNR1,
# MUV2, SY1, VOW3) e possono anche stare in testa: `1COV` era il simbolo di
# Covestro quando era nel DAX. Il vincolo "almeno una lettera" serve a scartare
# i codici puramente numerici che compaiono nei file holdings.
_XETRA_RE = re.compile(r"(?=[A-Z0-9]*[A-Z])[A-Z0-9]{1,6}")


def to_eodhd_us(symbol: str) -> str:
    """
    Simbolo di listino USA → ticker EODHD.

    EODHD codifica le share class con il **trattino**, non con il punto:
    `BRK.B` → `BRK-B.US`, `BF.B` → `BF-B.US`. Sono gli unici due casi presenti
    oggi nell'unione S&P 500 + Nasdaq 100 (518 simboli).

    Raises:
        ValueError: se il simbolo non ha una forma riconoscibile. Preferibile a
                    costruire in silenzio un ticker inesistente.
    """
    s = str(symbol).strip().upper()
    if not _US_RE.fullmatch(s):
        raise ValueError(f"simbolo USA non riconosciuto: {symbol!r}")
    return s.replace(".", "-") + ".US"


def to_eodhd_xetra(symbol: str) -> str:
    """
    Simbolo Deutsche Börse → ticker EODHD.

    Nessuna eccezione: tutti e 40 i titoli DAX risolvono con `{SIMBOLO}.XETRA`,
    inclusi Airbus (`AIR.XETRA`) e QIAGEN (`QIA.XETRA`), che nel file holdings
    dell'ETF compaiono su venue diverse da Xetra ma sono regolarmente quotati lì.
    Usare `AIR.PA` / `QIA.AS` introdurrebbe un calendario di borsa disallineato.

    Raises:
        ValueError: se il simbolo non ha una forma riconoscibile.
    """
    s = str(symbol).strip().upper()
    if s.endswith(".DE"):          # fonti con suffisso stile Yahoo Finance
        s = s[:-3]
    if not _XETRA_RE.fullmatch(s):
        raise ValueError(f"simbolo XETRA non riconosciuto: {symbol!r}")
    return s + ".XETRA"


_CONVERTERS: dict[str, Callable[[str], str]] = {
    "US":    to_eodhd_us,
    "XETRA": to_eodhd_xetra,
}


def _to_eodhd(symbols: list[str], market: str) -> tuple[list[str], list[str]]:
    """
    Converte una lista di simboli grezzi in ticker EODHD.

    Returns:
        (ticker validi ordinati e deduplicati, simboli scartati)
    """
    convert = _CONVERTERS[market]
    tickers: set[str] = set()
    rejected: list[str] = []

    for sym in symbols:
        try:
            tickers.add(convert(sym))
        except ValueError:
            rejected.append(str(sym).strip())

    return sorted(tickers), rejected


# ═══════════════════════════════════════════════════════════════════════════════
# PARSER GENERICI
# ═══════════════════════════════════════════════════════════════════════════════

def _read_csv_rows(text: str) -> list[dict[str, str]]:
    """CSV → lista di dict. Rimuove il BOM e normalizza gli header a minuscolo."""
    reader = csv.DictReader(io.StringIO(text.lstrip("﻿")))
    rows = []
    for row in reader:
        rows.append({
            (k or "").strip().lower(): (v or "").strip()
            for k, v in row.items()
        })
    return rows


def _column(rows: list[dict[str, str]], *candidates: str) -> str:
    """
    Individua una colonna **per nome** fra più alias possibili.

    Leggere la colonna per posizione è la rottura muta più costosa di tutta questa
    pipeline: se la fonte riordina i campi si ottengono i nomi delle società al
    posto dei ticker, con conteggio corretto e nessuna eccezione.
    """
    if not rows:
        raise ValueError("nessuna riga da cui dedurre le colonne")
    available = set(rows[0].keys())
    for cand in candidates:
        if cand.lower() in available:
            return cand.lower()
    raise ValueError(
        f"nessuna delle colonne {candidates} trovata; presenti: {sorted(available)}"
    )


def _parse_date(raw: str) -> Optional[date]:
    """Parsing tollerante delle date dichiarate dalle fonti (formati eterogenei)."""
    raw = (raw or "").strip().strip('"')
    if not raw:
        return None
    for fmt in ("%Y-%m-%d", "%d.%b.%Y", "%d-%b-%Y", "%d/%m/%Y", "%b %d, %Y", "%Y/%m/%d"):
        try:
            return datetime.strptime(raw, fmt).date()
        except ValueError:
            continue
    return None


# Le due pagine Wikipedia usano formati di tabella DIVERSI e servono entrambi.
#
#   S&P 500     →  |{{NYSE link|MMM}}   e anche  |{{BZX link|CBOE}}
#                  Il nome del template può contenere spazi e la riga può avere un
#                  commento HTML in coda (BRK.B, BF.B): con regex più strette si
#                  perdevano proprio le share class.
#   Nasdaq 100  →  | ADBE || [[Adobe Inc.]] || Technology || Software
#                  Tabella in chiaro, nessun template.
_WIKI_TEMPLATE_RE = re.compile(r"^\|\s*\{\{[^|{}]+\|([^}|]+)\}\}", re.M)
_WIKI_PLAIN_RE    = re.compile(r"^\|\s*([A-Z][A-Z0-9.\-]{0,5})\s*\|\|", re.M)


def _wikitext_tickers(section: str) -> list[str]:
    """
    Estrae i ticker da una sezione di wikitext, in entrambi i formati di tabella.

    Restituisce il risultato più ricco fra i due pattern invece di fermarsi al
    primo che matcha: una tabella mista darebbe altrimenti una lista parziale.
    """
    by_template = _WIKI_TEMPLATE_RE.findall(section)
    by_plain    = _WIKI_PLAIN_RE.findall(section)
    return by_template if len(by_template) >= len(by_plain) else by_plain

# Celle di tabella HTML (Wikipedia REST): <td>SAP</td>
_HTML_CELL_RE = re.compile(r"<t[dh][^>]*>(.*?)</t[dh]>", re.S | re.I)
_HTML_ROW_RE = re.compile(r"<tr[^>]*>(.*?)</tr>", re.S | re.I)
_HTML_TABLE_RE = re.compile(r"<table[^>]*>(.*?)</table>", re.S | re.I)
_TAG_RE = re.compile(r"<[^>]+>")


def _strip_html(cell: str) -> str:
    """Testo visibile di una cella HTML."""
    txt = _TAG_RE.sub("", cell)
    txt = txt.replace("&nbsp;", " ").replace("&amp;", "&").replace("\xa0", " ")
    return txt.strip()


def _wikitext_section(wikitext: str, start_id: str, *end_ids: str) -> str:
    """
    Isola la sezione di wikitext che contiene la tabella dei costituenti.

    Se l'ancora non esiste più **non** si fa lo slice: restituire l'intero
    documento produrrebbe una lista plausibile ma sbagliata, oppure zero simboli
    in silenzio. Meglio fallire e passare alla fonte successiva.
    """
    start = wikitext.find(start_id)
    if start == -1:
        raise ValueError(f"ancora {start_id!r} non trovata nel wikitext")

    end = len(wikitext)
    for end_id in end_ids:
        pos = wikitext.find(end_id, start)
        if pos != -1:
            end = min(end, pos)
    return wikitext[start:end]


def _html_table_column(html: str, header_names: tuple[str, ...]) -> list[str]:
    """
    Estrae una colonna da una tabella HTML cercando l'intestazione **per nome**.

    Scorre tutte le tabelle del documento: fissare l'indice 0 si rompe appena
    gli editor di Wikipedia inseriscono un box in cima alla pagina.
    """
    for table in _HTML_TABLE_RE.findall(html):
        rows = _HTML_ROW_RE.findall(table)
        if not rows:
            continue

        headers = [_strip_html(c).lower() for c in _HTML_CELL_RE.findall(rows[0])]
        col_idx = next(
            (i for i, h in enumerate(headers) if h in header_names),
            None,
        )
        if col_idx is None:
            continue

        values = []
        for row in rows[1:]:
            cells = _HTML_CELL_RE.findall(row)
            if len(cells) > col_idx:
                val = _strip_html(cells[col_idx])
                if val:
                    values.append(val)
        if values:
            return values

    raise ValueError(f"nessuna tabella con intestazione fra {header_names}")


# ═══════════════════════════════════════════════════════════════════════════════
# FONTI — S&P 500
# ═══════════════════════════════════════════════════════════════════════════════

def _sp500_chinobing() -> tuple[list[str], Optional[date]]:
    """
    GitHub `chinobing/historical_sp500_constituents` (MIT), rigenerato da una
    GitHub Action ogni 12 ore a partire da Wikipedia.

    È l'unica fonte S&P 500 che **dichiara la propria data** (colonna `date`):
    permette di accorgersi se si è congelata, cosa che nessun controllo
    strutturale sul conteggio riesce a fare.
    """
    url = ("https://raw.githubusercontent.com/chinobing/"
           "historical_sp500_constituents/main/sp500_constituents.csv")
    rows = _read_csv_rows(_http_get(url, ua=UA_PLAIN).text)

    sym_col = _column(rows, "symbol")
    symbols = [r[sym_col] for r in rows if r.get(sym_col)]

    as_of = None
    if rows and "date" in rows[0]:
        as_of = _parse_date(rows[0]["date"])

    return symbols, as_of


def _sp500_datasets() -> tuple[list[str], Optional[date]]:
    """
    GitHub `datasets/s-and-p-500-companies` (ODC-PDDL, 660+ stelle).

    Non dichiara alcuna data ed è **già rimasto congelato 204 giorni** continuando
    a rispondere 200: adatto solo come fallback, mai come primaria.
    """
    url = ("https://raw.githubusercontent.com/datasets/"
           "s-and-p-500-companies/main/data/constituents.csv")
    rows = _read_csv_rows(_http_get(url, ua=UA_PLAIN).text)

    sym_col = _column(rows, "symbol")
    return [r[sym_col] for r in rows if r.get(sym_col)], None


def _sp500_wikipedia() -> tuple[list[str], Optional[date]]:
    """Wikipedia EN, wikitext grezzo. Fonte a monte delle due precedenti."""
    url = ("https://en.wikipedia.org/w/index.php"
           "?title=List_of_S%26P_500_companies&action=raw")
    wikitext = _http_get(url, ua=UA_WMF).text

    section = _wikitext_section(wikitext, 'id="constituents"', 'id="changes"')
    return _wikitext_tickers(section), None


# ═══════════════════════════════════════════════════════════════════════════════
# FONTI — NASDAQ 100
# ═══════════════════════════════════════════════════════════════════════════════

def _ndx_wikipedia() -> tuple[list[str], Optional[date]]:
    """
    Wikipedia EN `List of NASDAQ-100 companies`, wikitext grezzo.

    Nota: la pagina `Nasdaq-100` non contiene più la tabella dei costituenti.
    """
    url = ("https://en.wikipedia.org/w/index.php"
           "?title=List_of_NASDAQ-100_companies&action=raw")
    wikitext = _http_get(url, ua=UA_WMF).text

    section = _wikitext_section(wikitext, 'id="constituents"', 'id="changes"',
                                "==Changes", "== Changes")
    return _wikitext_tickers(section), None


def _ndx_nasdaq_api() -> tuple[list[str], Optional[date]]:
    """
    Endpoint pubblico Nasdaq. Richiede `UA_PLAIN`: con uno User-Agent che
    contenga una email o un URL, Akamai resetta la connessione senza rispondere.
    """
    url = "https://api.nasdaq.com/api/quote/list-type/nasdaq100"
    payload = _http_get(url, ua=UA_PLAIN, timeout=10).json()

    rows = (payload.get("data") or {}).get("data", {}).get("rows") or []
    return [r["symbol"] for r in rows if r.get("symbol")], None


def _ndx_ishares() -> tuple[list[str], Optional[date]]:
    """iShares Nasdaq 100 UCITS ETF (CNDX) — lineage indipendente da Wikipedia."""
    url = ("https://www.ishares.com/de/privatanleger/de/produkte/253741/"
           "ishares-nasdaq-100-ucits-etf/1478358465952.ajax"
           "?fileType=csv&fileName=CNDX_holdings&dataType=fund")
    return _parse_ishares_csv(_http_get(url, ua=UA_PLAIN).content)


# ═══════════════════════════════════════════════════════════════════════════════
# FONTI — DAX 40
# ═══════════════════════════════════════════════════════════════════════════════

_ISHARES_TICKER_COLS = ("emittententicker", "ticker", "issuer ticker")
_ISHARES_CLASS_COLS  = ("anlageklasse", "asset class")
_ISHARES_EQUITY      = {"aktien", "equity"}


def _parse_ishares_csv(raw: bytes) -> tuple[list[str], Optional[date]]:
    """
    Parser dei file holdings di iShares/BlackRock.

    Due accorgimenti, entrambi nati da rotture riprodotte in laboratorio:

      * la riga di intestazione si trova **per contenuto**, non con `skiprows=2`:
        con una riga di preambolo in più o in meno il parser posizionale perde
        silenziosamente il primo titolo oppure restituisce colonne vuote;
      * il filtro sull'asset class accetta sia l'etichetta tedesca (`Aktien`) sia
        quella inglese (`Equity`) e **solleva** se il risultato è vuoto: la
        localizzazione dell'etichetta produrrebbe altrimenti zero titoli senza
        alcun errore.
    """
    text = raw.decode("utf-8-sig", errors="replace")
    lines = text.splitlines()

    as_of = None
    if lines:
        first = lines[0].split(",", 1)
        if len(first) == 2:
            as_of = _parse_date(first[1])

    hdr_idx = next(
        (i for i, line in enumerate(lines)
         if any(line.lower().lstrip('"').startswith(c) for c in _ISHARES_TICKER_COLS)),
        None,
    )
    if hdr_idx is None:
        raise ValueError("iShares: riga di intestazione non individuata")

    rows = _read_csv_rows("\n".join(lines[hdr_idx:]))
    tick_col = _column(rows, *_ISHARES_TICKER_COLS)

    try:
        class_col = _column(rows, *_ISHARES_CLASS_COLS)
        equities = [r for r in rows if r.get(class_col, "").lower() in _ISHARES_EQUITY]
        if not equities:
            raise ValueError(
                "iShares: 0 righe azionarie dopo il filtro sull'asset class "
                "(etichetta localizzata cambiata?)"
            )
    except ValueError as e:
        if "0 righe azionarie" in str(e):
            raise
        equities = rows      # colonna asset class assente: si tiene tutto

    # Seconda linea di difesa dopo il filtro sull'asset class: scarta liquidità,
    # collaterale, future e ISIN-like usati come pseudo-ticker.
    symbols = [
        r[tick_col] for r in equities
        if _XETRA_RE.fullmatch(r.get(tick_col, "").strip())
    ]
    return symbols, as_of


def _dax_ishares_de() -> tuple[list[str], Optional[date]]:
    """iShares Core DAX UCITS ETF (EXS1), sito tedesco. Nessuno header richiesto."""
    url = ("https://www.ishares.com/de/privatanleger/de/produkte/251464/"
           "ishares-core-dax-ucits-etf-de-fund/1478358465952.ajax"
           "?fileType=csv&dataType=fund")
    return _parse_ishares_csv(_http_get(url, ua=UA_PLAIN).content)


def _dax_ishares_ch() -> tuple[list[str], Optional[date]]:
    """
    Stesso fondo, mirror svizzero: l'id dell'endpoint AEM è diverso, quindi
    sopravvive a una rotazione dell'id sul sito tedesco.
    """
    url = ("https://www.ishares.com/ch/professionelle-anleger/de/produkte/251464/"
           "ishares-core-dax-ucits-etf-de-fund/1495092304805.ajax"
           "?fileType=csv&dataType=fund")
    return _parse_ishares_csv(_http_get(url, ua=UA_PLAIN).content)


def _dax_wikipedia_de() -> tuple[list[str], Optional[date]]:
    """
    Wikipedia **tedesca**, endpoint REST versionato.

    La pagina inglese `en.wikipedia.org/wiki/DAX` va evitata: restituisce comunque
    40 righe (quindi un controllo `len == 40` la promuove) ma è sbagliata — riporta
    `PAH3` al posto di `HOT` e scrive il ticker di Airbus come `AIR.PA`.
    """
    url = "https://de.wikipedia.org/api/rest_v1/page/html/DAX"
    html = _http_get(url, ua=UA_WMF).text
    return _html_table_column(html, ("symbol", "kürzel", "ticker")), None


def _dax_wikipedia_de_plain() -> tuple[list[str], Optional[date]]:
    """Wikipedia tedesca, pagina HTML normale (stesso parser del REST)."""
    html = _http_get("https://de.wikipedia.org/wiki/DAX", ua=UA_WMF).text
    return _html_table_column(html, ("symbol", "kürzel", "ticker")), None


# ═══════════════════════════════════════════════════════════════════════════════
# CATENE DI FONTI PER INDICE
# ═══════════════════════════════════════════════════════════════════════════════

#  index_key → (mercato, [(etichetta, funzione), ...])
_PROVIDERS: dict[str, tuple[str, list[tuple[str, Callable[[], tuple[list[str], Optional[date]]]]]]] = {
    "SP500": ("US", [
        ("GitHub chinobing (aggiornato ogni 12h)", _sp500_chinobing),
        ("GitHub datasets/s-and-p-500-companies", _sp500_datasets),
        ("Wikipedia EN (wikitext)",               _sp500_wikipedia),
    ]),
    "NASDAQ": ("US", [
        ("Wikipedia EN (wikitext)",  _ndx_wikipedia),
        ("api.nasdaq.com",           _ndx_nasdaq_api),
        ("iShares Nasdaq 100 UCITS", _ndx_ishares),
    ]),
    "DAX": ("XETRA", [
        ("iShares Core DAX UCITS (DE)", _dax_ishares_de),
        ("iShares Core DAX UCITS (CH)", _dax_ishares_ch),
        ("Wikipedia DE (REST)",         _dax_wikipedia_de),
        ("Wikipedia DE (HTML)",         _dax_wikipedia_de_plain),
    ]),
}


# ═══════════════════════════════════════════════════════════════════════════════
# CONTROLLI DI PLAUSIBILITÀ
# ═══════════════════════════════════════════════════════════════════════════════

def _check_plausible(index_key: str, tickers: list[str]) -> None:
    """
    Verifica che la lista sia plausibile prima di accettarla.

    Il controllo sul conteggio da solo non basta (una lista stale ha il conteggio
    giusto), ma intercetta il caso opposto: parser che restituisce mezza tabella,
    o zero simboli, o l'intera pagina.

    Raises:
        ValueError: se il conteggio è fuori range o mancano i titoli sentinella.
    """
    cfg = INDEX_CONFIG[index_key]
    lo, hi = cfg["expected_count"]

    if not lo <= len(tickers) <= hi:
        raise ValueError(
            f"conteggio implausibile: {len(tickers)} costituenti, attesi {lo}-{hi}"
        )

    missing = [t for t in cfg["sentinel_tickers"] if t not in tickers]
    if missing:
        raise ValueError(f"titoli sentinella mancanti: {', '.join(missing)}")


# ═══════════════════════════════════════════════════════════════════════════════
# API PUBBLICA
# ═══════════════════════════════════════════════════════════════════════════════

def get_index_components(index_key: str, today: Optional[date] = None) -> ComponentsResult:
    """
    Restituisce i costituenti correnti di un indice in formato ticker EODHD.

    Percorre la catena di fonti dell'indice e si ferma sulla prima che supera i
    controlli di plausibilità **ed** è aggiornata. Una fonte plausibile ma stale
    (o priva di data dichiarata, quando esiste un'alternativa datata) viene
    tenuta da parte come ripiego invece di essere scartata: meglio una lista di
    ieri che nessuna lista. Se nessuna fonte risponde si usa lo snapshot statico
    incluso nel repository, così la dashboard resta utilizzabile offline.

    Args:
        index_key: Chiave di INDEX_CONFIG ('SP500', 'NASDAQ', 'DAX')
        today:     Data di riferimento per il calcolo della staleness (per i test)

    Returns:
        ComponentsResult con ticker, fonte, as_of, warning ed eventuale flag degraded

    Raises:
        ComponentsError: se nessuna fonte è utilizzabile e manca lo snapshot
    """
    if index_key not in _PROVIDERS:
        raise ComponentsError(f"indice non mappato: {index_key!r}")

    today = today or date.today()
    market, providers = _PROVIDERS[index_key]

    warnings: list[str] = []
    fallback: Optional[ComponentsResult] = None
    seen_stale = False

    for label, provider in providers:
        try:
            raw_symbols, as_of = provider()
        except Exception as e:                                    # noqa: BLE001
            warnings.append(f"{label}: non disponibile ({type(e).__name__}: {e})")
            continue

        tickers, rejected = _to_eodhd(raw_symbols, market)

        try:
            _check_plausible(index_key, tickers)
        except ValueError as e:
            warnings.append(f"{label}: scartata — {e}")
            continue

        if rejected:
            warnings.append(
                f"{label}: {len(rejected)} simboli non convertibili e ignorati "
                f"({', '.join(rejected[:8])}{'…' if len(rejected) > 8 else ''})"
            )

        stale_days = (today - as_of).days if as_of else None
        is_stale = stale_days is not None and stale_days > CONSTITUENTS_MAX_STALE_DAYS

        if is_stale:
            seen_stale = True
            warnings.append(
                f"{label}: dati fermi al {as_of:%d/%m/%Y} ({stale_days} giorni fa), "
                "provo la fonte successiva"
            )
            if fallback is None:
                fallback = ComponentsResult(tickers, label, as_of, [], degraded=True)
            continue

        # Una fonte che non dichiara alcuna data non può essere provata stale.
        # Se però la fonte precedente lo era, il dubbio va reso esplicito invece
        # di essere sepolto: potrebbero essersi congelate entrambe.
        if as_of is None and seen_stale:
            warnings.append(
                f"{label}: non dichiara una data di aggiornamento, quindi la sua "
                "freschezza non è verificabile — ed è stata scelta dopo una fonte "
                "risultata stale."
            )

        return ComponentsResult(tickers, label, as_of, warnings, degraded=False)

    # ── Nessuna fonte fresca: si ripiega sulla migliore stale, poi sullo snapshot ──
    if fallback is not None:
        fallback.warnings = warnings + [
            "Nessuna fonte aggiornata disponibile: uso la più recente fra quelle stale."
        ]
        return fallback

    snapshot = SNAPSHOTS.get(index_key)
    if snapshot:
        return ComponentsResult(
            tickers=sorted(snapshot),
            source=f"snapshot statico incluso nel repo ({SNAPSHOT_DATE:%d/%m/%Y})",
            as_of=SNAPSHOT_DATE,
            warnings=warnings + [
                "Tutte le fonti online hanno fallito: la lista è quella congelata "
                "nel repository e non riflette i ribilanciamenti successivi."
            ],
            degraded=True,
        )

    raise ComponentsError(
        f"Nessuna fonte utilizzabile per {index_key}.\n" + "\n".join(warnings)
    )
