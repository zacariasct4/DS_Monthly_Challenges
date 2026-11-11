"""Interpretable English employment-title features for tabular ECL research.

Requires pandas, numpy and scikit-learn. No network calls or target encoding.
Rules are corpus-informed heuristics, not validated occupational ground truth.
"""
from __future__ import annotations

import html
import re
import unicodedata
from collections import Counter
from functools import lru_cache
from typing import Mapping

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted

RULE_VERSION = "1.0.0"
PLACEHOLDERS = frozenset({"unknown", "n a", "na", "none", "null", "nan", "not available", "not provided", "unspecified", "test"})

# Explicit, reviewable edits only. Short ambiguous acronyms (PA, MA, PM, QA)
# are deliberately not expanded into an assumed occupation.
ALIASES = {
    "mgr": "manager", "mngr": "manager", "manger": "manager",
    "mananger": "manager", "managment": "management", "mgmt": "management",
    "mgt": "management", "asst": "assistant", "supv": "supervisor",
    "supvr": "supervisor", "supervisior": "supervisor", "supt": "superintendent",
    "sr": "senior", "jr": "junior", "dept": "department", "dir": "director",
    "rep": "representative", "reps": "representative", "engr": "engineer",
    "acctg": "accounting", "exec": "executive", "adminstrative": "administrative",
    "maintanance": "maintenance", "maintenence": "maintenance",
    "maintainance": "maintenance", "maint": "maintenance",
    "technican": "technician", "technition": "technician", "technitian": "technician",
    "mecanic": "mechanic", "machanic": "mechanic", "forman": "foreman",
    "operater": "operator", "machinest": "machinist", "electrican": "electrician",
    "tracher": "teacher", "techer": "teacher", "nures": "nurse",
    "superviser": "supervisor", "superintendant": "superintendent",
    "wharehouse": "warehouse", "warehous": "warehouse",
    "rn": "registered nurse", "lpn": "licensed practical nurse",
    "lvn": "licensed vocational nurse", "cna": "certified nursing assistant",
    "crna": "certified registered nurse anesthetist",
    "cpa": "certified public accountant", "csr": "customer service representative",
    "hr": "human resources", "vp": "vice president", "svp": "senior vice president",
    "avp": "assistant vice president", "evp": "executive vice president",
    "ceo": "chief executive officer", "cfo": "chief financial officer",
    "coo": "chief operating officer", "cto": "chief technology officer",
    "cio": "chief information officer", "gm": "general manager",
    "sgt": "sergeant", "emt": "emergency medical technician",
}
PHRASES = {
    "sys admin": "systems administrator", "sysadmin": "systems administrator",
    "systems admin": "systems administrator", "network admin": "network administrator",
    "tech support": "technical support", "vet tech": "veterinary technician",
    "book keeper": "bookkeeper", "truckdriver": "truck driver",
    "mailman": "mail carrier", "mailcarrier": "mail carrier",
    "lettercarrier": "letter carrier", "teamleader": "team leader",
    "salesmanager": "sales manager", "wal mart": "walmart",
    "care giver": "caregiver", "hairstylist": "hair stylist", "mailhandler": "mail handler",
    "home health aid": "home health aide",
    "selfemployed": "self employed", "co owner": "owner", "coowner": "owner",
    "co founder": "founder", "cofounder": "founder", "hairdresser": "hair stylist",
}

def _boundary(pattern):
    return re.compile(r"\b(?:" + pattern + r")\b")

_ALIASES_RE = _boundary("|".join(sorted(map(re.escape, ALIASES), key=lambda x: (-len(x), x))))
_PHRASE_RE = _boundary("|".join(sorted(map(re.escape, PHRASES), key=lambda x: (-len(x), x))))

@lru_cache(maxsize=250_000)
def _normalize_text(text: str) -> str:
    text = html.unescape(text).casefold()
    text = "".join(c for c in unicodedata.normalize("NFKD", text) if not unicodedata.combining(c))
    text = text.replace("c++", " cplusplus ").replace("c#", " csharp ").replace("&", " and ")
    # Join dotted acronyms without joining ordinary words separated by a dot.
    text = re.sub(r"\b(?:[a-z]\.){2,}", lambda m: m[0].replace(".", ""), text)
    text = re.sub(r"[^a-z0-9]+", " ", text).strip()
    text = _PHRASE_RE.sub(lambda m: PHRASES[m[0]], text)
    text = _ALIASES_RE.sub(lambda m: ALIASES[m[0]], text)
    return re.sub(r"\s+", " ", text).strip()

def normalize_emp_title(value):
    """Return normalized text, preserving actual missing values as pd.NA."""
    if pd.isna(value):
        return pd.NA
    return _normalize_text(str(value))

# These categories describe occupational functions, NOT the employer's industry.
# All matching specific families are retained. Conflicts become 'ambiguous'.
ROLE_PATTERNS = {
    "healthcare": r"nurs\w*|nursing|physician|doctor|surgeon|surgical|medical assistant|medical technician|emergency medical|paramedic|pharmac\w*|dental|dentist|hygienist|radiolog\w*|sonograph\w*|ultrasound|phlebotom\w*|anesthe\w*|respiratory therapist|physical therapist|occupational therapist|speech therapist|veterinar\w*|caregiver|home health aide|patient care|clinical|clinician|psychologist|psychiatr\w*|dietitian|nutritionist|optometr\w*|optician|dialysis|therapist|physiologist",
    "education": r"teacher|teaching|professor|instructor|educator|librarian|paraprofessional|paraeducator|school principal|assistant principal|vice principal|school counselor|guidance counselor|dean|faculty|lecturer|tutor|school superintendent|superintendent of schools",
    "software_it_data": r"software|developer|programmer|information technology|information systems|information security|cyber\w*|network (?:engineer|administrator|technician)|systems? (?:administrator|engineer|analyst)|database|dba|web (?:designer|developer)|help ?desk|desktop support|technical support|devops|data (?:scientist|analyst|engineer|architect)|solutions architect|scrum master|chief (?:technology|information) officer|it (?:manager|director|specialist|technician|support|analyst|consultant|engineer)",
    "accounting_finance_insurance": r"accountant|accounting|accounts? payable|accounts? receivable|bookkeep\w*|controller|comptroller|auditor|audit|finance|financial|banker|banking|teller|loan officer|loan processor|loan originator|mortgage|underwrit\w*|insurance|claims adjuster|claims examiner|actuar\w*|payroll|tax preparer|investment|stockbroker|wealth|treasury|treasurer|chief financial officer|credit analyst|credit manager|loan servicing|collections? specialist",
    "legal": r"attorney|lawyer|paralegal|legal|judge|judicial|counsel|prosecutor|magistrate|court clerk",
    "protective_services": r"police|sheriff|correctional|corrections|correction officer|detention officer|probation officer|parole officer|trooper|detective|firefighter|fire fighter|fireman|security (?:officer|guard)|armed guard|loss prevention|border patrol|law enforcement|public safety|911|sergeant|military|soldier|infantry|corporal|airman",
    "sales": r"sales|salesman|account (?:manager|executive)|business development|cashier|retail|store manager|merchandis\w*|purchaser|buyer",
    "customer_service": r"customer (?:service|support|care)|call center|contact center|client service|customer success",
    "marketing_media_design": r"marketing|advertising|public relations|graphic design\w*|graphic artist|journalist|reporter|editor|copywriter|photographer|videographer|musician|social media|public affairs|communications? (?:manager|director|specialist)|art director|creative director",
    "social_community_services": r"social work\w*|case ?worker|case manager|social services|human services|pastor|minister|chaplain|clergy|priest|rabbi|counselor|direct support|youth worker",
    "transport_logistics": r"driver|driving|truck|bus operator|pilot|flight attendant|dispatcher|warehouse|logistics|shipping|receiving|freight|supply chain|forklift|mail (?:carrier|handler)|letter carrier|postal|conductor|locomotive|longshoreman|air traffic|delivery|dock worker|order filler|material handler",
    "food_hospitality": r"bartender|chef|cook|baker|waiter|waitress|barista|restaurant|food service|banquet|hotel manager|concierge|casino|table games|casino dealer",
    "construction_trades": r"carpenter|electrician|plumber|plumbing|welder|welding|hvac|roofer|roofing|mason|ironworker|pipe ?fitter|millwright|drywall|construction|crane operator|heavy equipment operator|sheet metal|boilermaker|lineman|journeyman|apprentice|painter|foreman|laborer|labor|superintendent|landscap\w*|groundskeeper|estimator|surveyor",
    "manufacturing_production": r"machinist|machine operator|production|manufacturing|assembler|assembly|factory|cnc|toolmaker|tool and die|plant manager|fabricator|press operator|packaging|packer|machining",
    "maintenance_repair": r"mechanic|maintenance|repair|service technician|service tech|field service|handyman|facilities|facility manager|locksmith|auto body",
    "personal_cleaning_services": r"custodian|custodial|janitor|janitorial|housekeep\w*|cleaner|cleaning|hair stylist|barber|cosmetolog\w*|manicurist|esthetician|nail technician|nanny|babysitter|personal trainer|fitness instructor|groomer|laundry|tailor|seamstress|sanitation",
    "real_estate": r"realtor|real estate|property manag\w*|leasing (?:agent|consultant|manager)|appraiser|landlord",
    "human_resources": r"human resources|recruit\w*|talent acquisition|staffing|hrbp|personnel|benefits administrator",
    "agriculture_energy": r"farmer|farming|rancher|agricultur\w*|oilfield|drilling|driller|miner|mining|petroleum|forester|forestry|rig operator|landman|roughneck|roustabout|derrick hand",
}

# Weak/general words describe functions without forcing an industry.
GENERIC_PATTERNS = {
    "administrative_support": r"administrative|executive assistant|office|secretary|receptionist|clerical|clerk|data entry|scheduler|records|admin",
    "engineering_science": r"engineer|engineering|architect|scientist|chemist|biologist|geologist|physicist|laboratory|lab|statistician|research",
    "quality_safety": r"quality|quality assurance|quality control|safety|inspector|inspection|compliance",
    "management": r"manager|management|supervisor|director|president|chief|executive|leader|team lead|head|superintendent",
    "ownership": r"owner|founder|proprietor|entrepreneur|partner",
    "technical_unspecified": r"technician|tech|technical|technologist",
    "operations_unspecified": r"operator|operations|operation",
    "analysis_consulting_unspecified": r"analyst|consultant|consulting|advisor|adviser",
    "support_unspecified": r"assistant|associate|specialist|coordinator|agent|representative|worker|employee|officer|staff|support|service",
}

# Industry requires employer/context evidence. A medical accountant remains an
# accounting role; 'accountant' alone cannot establish the employer's industry.
INDUSTRY_PATTERNS = {
    "healthcare": r"hospital|medical center|healthcare|health system|clinic|kaiser permanente|mayo clinic",
    "education": r"school|schools|school district|university|college|academy|board of education|isd",
    "financial_services": r"bank|credit union|wells fargo|bank of america|jpmorgan|jp morgan|citibank|goldman sachs|morgan stanley|capital one|insurance company|state farm|allstate|geico|prudential|metlife",
    "retail": r"walmart|target|home depot|lowes|costco|sams club|kroger|safeway|publix|macys|sears|best buy|walgreens|cvs|retail|department store",
    "technology_telecom": r"microsoft|google|ibm|oracle|cisco|dell|verizon|comcast|at and t|telecommunications|telecom|software company",
    "transport_postal": r"usps|united states postal|postal service|ups|fedex|amtrak|airlines|railroad|railway|trucking company",
    "manufacturing": r"boeing|lockheed|raytheon|northrop|general motors|ford motor|general electric|manufacturing company",
    "hospitality_food": r"hotel|resort|restaurant|starbucks|mcdonalds|hilton|marriott|casino",
    "government": r"city of|county of|state of|town of|government|municipal|department of|irs|usda|social security administration",
    "military": r"army|navy|air force|usaf|usmc|marine corps|national guard|department of defense",
    "construction": r"construction company|construction inc|construction llc",
    "energy_utilities": r"exxon|exxonmobil|chevron|schlumberger|halliburton|electric company|power company|national grid",
}

_LEGAL = _boundary(r"inc|incorporated|llc|ltd|corp|corporation|llp|pllc|company")
_ORG = _boundary(r"hospital|medical center|university|college|school district|bank|credit union|city of|county of|state of|department of|foundation|church|industries|technologies|associates|holdings")
_BRANDS = _boundary(r"walmart|wells fargo|bank of america|microsoft|google|ibm|oracle|cisco|dell|boeing|fedex|usps|ups|costco|verizon|comcast|starbucks|jpmorgan|kaiser|home depot|capital one")
_ROLE_WORD = _boundary(r"manager|supervisor|director|president|chief|officer|assistant|engineer|analyst|teacher|nurse|driver|technician|operator|representative|associate|clerk|specialist|coordinator|accountant|consultant|worker|employee|owner|founder|partner|administrator|professor|instructor|teller|mechanic|custodian|janitor|cashier|server|cook|chef|physician|therapist|attorney|agent|laborer|welder|electrician|recruiter|foreman|buyer|machinist|dispatcher|pilot|receptionist|secretary|bookkeeper|bartender|sergeant|sales|pharmacist|superintendent|realtor|pastor")
_MANAGEMENT = _boundary(r"manager|management|supervisor|director|president|chief|head|team lead|team leader|foreman|superintendent")
_OWNER = _boundary(r"owner|founder|proprietor|entrepreneur|partner|self employed")
_STATUS = {
    "retired": re.compile(r"^(?:retired|retiree)(?:\b.*)?$"),
    "unemployed": re.compile(r"^(?:unemployed|not employed|not working|between jobs)$"),
    "student": re.compile(r"^(?:(?:graduate|phd|college|full time) )?student$"),
    "homemaker": re.compile(r"^(?:homemaker|stay at home (?:mom|dad|parent))$"),
    "disabled": re.compile(r"^(?:disabled|on disability)$"),
}

# Additions found in the highest-frequency residuals of the supplied TRAIN CSV.
# Broad head nouns stay in a general function category instead of forcing a sector.
ROLE_PATTERNS["healthcare"] += r"|speech (?:language )?pathologist|home health aide|medical (?:biller|coder)|hha"
ROLE_PATTERNS["healthcare"] += r"|psychotherapist|chiropractor|radiographer"
ROLE_PATTERNS["education"] += r"|registrar|instructional designer"
ROLE_PATTERNS["transport_logistics"] += r"|courier|city carrier|rural carrier|postmaster|transportation|warehouseman|order selector|inventory control"
ROLE_PATTERNS["protective_services"] += r"|lieutenant|fire captain|us army|us navy|usaf|usmc"
ROLE_PATTERNS["food_hospitality"] += r"|food server|poker dealer"
ROLE_PATTERNS["sales"] += r"|purchasing"
ROLE_PATTERNS["manufacturing_production"] += r"|pressman|meat cutter"
ROLE_PATTERNS["accounting_finance_insurance"] += r"|adjuster|trader"
# Bare foreman/superintendent are supervisory functions across many industries.
ROLE_PATTERNS["construction_trades"] = ROLE_PATTERNS["construction_trades"].replace("|foreman", "").replace("|superintendent", "") + r"|construction (?:foreman|superintendent)"
ROLE_PATTERNS["accounting_finance_insurance"] = ROLE_PATTERNS["accounting_finance_insurance"].replace("|controller", r"|(?<!traffic )(?<!production )controller")
ROLE_PATTERNS["transport_logistics"] = ROLE_PATTERNS["transport_logistics"].replace("|driving", "")
GENERIC_PATTERNS["administrative_support"] += r"|administrator|administration|front desk"
GENERIC_PATTERNS["management"] += r"|lead|managing member|foreman"
GENERIC_PATTERNS["support_unspecified"] += r"|planner|processor|trainer|team member|contractor"
GENERIC_PATTERNS["design_unspecified"] = r"designer|design"
GENERIC_PATTERNS["security_unspecified"] = r"security|investigator|deputy|captain"
GENERIC_PATTERNS["installation_unspecified"] = r"installer|installation"
GENERIC_PATTERNS["commercial_unspecified"] = r"broker|dealer|producer|billing|collector|biller"
GENERIC_PATTERNS["service_unspecified"] = r"server|porter|doorman|carrier|loader|receiver|stocker|stylist"
_ROLES = {k: _boundary(v) for k, v in ROLE_PATTERNS.items()}
_GENERIC = {k: _boundary(v) for k, v in GENERIC_PATTERNS.items()}
# Avoid matching an ordinary business 'target' as an employer brand.
INDUSTRY_PATTERNS["retail"] = INDUSTRY_PATTERNS["retail"].replace("|target", "")
_INDUSTRIES = {k: _boundary(v) for k, v in INDUSTRY_PATTERNS.items()}
_ORG = re.compile(_ORG.pattern + "|" + _boundary(r"us army|us navy|army|navy|air force|usaf|usmc|marine corps|national guard").pattern)
_BRANDS = re.compile(_BRANDS.pattern + "|" + _boundary(r"at and t|jp morgan chase|lockheed martin|walgreens|best buy").pattern)

def _resolve_context(text, hits):
    """Resolve compound expressions whose modifiers identify the intended function.

    Genuine combinations such as 'nurse and teacher' remain ambiguous.
    """
    hits = list(hits)
    def drop(label):
        if label in hits:
            hits.remove(label)
    if "healthcare" in hits and "social_community_services" in hits:
        if _boundary(r"nurse|nursing").search(text) and "case manager" in text and not _boundary(r"social work\w*").search(text):
            drop("social_community_services")
        elif _boundary(r"social work\w*").search(text) and not _boundary(r"nurse|physician|doctor").search(text):
            drop("healthcare")
    if "accounting_finance_insurance" in hits and "counselor" in text and not _boundary(r"social|school|mental|youth").search(text):
        drop("social_community_services")
    if "software_it_data" in hits and "manufacturing_production" in hits and "cnc programmer" in text and "software" not in text:
        drop("software_it_data")
    if "protective_services" in hits and "dispatcher" in text and _boundary(r"911|police|public safety|emergency").search(text) and not _boundary(r"truck|freight|warehouse").search(text):
        drop("transport_logistics")
    if "software_it_data" in hits and "information security officer" in text and not _boundary(r"police|guard|firefighter").search(text):
        drop("protective_services")
    return hits

def _resolve(hits):
    return hits[0] if len(hits) == 1 else "ambiguous" if hits else "unknown"

def _classify(text):
    missing = pd.isna(text)
    placeholder = not missing and (text in PLACEHOLDERS or text == "")
    if missing or placeholder:
        return dict(emp_role_family="unknown", emp_industry="unknown", emp_record_type="unknown",
                    emp_seniority="unknown", emp_employment_status="unknown", emp_management=0,
                    emp_ownership=0, emp_role_ambiguous=0, emp_industry_ambiguous=0,
                    emp_role_evidence="", emp_industry_evidence="", emp_rule_source="missing" if missing else "placeholder")
    hits = [k for k, p in _ROLES.items() if p.search(text)]
    if text == "it":
        hits = ["software_it_data"]
    candidates = list(hits)
    hits = _resolve_context(text, hits)
    # Contextual overlap: a school counselor is a counseling occupation in education.
    if "education" in hits and "social_community_services" in hits and "counselor" in text:
        hits.remove("education")
    source = ("context_resolved" if hits != candidates else "specific") if hits else "unmatched"
    if not hits:
        # A hierarchy only for non-specific function words: an office manager
        # retains the administrative function plus the separate management flag.
        for k, p in _GENERIC.items():
            if p.search(text):
                hits = [k]
                source = "generic"
                break
    industries = [k for k, p in _INDUSTRIES.items() if p.search(text)]
    org = bool(_LEGAL.search(text) or _ORG.search(text) or _BRANDS.search(text))
    role = bool(_ROLE_WORD.search(text))
    status = next((k for k,p in _STATUS.items() if p.fullmatch(text)), "unknown")
    if _boundary(r"self employed").search(text):
        status = "self_employed"
    kind = "mixed" if org and role else "organization" if org else "job_title" if role else "unknown"
    if status != "unknown" and not role and not org:
        kind = "employment_status"
    # Standalone organization names must not be mistaken for occupational roles.
    if kind == "organization":
        hits, source = [], "organization_only"
    seniority = "unspecified"
    if _boundary(r"chief (?:executive|financial|operating|information|technology|marketing) officer|president|executive director|managing director").search(text):
        seniority = "executive"
    elif _boundary(r"director").search(text):
        seniority = "director"
    elif _boundary(r"manager|supervisor|foreman|superintendent").search(text):
        seniority = "manager_supervisor"
    elif _boundary(r"senior|lead|leader|chief|principal (?:engineer|scientist|consultant|architect|analyst)").search(text):
        seniority = "senior_lead"
    elif _boundary(r"junior|intern|trainee|apprentice").search(text):
        seniority = "junior_trainee"
    if kind == "organization":
        seniority = "unknown"
    return dict(emp_role_family=_resolve(hits), emp_industry=_resolve(industries), emp_record_type=kind,
                emp_seniority=seniority, emp_employment_status=status,
                emp_management=int(bool(_MANAGEMENT.search(text)) and kind != "organization"),
                emp_ownership=int(bool(_OWNER.search(text)) and kind != "organization"),
                emp_role_ambiguous=int(len(hits)>1), emp_industry_ambiguous=int(len(industries)>1),
                emp_role_evidence="|".join(hits), emp_industry_evidence="|".join(industries), emp_rule_source=source)

MODEL_FEATURES = ["emp_role_family", "emp_industry", "emp_record_type", "emp_seniority",
                  "emp_employment_status", "emp_management", "emp_ownership",
                  "emp_is_placeholder", "emp_is_blank", "emp_role_ambiguous", "emp_industry_ambiguous"]

def _series(X, column):
    if isinstance(X, pd.DataFrame):
        return X[column]
    if isinstance(X, pd.Series):
        return X
    arr = np.asarray(X, dtype=object)
    if arr.ndim == 2 and arr.shape[1] == 1:
        arr = arr[:, 0]
    if arr.ndim != 1:
        raise ValueError("Expected a Series, a 1D array or a DataFrame with the configured column.")
    return pd.Series(arr, name=column)

def _key(value):
    return "missing:" if pd.isna(value) else "text:" + value

class EmpTitleTransformer(TransformerMixin, BaseEstimator):
    """Generate fixed rule features with optional train-only frequency features.

    Fit on each training fold. transform never learns or changes rules.
    diagnostics=True adds normalized text, evidence labels and rule source.
    Frequency measures normalized full titles, not individual token frequency.
    Unknown frequency at inference is zero; a separate unseen flag is returned.
    """
    def __init__(self, column="emp_title", diagnostics=False, include_frequency=False, aliases=None):
        self.column = column
        self.diagnostics = diagnostics
        self.include_frequency = include_frequency
        self.aliases = aliases

    def _normalize(self, s):
        # User edits are whole-normalized-title aliases and must be reviewed on train.
        aliases = self.aliases or {}
        result = s.map(normalize_emp_title)
        return result.map(lambda x: x if pd.isna(x) else aliases.get(x, x))

    def fit(self, X, y=None):
        s = _series(X, self.column)
        return self.fit_from_counts(s.value_counts(dropna=False))

    def fit_from_counts(self, counts: pd.Series):
        """Fit from TRAIN value_counts without expanding millions of rows.

        counts.index holds titles, counts.values holds positive integer counts.
        Nulls lost before export cannot be reconstructed.
        """
        if not isinstance(counts, pd.Series):
            raise TypeError("counts must be a pandas Series indexed by raw titles")
        values = pd.to_numeric(counts, errors="raise").to_numpy(dtype=float)
        if not len(values) or not np.isfinite(values).all() or (values <= 0).any() or (values != np.floor(values)).any():
            raise ValueError("Counts must be finite positive integers and cannot be empty.")
        if self.aliases is not None:
            if not isinstance(self.aliases, Mapping) or not all(isinstance(k,str) and isinstance(v,str) and _normalize_text(k)==k and _normalize_text(v)==v for k,v in self.aliases.items()):
                raise ValueError("aliases must map normalized strings to normalized strings")
        norm = self._normalize(pd.Series(counts.index))
        totals = Counter()
        for title, n in zip(norm, values):
            totals[_key(title)] += int(n)
        self.title_counts_ = dict(totals)
        self.n_train_ = sum(totals.values())
        self.rule_version_ = RULE_VERSION
        self.feature_names_in_ = np.asarray([self.column], dtype=object)
        self.n_features_in_ = 1
        self.is_fitted_ = True
        return self

    def transform(self, X):
        check_is_fitted(self, "is_fitted_")
        s = _series(X, self.column)
        norm = self._normalize(s)
        # Classify each normalized unique title once, then restore exact row order/index.
        keys = norm.map(_key)
        unique = dict(zip(keys, norm))
        lookup = {k: _classify(v) for k,v in unique.items()}
        out = pd.DataFrame([lookup[k] for k in keys], index=s.index,
                           columns=list(_classify(pd.NA)))
        out["emp_is_blank"] = norm.map(lambda v: not pd.isna(v) and v == "").to_numpy(dtype=np.int8)
        out["emp_is_placeholder"] = norm.map(lambda v: not pd.isna(v) and v in PLACEHOLDERS).to_numpy(dtype=np.int8)
        cols = list(MODEL_FEATURES)
        if self.include_frequency:
            n = keys.map(self.title_counts_).fillna(0).to_numpy(dtype=float)
            out["emp_title_log_count"] = np.log1p(n)
            out["emp_title_train_share"] = n / self.n_train_
            out["emp_title_unseen"] = (n == 0).astype(np.int8)
            cols += ["emp_title_log_count", "emp_title_train_share", "emp_title_unseen"]
        if self.diagnostics:
            out["emp_title_normalized"] = norm.to_numpy()
            cols += ["emp_title_normalized", "emp_role_evidence", "emp_industry_evidence", "emp_rule_source"]
        return out[cols]

    def get_feature_names_out(self, input_features=None):
        check_is_fitted(self, "is_fitted_")
        empty = self.transform(pd.Series([], dtype=object))
        return np.asarray(empty.columns, dtype=object)

def classify_emp_titles(emp_title, *, diagnostics=False):
    """Stateless fixed-rule convenience function (no learned frequencies)."""
    transformer = EmpTitleTransformer(diagnostics=diagnostics)
    # No data-dependent state is needed for the fixed rule convenience function.
    transformer.is_fitted_ = True
    return transformer.transform(emp_title)
