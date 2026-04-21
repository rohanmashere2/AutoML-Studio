"""
AutoML Studio — Automated PII Detection and Masking
Scans columns for personally identifiable information using regex patterns
and column-name heuristics. Offers hash, tokenise, and redact masking.
"""

import re
import hashlib
import uuid
import numpy as np
import pandas as pd


# ── PII Pattern Definitions ──────────────────────────────────

PII_PATTERNS = {
    'email': {
        'regex': r'[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}',
        'col_hints': ['email', 'e_mail', 'mail', 'email_address'],
        'risk': 'high',
        'description': 'Email address',
    },
    'phone': {
        'regex': r'(\+?\d{1,3}[\-.\s]?)?\(?\d{3}\)?[\-.\s]?\d{3}[\-.\s]?\d{4}',
        'col_hints': ['phone', 'telephone', 'mobile', 'cell', 'tel', 'phone_number'],
        'risk': 'high',
        'description': 'Phone number',
    },
    'ssn': {
        'regex': r'\b\d{3}-\d{2}-\d{4}\b',
        'col_hints': ['ssn', 'social_security', 'social_security_number'],
        'risk': 'critical',
        'description': 'Social Security Number',
    },
    'credit_card': {
        'regex': r'\b\d{4}[\-\s]?\d{4}[\-\s]?\d{4}[\-\s]?\d{4}\b',
        'col_hints': ['credit_card', 'card_number', 'cc_number', 'card_no'],
        'risk': 'critical',
        'description': 'Credit card number',
    },
    'ip_address': {
        'regex': r'\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b',
        'col_hints': ['ip', 'ip_address', 'ipv4', 'client_ip', 'server_ip'],
        'risk': 'medium',
        'description': 'IP address',
    },
    'aadhaar': {
        'regex': r'\b\d{4}\s?\d{4}\s?\d{4}\b',
        'col_hints': ['aadhaar', 'aadhar', 'uid'],
        'risk': 'critical',
        'description': 'Aadhaar number (India)',
    },
    'passport': {
        'regex': r'\b[A-Z]{1,2}\d{6,8}\b',
        'col_hints': ['passport', 'passport_number', 'passport_no'],
        'risk': 'critical',
        'description': 'Passport number',
    },
    'date_of_birth': {
        'regex': r'\b\d{1,2}[/\-]\d{1,2}[/\-]\d{2,4}\b',
        'col_hints': ['dob', 'date_of_birth', 'birth_date', 'birthdate', 'birthday'],
        'risk': 'medium',
        'description': 'Date of birth',
    },
    'name': {
        'regex': None,  # Name detection via column hints only
        'col_hints': [
            'name', 'first_name', 'last_name', 'full_name', 'firstname',
            'lastname', 'surname', 'given_name', 'family_name', 'patient_name',
            'customer_name', 'employee_name', 'user_name', 'username',
        ],
        'risk': 'high',
        'description': 'Personal name',
    },
    'address': {
        'regex': None,  # Address detection via column hints
        'col_hints': [
            'address', 'street', 'street_address', 'home_address',
            'mailing_address', 'addr', 'postal_address',
        ],
        'risk': 'high',
        'description': 'Physical address',
    },
    'zipcode': {
        'regex': r'\b\d{5}(-\d{4})?\b',
        'col_hints': ['zip', 'zipcode', 'zip_code', 'postal_code', 'pincode', 'pin_code'],
        'risk': 'low',
        'description': 'Postal/ZIP code',
    },
}


# ── Luhn Check for Credit Cards ──────────────────────────────

def _luhn_check(number_str):
    """Validate credit card number using Luhn algorithm."""
    digits = [int(d) for d in number_str if d.isdigit()]
    if len(digits) < 13 or len(digits) > 19:
        return False
    checksum = 0
    reverse_digits = digits[::-1]
    for i, d in enumerate(reverse_digits):
        if i % 2 == 1:
            d *= 2
            if d > 9:
                d -= 9
        checksum += d
    return checksum % 10 == 0


# ── Core Scanner ─────────────────────────────────────────────

def scan_for_pii(df, sample_size=1000):
    """
    Scan every column in the DataFrame for PII patterns.

    Args:
        df: pandas DataFrame
        sample_size: max rows to sample per column (for performance)

    Returns:
        dict with pii_columns list, summary, and risk assessment
    """
    pii_columns = []
    n_rows = len(df)

    for col in df.columns:
        if df[col].isnull().all():
            continue

        col_lower = col.lower().strip().replace(' ', '_')

        # Sample for regex testing
        sample = df[col].dropna()
        if len(sample) > sample_size:
            sample = sample.sample(sample_size, random_state=42)

        best_match = None
        best_confidence = 0.0
        best_pii_type = None
        match_count = 0

        for pii_type, config in PII_PATTERNS.items():
            confidence = 0.0
            type_match_count = 0

            # ── Column name heuristic
            name_match = any(hint == col_lower or hint in col_lower
                             for hint in config['col_hints'])
            if name_match:
                confidence += 0.5

            # ── Regex pattern matching
            if config['regex'] and df[col].dtype == 'object':
                try:
                    str_sample = sample.astype(str)
                    matches = str_sample.apply(
                        lambda x: bool(re.search(config['regex'], str(x)))
                    )
                    type_match_count = int(matches.sum())
                    match_pct = type_match_count / max(len(str_sample), 1)

                    if match_pct > 0.3:
                        confidence += 0.4
                    elif match_pct > 0.1:
                        confidence += 0.2
                    elif match_pct > 0.01:
                        confidence += 0.1

                    # Extra validation for credit cards (Luhn)
                    if pii_type == 'credit_card' and type_match_count > 0:
                        luhn_valid = str_sample.apply(
                            lambda x: _luhn_check(re.sub(r'[\s\-]', '', str(x)))
                        ).sum()
                        if luhn_valid < type_match_count * 0.5:
                            confidence *= 0.3  # Many false positives

                except Exception:
                    pass

            # ── Name detection heuristic (unique strings, title case)
            if pii_type == 'name' and name_match and df[col].dtype == 'object':
                try:
                    str_sample = sample.astype(str)
                    title_pct = str_sample.apply(
                        lambda x: x.istitle() or x.isupper()
                    ).mean()
                    unique_ratio = df[col].nunique() / max(n_rows, 1)
                    if title_pct > 0.5 and unique_ratio > 0.3:
                        confidence += 0.3
                        type_match_count = int(title_pct * len(str_sample))
                except Exception:
                    pass

            if confidence > best_confidence:
                best_confidence = confidence
                best_pii_type = pii_type
                match_count = type_match_count

        # Threshold: only flag if confidence >= 0.4
        if best_confidence >= 0.4 and best_pii_type:
            config = PII_PATTERNS[best_pii_type]

            # Redacted sample matches
            sample_matches = []
            if match_count > 0 and config['regex'] and df[col].dtype == 'object':
                try:
                    matches = sample.astype(str)[
                        sample.astype(str).apply(
                            lambda x: bool(re.search(config['regex'], str(x)))
                        )
                    ].head(3)
                    sample_matches = [_redact_value(str(v), best_pii_type)
                                      for v in matches.values]
                except Exception:
                    pass

            risk_level = config['risk']
            match_pct = round(match_count / max(len(sample), 1) * 100, 1)

            pii_columns.append({
                'column': col,
                'pii_type': best_pii_type,
                'pii_description': config['description'],
                'confidence': round(best_confidence, 2),
                'match_count': match_count,
                'match_pct': match_pct,
                'sample_matches': sample_matches,
                'risk_level': risk_level,
                'recommendation': _get_recommendation(best_pii_type, risk_level),
                'icon': _get_icon(risk_level),
            })

    # Risk summary
    critical = sum(1 for p in pii_columns if p['risk_level'] == 'critical')
    high = sum(1 for p in pii_columns if p['risk_level'] == 'high')
    medium = sum(1 for p in pii_columns if p['risk_level'] == 'medium')

    if critical > 0:
        risk_summary = f'🔴 CRITICAL: {critical} columns contain highly sensitive PII (SSN, credit card). Immediate masking recommended.'
    elif high > 0:
        risk_summary = f'🟠 HIGH: {high} columns contain PII (names, emails, phones). Mask before training.'
    elif medium > 0:
        risk_summary = f'🟡 MEDIUM: {medium} columns contain potentially identifying info. Review before training.'
    else:
        risk_summary = '🟢 No PII detected. Dataset appears safe for training.'

    return {
        'pii_columns': pii_columns,
        'total_pii_columns': len(pii_columns),
        'risk_summary': risk_summary,
        'risk_counts': {'critical': critical, 'high': high, 'medium': medium},
    }


# ── Masking Functions ────────────────────────────────────────

def mask_pii(df, columns, method='hash'):
    """
    Pseudonymise PII columns.

    Args:
        df: DataFrame
        columns: list of column names to mask
        method: 'hash' | 'tokenise' | 'redact'

    Returns:
        (masked_df, masking_report)
    """
    df = df.copy()
    report = {'method': method, 'columns_masked': [], 'details': {}}

    for col in columns:
        if col not in df.columns:
            continue

        original_unique = df[col].nunique()

        if method == 'hash':
            # Deterministic SHA256 hash (keeps joins possible)
            salt = 'automl_pii_salt_v1'
            df[col] = df[col].apply(
                lambda x: hashlib.sha256(
                    f'{salt}:{x}'.encode()
                ).hexdigest()[:16] if pd.notna(x) else x
            )

        elif method == 'tokenise':
            # Random UUID per unique value
            unique_vals = df[col].dropna().unique()
            token_map = {v: str(uuid.uuid4())[:8] for v in unique_vals}
            df[col] = df[col].map(token_map)

        elif method == 'redact':
            df[col] = df[col].apply(
                lambda x: '[REDACTED]' if pd.notna(x) else x
            )

        report['columns_masked'].append(col)
        report['details'][col] = {
            'original_unique': int(original_unique),
            'masked_unique': int(df[col].nunique()),
            'method': method,
        }

    return df, report


def generate_pii_report(scan_result):
    """Generate a human-readable PII report summary."""
    lines = [
        '# PII Detection Report',
        '',
        f'**{scan_result["total_pii_columns"]}** columns with potential PII detected.',
        '',
        scan_result['risk_summary'],
        '',
    ]

    if scan_result['pii_columns']:
        lines.append('## Detected PII Columns')
        lines.append('')
        lines.append('| Column | Type | Risk | Confidence | Matches |')
        lines.append('|--------|------|------|------------|---------|')
        for p in scan_result['pii_columns']:
            lines.append(
                f"| {p['column']} | {p['pii_description']} | "
                f"{p['icon']} {p['risk_level']} | {p['confidence']:.0%} | "
                f"{p['match_pct']}% |"
            )

    return '\n'.join(lines)


# ── Internal Helpers ─────────────────────────────────────────

def _redact_value(value, pii_type):
    """Partially redact a value for display."""
    if not value or len(value) < 3:
        return '***'

    if pii_type == 'email':
        parts = value.split('@')
        if len(parts) == 2:
            return f'{parts[0][:2]}***@{parts[1]}'
    elif pii_type == 'phone':
        return f'***{value[-4:]}'
    elif pii_type in ('ssn', 'credit_card', 'aadhaar'):
        return f'***{value[-4:]}'
    elif pii_type == 'name':
        return f'{value[0]}***'

    return f'{value[:2]}***'


def _get_recommendation(pii_type, risk_level):
    """Get masking recommendation for a PII type."""
    if risk_level == 'critical':
        return f'MUST mask {pii_type} before training. Use hash or redact method.'
    elif risk_level == 'high':
        return f'Strongly recommend masking {pii_type}. Use hash to preserve join capability.'
    elif risk_level == 'medium':
        return f'Consider masking {pii_type} if not needed for modelling.'
    return f'Low risk — {pii_type} may be acceptable depending on use case.'


def _get_icon(risk_level):
    """Get risk icon."""
    return {
        'critical': '🔴',
        'high': '🟠',
        'medium': '🟡',
        'low': '🟢',
    }.get(risk_level, '⚪')
