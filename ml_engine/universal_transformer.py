"""
Universal Data Transformer — Handle ANY dataset on Earth.
16+ sub-transformers that detect and fix every type of data quality issue.
"""

import re
import json
import base64
import numpy as np
import pandas as pd
from datetime import datetime


# ── Sub-Transformer Base ────────────────────────────────

class SubTransformer:
    name = "Base"
    icon = "🔧"
    
    def apply(self, df):
        return df, []


# ── 1. Column Name Cleaner ──────────────────────────────

class ColumnNameCleaner(SubTransformer):
    name = "Column Name Cleaner"
    icon = "🏷️"
    
    def apply(self, df):
        changes = []
        new_cols = {}
        for col in df.columns:
            clean = str(col).strip()
            clean = re.sub(r'[^\w\s]', '_', clean)
            clean = re.sub(r'\s+', '_', clean)
            clean = re.sub(r'_+', '_', clean)
            clean = clean.strip('_').lower()
            if not clean:
                clean = f'col_{df.columns.get_loc(col)}'
            if clean != str(col):
                new_cols[col] = clean
        
        if new_cols:
            # Handle duplicates
            seen = {}
            final_map = {}
            for old, new in new_cols.items():
                if new in seen:
                    seen[new] += 1
                    final_map[old] = f"{new}_{seen[new]}"
                else:
                    seen[new] = 0
                    final_map[old] = new
            
            df = df.rename(columns=final_map)
            changes.append({
                'name': self.name, 'icon': self.icon, 'applied': True,
                'description': f'Cleaned {len(final_map)} column names (removed special chars, spaces, normalized case)'
            })
        return df, changes


# ── 2. Constant Column Dropper ──────────────────────────

class ConstantDropper(SubTransformer):
    name = "Constant Column Dropper"
    icon = "🗑️"
    
    def apply(self, df):
        changes = []
        constant_cols = [col for col in df.columns if df[col].nunique(dropna=False) <= 1]
        if constant_cols:
            df = df.drop(columns=constant_cols)
            changes.append({
                'name': self.name, 'icon': self.icon, 'applied': True,
                'description': f'Dropped {len(constant_cols)} constant columns: {", ".join(constant_cols[:5])}'
            })
        return df, changes


# ── 3. High Missing Dropper ─────────────────────────────

class HighMissingDropper(SubTransformer):
    name = "High Missing Column Dropper"
    icon = "🕳️"
    threshold = 0.90
    
    def apply(self, df):
        changes = []
        missing_pct = df.isnull().mean()
        drop_cols = missing_pct[missing_pct > self.threshold].index.tolist()
        if drop_cols:
            df = df.drop(columns=drop_cols)
            changes.append({
                'name': self.name, 'icon': self.icon, 'applied': True,
                'description': f'Dropped {len(drop_cols)} columns with >{self.threshold*100:.0f}% missing: {", ".join(drop_cols[:5])}'
            })
        return df, changes


# ── 4. ID Column Detector ──────────────────────────────

class IDColumnDetector(SubTransformer):
    name = "ID/Fingerprint Column Detector"
    icon = "🔑"
    
    ID_PATTERNS = re.compile(r'(^id$|_id$|^uuid|^guid|^key$|^index$|^row_?num)', re.I)
    
    def apply(self, df):
        changes = []
        id_cols = []
        for col in df.columns:
            if self.ID_PATTERNS.search(str(col)):
                id_cols.append(col)
                continue
            if df[col].dtype == 'object':
                if df[col].nunique() == len(df) and len(df) > 20:
                    sample = df[col].dropna().head(20).astype(str)
                    if all(len(s) > 8 and not s.replace('.', '').replace(',', '').replace(' ', '').isalpha() for s in sample):
                        id_cols.append(col)
            elif pd.api.types.is_integer_dtype(df[col]):
                if df[col].nunique() == len(df) and df[col].is_monotonic_increasing:
                    id_cols.append(col)
        
        if id_cols:
            df = df.drop(columns=id_cols)
            changes.append({
                'name': self.name, 'icon': self.icon, 'applied': True,
                'description': f'Detected and dropped {len(id_cols)} ID columns: {", ".join(id_cols[:5])}'
            })
        return df, changes


# ── 5. Currency Parser ──────────────────────────────────

class CurrencyParser(SubTransformer):
    name = "Currency Parser"
    icon = "💰"
    
    CURRENCY_RE = re.compile(r'^[\s]*[$€£¥₹₩₽₿]?\s*-?\s*[\d,]+\.?\d*\s*$|^[\s]*-?\s*[\d.]+,\d{2}\s*[€£]?\s*$')
    
    def _parse_currency(self, val):
        if pd.isna(val):
            return np.nan
        s = str(val).strip()
        s = re.sub(r'[$€£¥₹₩₽₿\s]', '', s)
        if re.match(r'^-?[\d.]+,\d{2}$', s):
            s = s.replace('.', '').replace(',', '.')
        else:
            s = s.replace(',', '')
        try:
            return float(s)
        except ValueError:
            return np.nan
    
    def apply(self, df):
        changes = []
        converted = []
        for col in df.select_dtypes(include='object').columns:
            sample = df[col].dropna().head(50).astype(str)
            if len(sample) == 0:
                continue
            matches = sum(1 for s in sample if self.CURRENCY_RE.match(s))
            if matches / len(sample) > 0.7:
                df[col] = df[col].apply(self._parse_currency)
                converted.append(col)
        
        if converted:
            changes.append({
                'name': self.name, 'icon': self.icon, 'applied': True,
                'description': f'Parsed currency values in {len(converted)} columns: {", ".join(converted[:5])}'
            })
        return df, changes


# ── 6. Percentage Parser ────────────────────────────────

class PercentParser(SubTransformer):
    name = "Percentage Parser"
    icon = "📊"
    
    PCT_RE = re.compile(r'^[\s]*-?\d+\.?\d*\s*%\s*$')
    
    def apply(self, df):
        changes = []
        converted = []
        for col in df.select_dtypes(include='object').columns:
            sample = df[col].dropna().head(50).astype(str)
            if len(sample) == 0:
                continue
            matches = sum(1 for s in sample if self.PCT_RE.match(s))
            if matches / len(sample) > 0.7:
                df[col] = df[col].astype(str).str.strip().str.rstrip('%').astype(float) / 100.0
                converted.append(col)
        
        if converted:
            changes.append({
                'name': self.name, 'icon': self.icon, 'applied': True,
                'description': f'Converted percentage strings to decimal in {len(converted)} columns'
            })
        return df, changes


# ── 7. Boolean Unifier ──────────────────────────────────

class BooleanUnifier(SubTransformer):
    name = "Boolean Unifier"
    icon = "☑️"
    
    TRUE_VALS = {'yes', 'y', 'true', 't', '1', 'on', 'si', 'oui', 'ja', 'da'}
    FALSE_VALS = {'no', 'n', 'false', 'f', '0', 'off', 'non', 'nein', 'nyet'}
    
    def apply(self, df):
        changes = []
        converted = []
        for col in df.select_dtypes(include='object').columns:
            vals = df[col].dropna().astype(str).str.strip().str.lower().unique()
            if len(vals) <= 3:
                all_bool = all(v in self.TRUE_VALS | self.FALSE_VALS for v in vals if v)
                if all_bool and len(vals) >= 2:
                    df[col] = df[col].astype(str).str.strip().str.lower().map(
                        lambda x: 1 if x in self.TRUE_VALS else (0 if x in self.FALSE_VALS else np.nan)
                    )
                    converted.append(col)
        
        if converted:
            changes.append({
                'name': self.name, 'icon': self.icon, 'applied': True,
                'description': f'Unified boolean values (yes/no/true/false/Y/N → 0/1) in {len(converted)} columns'
            })
        return df, changes


# ── 8. Date Unifier ─────────────────────────────────────

class DateUnifier(SubTransformer):
    name = "Date Unifier"
    icon = "📅"
    
    def apply(self, df):
        changes = []
        converted = []
        for col in df.select_dtypes(include='object').columns:
            sample = df[col].dropna().head(30).astype(str)
            if len(sample) < 5:
                continue
            try:
                parsed = pd.to_datetime(sample, infer_datetime_format=True, dayfirst=False)
                success_rate = parsed.notna().sum() / len(sample)
                if success_rate > 0.8:
                    df[col] = pd.to_datetime(df[col], infer_datetime_format=True, errors='coerce')
                    # Extract useful features
                    col_clean = col.replace(' ', '_')
                    df[f'{col_clean}_year'] = df[col].dt.year
                    df[f'{col_clean}_month'] = df[col].dt.month
                    df[f'{col_clean}_dayofweek'] = df[col].dt.dayofweek
                    df = df.drop(columns=[col])
                    converted.append(col)
            except Exception:
                continue
        
        if converted:
            changes.append({
                'name': self.name, 'icon': self.icon, 'applied': True,
                'description': f'Parsed dates and extracted temporal features from {len(converted)} columns'
            })
        return df, changes


# ── 9. HTML/XML Stripper ────────────────────────────────

class HTMLStripper(SubTransformer):
    name = "HTML/XML Stripper"
    icon = "🌐"
    
    TAG_RE = re.compile(r'<[^>]+>')
    
    def apply(self, df):
        changes = []
        cleaned = []
        for col in df.select_dtypes(include='object').columns:
            sample = df[col].dropna().head(20).astype(str)
            has_html = sum(1 for s in sample if self.TAG_RE.search(s))
            if has_html / max(len(sample), 1) > 0.3:
                df[col] = df[col].astype(str).apply(lambda x: self.TAG_RE.sub(' ', x).strip())
                df[col] = df[col].str.replace(r'\s+', ' ', regex=True)
                cleaned.append(col)
        
        if cleaned:
            changes.append({
                'name': self.name, 'icon': self.icon, 'applied': True,
                'description': f'Stripped HTML/XML tags from {len(cleaned)} columns'
            })
        return df, changes


# ── 10. Nested JSON Flattener ───────────────────────────

class NestedFlattener(SubTransformer):
    name = "Nested JSON Flattener"
    icon = "📦"
    
    def apply(self, df):
        changes = []
        flattened = []
        for col in df.select_dtypes(include='object').columns:
            sample = df[col].dropna().head(10).astype(str)
            json_count = 0
            for s in sample:
                s = s.strip()
                if (s.startswith('{') and s.endswith('}')) or (s.startswith('[') and s.endswith(']')):
                    try:
                        json.loads(s)
                        json_count += 1
                    except (json.JSONDecodeError, ValueError):
                        pass
            
            if json_count / max(len(sample), 1) > 0.5:
                try:
                    parsed = df[col].apply(lambda x: json.loads(str(x)) if pd.notna(x) and str(x).strip().startswith(('{', '[')) else {})
                    if parsed.apply(lambda x: isinstance(x, dict)).all():
                        flat = pd.json_normalize(parsed)
                        flat.columns = [f'{col}_{c}' for c in flat.columns]
                        df = df.drop(columns=[col])
                        df = pd.concat([df, flat], axis=1)
                        flattened.append(col)
                except Exception:
                    continue
        
        if flattened:
            changes.append({
                'name': self.name, 'icon': self.icon, 'applied': True,
                'description': f'Flattened nested JSON in {len(flattened)} columns'
            })
        return df, changes


# ── 11. Sentinel Value Detector ─────────────────────────

class SentinelDetector(SubTransformer):
    name = "Sentinel Value Detector"
    icon = "🚩"
    
    SENTINEL_VALUES = {-1, -99, -999, -9999, 99, 999, 9999, 99999, 999999, 0}
    
    def apply(self, df):
        changes = []
        fixed = []
        for col in df.select_dtypes(include='number').columns:
            vals = df[col].dropna()
            if len(vals) < 10:
                continue
            for sentinel in self.SENTINEL_VALUES:
                count = (vals == sentinel).sum()
                pct = count / len(vals)
                if 0.001 < pct < 0.15:
                    q1, q3 = vals.quantile(0.25), vals.quantile(0.75)
                    iqr = q3 - q1
                    if sentinel < q1 - 3 * iqr or sentinel > q3 + 3 * iqr:
                        df.loc[df[col] == sentinel, col] = np.nan
                        fixed.append(f'{col}({sentinel})')
        
        if fixed:
            changes.append({
                'name': self.name, 'icon': self.icon, 'applied': True,
                'description': f'Replaced sentinel/placeholder values with NaN: {", ".join(fixed[:8])}'
            })
        return df, changes


# ── 12. Ordinal Detector ───────────────────────────────

class OrdinalDetector(SubTransformer):
    name = "Ordinal Detector"
    icon = "📶"
    
    ORDINAL_MAPS = {
        frozenset({'low', 'medium', 'high'}): {'low': 1, 'medium': 2, 'high': 3},
        frozenset({'small', 'medium', 'large'}): {'small': 1, 'medium': 2, 'large': 3},
        frozenset({'small', 'medium', 'large', 'xlarge'}): {'small': 1, 'medium': 2, 'large': 3, 'xlarge': 4},
        frozenset({'poor', 'fair', 'good', 'excellent'}): {'poor': 1, 'fair': 2, 'good': 3, 'excellent': 4},
        frozenset({'never', 'rarely', 'sometimes', 'often', 'always'}): {'never': 1, 'rarely': 2, 'sometimes': 3, 'often': 4, 'always': 5},
        frozenset({'very low', 'low', 'medium', 'high', 'very high'}): {'very low': 1, 'low': 2, 'medium': 3, 'high': 4, 'very high': 5},
        frozenset({'beginner', 'intermediate', 'advanced', 'expert'}): {'beginner': 1, 'intermediate': 2, 'advanced': 3, 'expert': 4},
    }
    
    def apply(self, df):
        changes = []
        converted = []
        for col in df.select_dtypes(include='object').columns:
            vals = frozenset(df[col].dropna().astype(str).str.strip().str.lower().unique())
            for pattern, mapping in self.ORDINAL_MAPS.items():
                if vals == pattern or vals.issubset(pattern):
                    df[col] = df[col].astype(str).str.strip().str.lower().map(mapping)
                    converted.append(col)
                    break
        
        if converted:
            changes.append({
                'name': self.name, 'icon': self.icon, 'applied': True,
                'description': f'Applied ordinal encoding to {len(converted)} columns: {", ".join(converted[:5])}'
            })
        return df, changes


# ── 13. Type Coercer ───────────────────────────────────

class TypeCoercer(SubTransformer):
    name = "Smart Type Coercer"
    icon = "🔄"
    
    def apply(self, df):
        changes = []
        coerced = []
        for col in df.select_dtypes(include='object').columns:
            sample = df[col].dropna().head(50).astype(str)
            if len(sample) == 0:
                continue
            numeric_count = sum(1 for s in sample if self._is_numeric(s))
            if numeric_count / len(sample) > 0.8:
                df[col] = pd.to_numeric(df[col], errors='coerce')
                coerced.append(col)
        
        if coerced:
            changes.append({
                'name': self.name, 'icon': self.icon, 'applied': True,
                'description': f'Coerced {len(coerced)} mixed-type columns to numeric'
            })
        return df, changes
    
    def _is_numeric(self, s):
        s = str(s).strip().replace(',', '')
        try:
            float(s)
            return True
        except ValueError:
            return False


# ── 14. Width Pruner ────────────────────────────────────

class WidthPruner(SubTransformer):
    name = "Wide Dataset Pruner"
    icon = "📐"
    MAX_COLS = 500
    
    def apply(self, df):
        changes = []
        if len(df.columns) > self.MAX_COLS:
            numeric_cols = df.select_dtypes(include='number').columns
            if len(numeric_cols) > self.MAX_COLS:
                variances = df[numeric_cols].var().sort_values(ascending=False)
                keep = variances.head(self.MAX_COLS).index.tolist()
                non_numeric = df.select_dtypes(exclude='number').columns.tolist()
                df = df[non_numeric + keep]
                changes.append({
                    'name': self.name, 'icon': self.icon, 'applied': True,
                    'description': f'Pruned to top {self.MAX_COLS} features by variance (from {len(numeric_cols)})'
                })
        return df, changes


# ── 15. Email/Phone/URL Detector ────────────────────────

class ContactInfoDetector(SubTransformer):
    name = "Contact Info Detector"
    icon = "📧"
    
    EMAIL_RE = re.compile(r'^[\w.+-]+@[\w-]+\.[\w.]+$')
    PHONE_RE = re.compile(r'^[\+]?[\d\s\-\(\)]{7,15}$')
    URL_RE = re.compile(r'^https?://|^www\.', re.I)
    
    def apply(self, df):
        changes = []
        detected = []
        for col in df.select_dtypes(include='object').columns:
            sample = df[col].dropna().head(30).astype(str)
            if len(sample) < 5:
                continue
            email_pct = sum(1 for s in sample if self.EMAIL_RE.match(s.strip())) / len(sample)
            phone_pct = sum(1 for s in sample if self.PHONE_RE.match(s.strip())) / len(sample)
            url_pct = sum(1 for s in sample if self.URL_RE.match(s.strip())) / len(sample)
            
            if email_pct > 0.5:
                df[f'{col}_domain'] = df[col].astype(str).str.extract(r'@([\w.]+)', expand=False)
                df = df.drop(columns=[col])
                detected.append(f'{col}(email→domain)')
            elif url_pct > 0.5:
                df = df.drop(columns=[col])
                detected.append(f'{col}(URL→dropped)')
            elif phone_pct > 0.5:
                df = df.drop(columns=[col])
                detected.append(f'{col}(phone→dropped)')
        
        if detected:
            changes.append({
                'name': self.name, 'icon': self.icon, 'applied': True,
                'description': f'Handled contact info columns: {", ".join(detected[:5])}'
            })
        return df, changes


# ── 16. Base64 Detector ─────────────────────────────────

class EncodedDataDetector(SubTransformer):
    name = "Encoded Data Detector"
    icon = "🔐"
    
    def apply(self, df):
        changes = []
        dropped = []
        for col in df.select_dtypes(include='object').columns:
            sample = df[col].dropna().head(20).astype(str)
            if len(sample) < 5:
                continue
            b64_count = 0
            for s in sample:
                s = s.strip()
                if len(s) > 20 and re.match(r'^[A-Za-z0-9+/=]+$', s):
                    try:
                        decoded = base64.b64decode(s)
                        if len(decoded) > 0:
                            b64_count += 1
                    except Exception:
                        pass
            if b64_count / len(sample) > 0.5:
                df = df.drop(columns=[col])
                dropped.append(col)
        
        if dropped:
            changes.append({
                'name': self.name, 'icon': self.icon, 'applied': True,
                'description': f'Detected and dropped {len(dropped)} base64-encoded columns'
            })
        return df, changes


# ── Main Universal Transformer ──────────────────────────

class UniversalDataTransformer:
    """Applies all sub-transformers in sequence to handle any dataset."""
    
    def __init__(self):
        self.transformers = [
            ColumnNameCleaner(),
            NestedFlattener(),
            HTMLStripper(),
            EncodedDataDetector(),
            ContactInfoDetector(),
            CurrencyParser(),
            PercentParser(),
            BooleanUnifier(),
            TypeCoercer(),
            DateUnifier(),
            IDColumnDetector(),
            ConstantDropper(),
            HighMissingDropper(),
            SentinelDetector(),
            OrdinalDetector(),
            WidthPruner(),
        ]
    
    def transform(self, df):
        """Apply all transformers and return cleaned df + report."""
        all_changes = []
        original_shape = df.shape
        
        for transformer in self.transformers:
            try:
                df, changes = transformer.apply(df)
                all_changes.extend(changes)
            except Exception as e:
                all_changes.append({
                    'name': transformer.name,
                    'icon': '⚠️',
                    'applied': False,
                    'description': f'Skipped due to error: {str(e)[:100]}'
                })
        
        summary = {
            'original_shape': list(original_shape),
            'final_shape': list(df.shape),
            'columns_removed': original_shape[1] - df.shape[1],
            'transformations_applied': sum(1 for c in all_changes if c.get('applied')),
            'total_checks': len(self.transformers),
        }
        
        return df, {
            'steps': all_changes,
            'summary': summary
        }
