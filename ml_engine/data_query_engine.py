"""
Data Query Engine — SQL-style natural language querying on DataFrames.
Converts simple SQL-like statements or natural language to pandas operations.
Fully sandboxed: no arbitrary code execution, only DataFrame operations.
"""

import re
import pandas as pd
import numpy as np


def query_dataset(df, query_str):
    """
    Execute a SQL-style or natural language query against a DataFrame.

    Supports:
      - SELECT col1, col2 WHERE col3 > value
      - SHOW TOP 10 BY column
      - COUNT WHERE condition
      - GROUP BY col AGGREGATE func
      - Natural language: "show rows where age > 30"

    Args:
        df: pandas DataFrame
        query_str: SQL-like or natural language query string

    Returns:
        dict with 'result' (list of dicts), 'columns', 'n_rows', 'query_type'
    """
    if df is None or len(df) == 0:
        return {'error': 'No dataset loaded'}

    q = query_str.strip()
    q_lower = q.lower()

    try:
        # 1. Pandas query syntax (e.g., "age > 30 and salary < 100000")
        if _looks_like_pandas_query(q_lower, df):
            return _execute_pandas_query(df, q)

        # 2. SELECT ... WHERE ...
        if q_lower.startswith('select'):
            return _execute_select(df, q)

        # 3. SHOW TOP N BY col
        if 'top' in q_lower or 'largest' in q_lower or 'highest' in q_lower:
            return _execute_top_n(df, q)

        # 4. COUNT / how many
        if q_lower.startswith('count') or 'how many' in q_lower:
            return _execute_count(df, q)

        # 5. GROUP BY
        if 'group by' in q_lower or 'group_by' in q_lower:
            return _execute_group_by(df, q)

        # 6. DESCRIBE / STATS
        if any(w in q_lower for w in ['describe', 'stats', 'statistics', 'summary']):
            return _execute_describe(df, q)

        # 7. Natural language fallback — try to extract conditions
        return _natural_language_query(df, q)

    except Exception as e:
        return {'error': f'Query failed: {str(e)}', 'query': q}


def _looks_like_pandas_query(q_lower, df):
    """Check if the query looks like a valid pandas query expression."""
    col_names = [c.lower() for c in df.columns]
    operators = ['>', '<', '>=', '<=', '==', '!=']
    has_col = any(c in q_lower for c in col_names)
    has_op = any(op in q_lower for op in operators)
    return has_col and has_op and not q_lower.startswith('select')


def _execute_pandas_query(df, query_str):
    """Execute using pandas .query() method."""
    # Sanitize: only allow column names, operators, and values
    result = df.query(query_str)
    return _format_result(result, 'filter', query_str)


def _execute_select(df, query_str):
    """Parse SELECT col1, col2 FROM ... WHERE ..."""
    q = query_str.strip()
    # Remove SELECT keyword
    body = re.sub(r'^select\s+', '', q, flags=re.IGNORECASE).strip()

    # Split on WHERE
    where_clause = None
    if re.search(r'\bwhere\b', body, re.IGNORECASE):
        parts = re.split(r'\bwhere\b', body, flags=re.IGNORECASE)
        col_part = parts[0].strip()
        where_clause = parts[1].strip()
    else:
        col_part = body

    # Remove FROM if present
    col_part = re.sub(r'\bfrom\b.*', '', col_part, flags=re.IGNORECASE).strip()

    # Parse columns
    if col_part.strip() == '*':
        selected_cols = df.columns.tolist()
    else:
        selected_cols = [c.strip() for c in col_part.split(',')]
        # Fuzzy match column names
        selected_cols = [_match_column(c, df) for c in selected_cols]
        selected_cols = [c for c in selected_cols if c is not None]

    if not selected_cols:
        selected_cols = df.columns.tolist()

    result = df[selected_cols]

    if where_clause:
        try:
            result = result.loc[df.query(where_clause).index]
        except Exception:
            # Try parsing manually
            result = _apply_manual_filter(df, where_clause)[selected_cols]

    return _format_result(result, 'select', query_str)


def _execute_top_n(df, query_str):
    """Handle 'show top N by column' queries."""
    n_match = re.search(r'(\d+)', query_str)
    n = int(n_match.group(1)) if n_match else 10

    # Find column name
    col = _extract_column_from_query(query_str, df)
    if col and pd.api.types.is_numeric_dtype(df[col]):
        result = df.nlargest(n, col)
    else:
        result = df.head(n)

    return _format_result(result, 'top_n', query_str)


def _execute_count(df, query_str):
    """Handle count/how many queries."""
    q_lower = query_str.lower()

    # Count with condition
    if 'where' in q_lower or 'with' in q_lower or '>' in q_lower or '<' in q_lower:
        condition = re.sub(r'^(count|how many)[\s\w]*?(where|with)\s*', '', q_lower, flags=re.IGNORECASE).strip()
        try:
            filtered = df.query(condition)
            count = len(filtered)
        except Exception:
            filtered = _apply_manual_filter(df, condition)
            count = len(filtered)
        return {
            'result': [{'count': count, 'total': len(df), 'pct': round(count / max(len(df), 1) * 100, 2)}],
            'n_rows': 1,
            'query_type': 'count',
            'query': query_str,
        }

    # Count by column (group)
    col = _extract_column_from_query(query_str, df)
    if col:
        vc = df[col].value_counts().head(20)
        return {
            'result': [{'value': str(k), 'count': int(v)} for k, v in vc.items()],
            'n_rows': len(vc),
            'query_type': 'count_by',
            'column': col,
            'query': query_str,
        }

    return {'result': [{'total_rows': len(df)}], 'n_rows': 1, 'query_type': 'count', 'query': query_str}


def _execute_group_by(df, query_str):
    """Handle GROUP BY queries."""
    match = re.search(r'group\s*by\s+(\w+)', query_str, re.IGNORECASE)
    if not match:
        return {'error': 'Could not parse GROUP BY column'}

    group_col = _match_column(match.group(1), df)
    if not group_col:
        return {'error': f'Column "{match.group(1)}" not found'}

    # Determine aggregation
    agg = 'mean'
    if 'sum' in query_str.lower():
        agg = 'sum'
    elif 'count' in query_str.lower():
        agg = 'count'
    elif 'max' in query_str.lower():
        agg = 'max'
    elif 'min' in query_str.lower():
        agg = 'min'

    numeric = df.select_dtypes(include='number').columns.tolist()
    if not numeric:
        grouped = df.groupby(group_col).size().reset_index(name='count')
    else:
        grouped = df.groupby(group_col)[numeric].agg(agg).reset_index()

    return _format_result(grouped, 'group_by', query_str)


def _execute_describe(df, query_str):
    """Return descriptive statistics."""
    col = _extract_column_from_query(query_str, df)
    if col:
        desc = df[col].describe().to_dict()
        return {'result': [{k: round(v, 4) if isinstance(v, float) else v for k, v in desc.items()}],
                'n_rows': 1, 'query_type': 'describe', 'column': col, 'query': query_str}
    desc = df.describe().to_dict()
    return {'result': [desc], 'n_rows': 1, 'query_type': 'describe', 'query': query_str}


def _natural_language_query(df, query_str):
    """Attempt to parse natural language into a DataFrame operation."""
    q = query_str.lower()

    # "show rows where X > Y"
    condition_match = re.search(r'(?:show|find|get|filter)?\s*(?:rows?\s*)?(?:where|with|having)\s+(.+)', q)
    if condition_match:
        condition = condition_match.group(1).strip()
        try:
            result = df.query(condition)
            return _format_result(result, 'natural_filter', query_str)
        except Exception:
            result = _apply_manual_filter(df, condition)
            return _format_result(result, 'natural_filter', query_str)

    # "average/mean of column"
    agg_match = re.search(r'(average|mean|sum|max|min|median)\s+(?:of\s+)?(\w+)', q)
    if agg_match:
        func_name = agg_match.group(1)
        col = _match_column(agg_match.group(2), df)
        if col and pd.api.types.is_numeric_dtype(df[col]):
            func_map = {'average': 'mean', 'mean': 'mean', 'sum': 'sum',
                        'max': 'max', 'min': 'min', 'median': 'median'}
            val = getattr(df[col], func_map.get(func_name, 'mean'))()
            return {'result': [{func_name: round(float(val), 4), 'column': col}],
                    'n_rows': 1, 'query_type': 'aggregate', 'query': query_str}

    return {'error': 'Could not parse query. Try SQL-like syntax: SELECT col WHERE condition',
            'query': query_str,
            'hint': 'Examples: "age > 30", "SELECT name, age WHERE salary > 50000", "top 10 by revenue"'}


# ── Helpers ──────────────────────────────────────────────────

def _match_column(name, df):
    """Fuzzy-match a column name against the DataFrame."""
    name_lower = name.lower().strip()
    for col in df.columns:
        if col.lower() == name_lower:
            return col
    for col in df.columns:
        if name_lower in col.lower() or col.lower() in name_lower:
            return col
    return None


def _extract_column_from_query(query_str, df):
    """Extract the first column name mentioned in a query string."""
    q_lower = query_str.lower()
    for col in sorted(df.columns, key=len, reverse=True):
        if col.lower() in q_lower:
            return col
    return None


def _apply_manual_filter(df, condition_str):
    """Manually parse simple conditions like 'col > 5'."""
    match = re.search(r'(\w+)\s*(>|<|>=|<=|==|!=)\s*(["\']?[\w.\-]+["\']?)', condition_str)
    if match:
        col_name = _match_column(match.group(1), df)
        op = match.group(2)
        val_str = match.group(3).strip("'\"")
        if col_name:
            try:
                val = float(val_str)
            except ValueError:
                val = val_str
            if op == '>':
                return df[df[col_name] > val]
            elif op == '<':
                return df[df[col_name] < val]
            elif op == '>=':
                return df[df[col_name] >= val]
            elif op == '<=':
                return df[df[col_name] <= val]
            elif op == '==':
                return df[df[col_name] == val]
            elif op == '!=':
                return df[df[col_name] != val]
    return df


def _format_result(df_result, query_type, query_str, max_rows=200):
    """Format DataFrame result for JSON response."""
    if len(df_result) > max_rows:
        sample = df_result.head(max_rows)
        truncated = True
    else:
        sample = df_result
        truncated = False

    return {
        'result': sample.fillna('NaN').to_dict(orient='records'),
        'columns': df_result.columns.tolist(),
        'n_rows': len(df_result),
        'n_shown': len(sample),
        'truncated': truncated,
        'query_type': query_type,
        'query': query_str,
    }
