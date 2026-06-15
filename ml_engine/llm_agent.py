"""
LLM Chat Agent — Gemini-powered natural language interface for data science.
Uses Google Gemini API (gemini-2.5-flash) for intelligent responses with full session context.
Falls back to local flan-t5-small, then rule-based intent detection.
Features: conversation memory per session, system prompt with leaderboard/features/drift.
"""

import os
import re
import json
import time
import numpy as np
import pandas as pd
from dotenv import load_dotenv

load_dotenv()


# ── Conversation Memory ──────────────────────────────────────

class ConversationMemory:
    """Sliding-window conversation history per session."""

    def __init__(self, max_messages=20):
        self.messages = []
        self.max_messages = max_messages

    def add_user_message(self, content):
        self.messages.append({'role': 'user', 'content': content, 'ts': time.time()})
        self._trim()

    def add_assistant_message(self, content):
        self.messages.append({'role': 'assistant', 'content': content, 'ts': time.time()})
        self._trim()

    def get_history(self):
        """Return messages formatted for the API (without timestamps)."""
        return [{'role': m['role'], 'content': m['content']} for m in self.messages]

    def _trim(self):
        if len(self.messages) > self.max_messages:
            self.messages = self.messages[-self.max_messages:]


# ── Main Agent ────────────────────────────────────────────────

class AutoMLChatAgent:
    """Chat agent that uses Google Gemini for intelligent, context-aware responses."""

    # Class-level shared local pipeline (persists across instances to save RAM)
    _shared_pipeline = None

    # Pipeline actions the LLM can trigger (class-level constant)
    PIPELINE_TOOLS = {
        'run_eda': {
            'description': 'Run automated exploratory data analysis on the dataset',
            'requires': ['session_id'],
        },
        'start_training': {
            'description': 'Train ML models on the dataset. Optional param: time_budget (seconds)',
            'requires': ['session_id'],
            'optional': ['time_budget'],
        },
        'explain_model': {
            'description': 'Generate SHAP explanations and feature importance for the best model',
            'requires': ['session_id'],
        },
        'check_drift': {
            'description': 'Check for data drift between training and new data',
            'requires': ['session_id'],
        },
        'set_target': {
            'description': 'Set or change the target column for prediction',
            'requires': ['session_id', 'column_name'],
        },
        'run_diagnostics': {
            'description': 'Run model diagnostics (overfitting, residuals, learning curves)',
            'requires': ['session_id'],
        },
        'detect_label_errors': {
            'description': 'Use confident learning to detect potentially mislabeled training samples',
            'requires': ['session_id'],
        },
        'optimize_hyperparams': {
            'description': 'Run advanced hyperparameter optimization on trained models',
            'requires': ['session_id'],
            'optional': ['method', 'n_trials'],
        },
    }

    def __init__(self, pipeline_manager=None):
        self.pm = pipeline_manager
        self.conversations = {}  # session_id -> ConversationMemory
        self.llm_provider = None  # 'gemini' | 'openai' | 'local' | 'rules'
        self.openai_client = None
        self.gemini_model = None
        self.llm_available = False
        self._init_llm_provider()

        # Rule-based fallback patterns
        self.intents = {
            'explore': {
                'patterns': [r'show.*correlation', r'distribution', r'describe', r'summary', r'statistics',
                             r'how many', r'count', r'mean|average|median', r'unique', r'missing', r'null'],
                'handler': self._handle_explore
            },
            'visualize': {
                'patterns': [r'plot', r'chart', r'graph', r'histogram', r'scatter', r'heatmap', r'visualize'],
                'handler': self._handle_visualize
            },
            'explain': {
                'patterns': [r'why.*predict', r'explain', r'shap', r'feature importance', r'what.*important',
                             r'counterfactual', r'what.*change', r'what.*flip'],
                'handler': self._handle_explain
            },
            'predict': {
                'patterns': [r'predict', r'forecast', r'what.*will|would', r'estimate'],
                'handler': self._handle_predict
            },
            'compare': {
                'patterns': [r'compare', r'difference', r'vs\.?', r'versus', r'better.*model'],
                'handler': self._handle_compare
            },
            'recommend': {
                'patterns': [r'recommend', r'suggest', r'improve', r'optimize', r'how.*better'],
                'handler': self._handle_recommend
            },
            'clean': {
                'patterns': [r'clean', r'remove.*outlier', r'handle.*missing', r'drop.*column', r'exclude'],
                'handler': self._handle_clean
            },
            'drift': {
                'patterns': [r'drift', r'monitor', r'retrain', r'degrad', r'performance.*drop'],
                'handler': self._handle_drift
            },
            'greeting': {
                'patterns': [r'^hi$', r'^hello$', r'^hey$', r'^yo$', r'good morning', r'good afternoon',
                             r'how are you', r'greetings'],
                'handler': self._handle_greeting
            },
            'help': {
                'patterns': [r'^help$', r'what.*can.*do', r'commands', r'how.*use'],
                'handler': self._handle_help
            },
        }

    # ── LLM Initialization ────────────────────────────────────

    def _init_llm_provider(self):
        """Initialize the best available LLM provider."""
        # Try Gemini first
        gemini_key = os.getenv('GEMINI_API_KEY')
        if gemini_key:
            self._init_gemini(gemini_key)
            if self.llm_provider == 'gemini':
                return

        # Try OpenAI next
        openai_key = os.getenv('OPENAI_API_KEY')
        if openai_key and openai_key.startswith('sk-'):
            self._init_openai(openai_key)
            if self.llm_provider == 'openai':
                return

        # Fall back to local LLM
        self._init_local_llm()

    def _init_gemini(self, api_key):
        """Initialize Gemini model without making a test API call (saves credits and startup time)."""
        try:
            import google.generativeai as genai
            genai.configure(api_key=api_key)
            self.gemini_model = genai.GenerativeModel('gemini-2.5-flash')
            self.llm_provider = 'gemini'
            self.llm_available = True
            print("[OK] LLM Agent: Gemini 2.5 Flash ready")
        except ImportError:
            print("[ERROR] google-generativeai not installed. Run: pip install google-generativeai")
        except Exception as e:
            print(f"[ERROR] Gemini init failed: {type(e).__name__}: {e}")

    def _init_openai(self, api_key):
        """Initialize OpenAI client."""
        try:
            import openai
            self.openai_client = openai.OpenAI(api_key=api_key)
            self.openai_client.models.list()  # validate key
            self.llm_provider = 'openai'
            self.llm_available = True
            print("[OK] LLM Agent: OpenAI GPT-4o-mini ready")
        except Exception as e:
            print(f"[WARN] OpenAI init failed: {e}")
            self.llm_provider = None
            self.llm_available = False

    def _init_local_llm(self):
        """Initialize Local LLM using huggingface transformers."""
        if AutoMLChatAgent._shared_pipeline is not None:
            self.llm_provider = 'local'
            self.llm_available = True
            return

        try:
            from transformers import pipeline
            AutoMLChatAgent._shared_pipeline = pipeline("text2text-generation", model="google/flan-t5-small")
            self.llm_provider = 'local'
            self.llm_available = True
            print("[OK] LLM Agent: Local flan-t5-small ready")
        except Exception as e:
            print(f"[WARN] Local LLM init warning: {e}")
            self.llm_provider = 'rules'
            self.llm_available = False
            print("[INFO] LLM Agent: Rule-based fallback active")

    @classmethod
    def set_api_key(cls, api_key):
        """Update the Gemini or OpenAI API key at runtime."""
        if api_key and api_key.startswith('sk-'):
            os.environ['OPENAI_API_KEY'] = api_key
            return {'success': True, 'message': 'OpenAI API key updated. Restart chat to activate.'}
        elif api_key:
            os.environ['GEMINI_API_KEY'] = api_key
            return {'success': True, 'message': 'Gemini API key updated. Restart chat to activate.'}
        return {'success': False, 'message': 'Invalid API key.'}

    # ── Main Chat Entry ───────────────────────────────────────

    def chat(self, message, session_data=None, session_id=None):
        """Process a chat message with full context and memory."""
        # Get or create conversation memory for this session
        if session_id:
            memory = self.conversations.setdefault(session_id, ConversationMemory())
        else:
            memory = ConversationMemory()

        memory.add_user_message(message)

        # Route to best available provider
        if self.llm_provider == 'gemini':
            response = self._gemini_chat(message, session_data, memory)
        elif self.llm_provider == 'openai':
            response = self._openai_chat(message, session_data, memory)
        elif self.llm_provider == 'local' and self.llm_available:
            response = self._local_llm_chat(message, session_data)
        else:
            intent = self._classify_intent(message)
            handler = self.intents.get(intent, {}).get('handler', self._handle_unknown)
            response = handler(message, session_data)

        memory.add_assistant_message(response['text'])

        result = {
            'intent': response.get('intent', self.llm_provider),
            'response': response['text'],
            'data': response.get('data'),
            'chart': response.get('chart'),
            'powered_by': self.llm_provider,
        }

        # Check if the LLM suggested a pipeline action
        suggested = response.get('suggested_action') or self._parse_suggested_action(response['text'])
        if suggested:
            result['suggested_action'] = suggested

        return result

    # ── Gemini Chat ──────────────────────────────────────────

    def _gemini_chat(self, message, session_data, memory):
        """Use Google Gemini 1.5 Flash for intelligent, context-aware responses."""
        try:
            system_prompt = self._build_system_prompt(session_data)

            # Build conversation history in Gemini format
            history = []
            for m in memory.get_history()[:-1][-16:]:
                role = 'user' if m['role'] == 'user' else 'model'
                history.append({'role': role, 'parts': [m['content']]})

            chat = self.gemini_model.start_chat(history=history)

            # Prepend system prompt to the first/current user message
            full_message = f"{system_prompt}\n\nUser: {message}"
            response = chat.send_message(full_message)

            return {
                'text': response.text,
                'intent': 'gemini',
            }

        except Exception as e:
            error_msg = str(e)
            # Fallback to rule-based on error
            intent = self._classify_intent(message)
            handler = self.intents.get(intent, {}).get('handler', self._handle_unknown)
            result = handler(message, session_data)
            result['text'] = f"⚡ *Gemini API call failed ({error_msg[:80]}) — using smart fallback:*\n\n" + result['text']
            return result

    # ── OpenAI GPT Chat ──────────────────────────────────────

    def _openai_chat(self, message, session_data, memory):
        """Use OpenAI GPT-4o-mini for intelligent, context-aware responses."""
        try:
            system_prompt = self._build_system_prompt(session_data)
            messages = [{'role': 'system', 'content': system_prompt}]

            # Add conversation history (last N messages)
            history = memory.get_history()
            if len(history) > 1:
                messages.extend(history[:-1][-16:])

            # Current message
            messages.append({'role': 'user', 'content': message})

            response = self.openai_client.chat.completions.create(
                model='gpt-4o-mini',
                messages=messages,
                max_tokens=800,
                temperature=0.7,
            )

            reply_text = response.choices[0].message.content

            return {
                'text': reply_text,
                'intent': 'openai_gpt',
            }

        except Exception as e:
            error_msg = str(e)
            intent = self._classify_intent(message)
            handler = self.intents.get(intent, {}).get('handler', self._handle_unknown)
            result = handler(message, session_data)
            result['text'] = f"⚡ *API call failed ({error_msg[:80]}) — using smart fallback:*\n\n" + result['text']
            return result

    # ── Local LLM Chat ────────────────────────────────────────

    def _local_llm_chat(self, message, session_data):
        """Use Local HuggingFace LLM to generate response."""
        try:
            df = session_data.get('df') if session_data else None
            context = self._build_data_context(df) if df is not None else "No dataset loaded yet."

            # Simple context truncation (Flan-T5 limit is 512 tokens)
            if len(context) > 1000:
                context = context[:1000] + "..."

            prompt = f"Data context: {context}\n\nUser Question: {message}\n\nAnswer:"

            generator = AutoMLChatAgent._shared_pipeline
            result = generator(prompt, max_length=150, num_return_sequences=1, do_sample=True, temperature=0.7)
            reply_text = result[0]['generated_text']

            return {
                'text': reply_text,
                'intent': 'local_llm',
            }

        except Exception as e:
            error_msg = str(e)
            intent = self._classify_intent(message)
            handler = self.intents.get(intent, {}).get('handler', self._handle_unknown)
            result = handler(message, session_data)
            result['text'] = f"⚡ *Local LLM error ({error_msg}) — using smart fallback:*\n\n" + result['text']
            return result

    # ── System Prompt Builder ─────────────────────────────────

    def _build_system_prompt(self, session_data):
        """Build a rich system prompt with full session context."""
        base = (
            "You are the AutoML Studio AI Assistant — an expert data scientist embedded in an automated "
            "machine learning platform. You help users understand their data, interpret model results, "
            "and make decisions about their ML pipeline.\n\n"
            "RULES:\n"
            "- Be concise but thorough. Use markdown formatting.\n"
            "- Reference specific column names, model scores, and feature importances when available.\n"
            "- If the user asks about something you don't have data for, tell them which "
            "pipeline step to run (Upload, Clean, Train, Tune, Monitor).\n"
            "- Use emojis sparingly for section headers.\n"
            "- When suggesting actions, be specific: name the exact feature, threshold, or setting.\n"
            "- You can reference previous messages in the conversation.\n\n"
            "PIPELINE ACTIONS:\n"
            "You can suggest pipeline actions by including a JSON block in your response like:\n"
            '```action\n{"action": "start_training", "params": {"time_budget": 300}}\n```\n'
            "Available actions: " + ", ".join(self.PIPELINE_TOOLS.keys()) + "\n"
            "Only suggest an action when the user explicitly asks to DO something (train, analyze, explain).\n"
            "Do NOT suggest actions when they're just asking questions.\n\n"
        )

        if not session_data:
            return base + "STATUS: No session data available. Guide the user to upload a dataset first."

        context_parts = []

        # Dataset profile
        profile = session_data.get('profile')
        if profile:
            context_parts.append(
                f"📊 DATASET PROFILE:\n"
                f"- Shape: {profile.get('n_rows', '?'):,} rows × {profile.get('n_cols', '?')} columns\n"
                f"- Target: '{profile.get('target_column', 'N/A')}' ({profile.get('problem_type', 'unknown')})\n"
                f"- Missing: {profile.get('total_missing_pct', 0):.1f}%\n"
                f"- Duplicates: {profile.get('duplicates', 0)}\n"
            )

            # Column summary
            columns = profile.get('columns', [])
            if columns:
                col_lines = []
                for c in columns[:25]:
                    if isinstance(c, dict):
                        col_lines.append(f"  • {c.get('name', '?')} ({c.get('dtype', '?')}): {c.get('nunique', '?')} unique, {c.get('missing_pct', 0):.1f}% missing")
                    elif isinstance(c, str):
                        col_lines.append(f"  • {c}")
                if col_lines:
                    context_parts.append("COLUMNS:\n" + "\n".join(col_lines))

        # DataFrame quick stats
        df = session_data.get('df')
        if df is not None:
            context_parts.append(self._build_data_context(df))

        # Model leaderboard
        training = session_data.get('training_results')
        if training:
            lb = training.get('leaderboard', [])
            if lb:
                lb_lines = [f"  {e.get('rank', '?')}. {e.get('model', '?')} — {e.get('primary_metric', 0):.4f}" for e in lb[:7]]
                context_parts.append(
                    f"🏆 MODEL LEADERBOARD ({training.get('primary_metric_name', 'score')}):\n"
                    + "\n".join(lb_lines)
                    + f"\n  Best: {training.get('best_model', '?')} ({training.get('best_score', 0):.4f})"
                )

            # Feature importance
            fi = training.get('feature_importance', [])
            if fi:
                fi_lines = [f"  • {f['feature']}: {f['importance']*100:.1f}%" for f in fi[:10]]
                context_parts.append("🎯 TOP FEATURE IMPORTANCE:\n" + "\n".join(fi_lines))

        # Recommendations
        recs = session_data.get('recommendations')
        if recs and isinstance(recs, list):
            rec_lines = [f"  • [{r.get('impact', '?')}] {r.get('title', '')}" for r in recs[:5]]
            context_parts.append("💡 ACTIVE RECOMMENDATIONS:\n" + "\n".join(rec_lines))

        # Drift status
        drift = session_data.get('drift_report')
        if drift:
            context_parts.append(
                f"📡 DRIFT STATUS: {drift.get('status', 'unknown')}\n"
                f"  {drift.get('status_text', '')}"
            )

        # Explainability
        explain = session_data.get('explainability')
        if explain and 'global_importance' in explain:
            top3 = explain['global_importance'][:3]
            exp_lines = [f"  • {f['feature']}: {f['importance']*100:.1f}% SHAP importance" for f in top3]
            context_parts.append("🔍 SHAP EXPLAINABILITY (Top 3):\n" + "\n".join(exp_lines))

        # Pipeline status
        status = session_data.get('current_step', 'unknown')
        context_parts.append(f"⚙️ PIPELINE STATUS: {status}")

        # Hyperopt results
        hyperopt = session_data.get('hyperopt_results')
        if hyperopt:
            context_parts.append(
                f"🔧 HYPEROPTIMIZATION: {hyperopt.get('method', '?')} — "
                f"Best: {hyperopt.get('best_model', '?')} ({hyperopt.get('best_score', 0):.4f})"
            )

        return base + "\n\n".join(context_parts)

    # ── Data Context Builder ──────────────────────────────────

    def _build_data_context(self, df):
        """Build a rich data summary for context."""
        if df is None:
            return "No dataset loaded."

        lines = []
        lines.append(f"**Shape:** {df.shape[0]:,} rows × {df.shape[1]} columns")
        lines.append(f"**Memory:** {df.memory_usage(deep=True).sum() / 1024 / 1024:.1f} MB")
        lines.append(f"**Missing cells:** {df.isnull().sum().sum():,} ({df.isnull().mean().mean()*100:.1f}%)")
        lines.append(f"**Duplicated rows:** {df.duplicated().sum()}")

        # Column details
        lines.append("\n**COLUMNS:**")
        for col in df.columns[:30]:
            dtype = str(df[col].dtype)
            nunique = df[col].nunique()
            null_pct = df[col].isnull().mean() * 100

            if pd.api.types.is_numeric_dtype(df[col]):
                vals = df[col].dropna()
                if len(vals) > 0:
                    lines.append(f"  • {col} ({dtype}): min={vals.min():.2f}, mean={vals.mean():.2f}, max={vals.max():.2f}, std={vals.std():.2f}, null={null_pct:.1f}%, unique={nunique}")
                else:
                    lines.append(f"  • {col} ({dtype}): all null")
            else:
                top_vals = df[col].value_counts().head(3)
                top_str = ", ".join(f"{v}({c})" for v, c in top_vals.items())
                lines.append(f"  • {col} ({dtype}): unique={nunique}, null={null_pct:.1f}%, top=[{top_str}]")

        if df.shape[1] > 30:
            lines.append(f"  ... and {df.shape[1] - 30} more columns")

        # Correlation highlights
        numeric = df.select_dtypes(include='number')
        if numeric.shape[1] >= 2:
            corr = numeric.corr()
            high_corrs = []
            cols = corr.columns.tolist()
            for i in range(len(cols)):
                for j in range(i+1, len(cols)):
                    val = corr.iloc[i, j]
                    if not np.isnan(val) and abs(val) > 0.5:
                        high_corrs.append((cols[i], cols[j], val))
            high_corrs.sort(key=lambda x: abs(x[2]), reverse=True)
            if high_corrs:
                lines.append("\n**TOP CORRELATIONS:**")
                for c1, c2, val in high_corrs[:8]:
                    lines.append(f"  • {c1} ↔ {c2}: {val:.3f}")

        return "\n".join(lines)

    # ── Action Parsing ───────────────────────────────────────

    def _parse_suggested_action(self, response_text):
        """Extract a pipeline action from the LLM response text.
        
        Looks for JSON blocks wrapped in ```action ... ``` or inline
        {"action": "..."} patterns.
        
        Returns:
            dict with 'action' and 'params' keys, or None if no action found.
        """
        if not response_text:
            return None
        
        # Try ```action block first
        action_pattern = r'```action\s*\n?\s*(\{.*?\})\s*\n?\s*```'
        match = re.search(action_pattern, response_text, re.DOTALL)
        if match:
            try:
                action_data = json.loads(match.group(1))
                if 'action' in action_data and action_data['action'] in self.PIPELINE_TOOLS:
                    return {
                        'action': action_data['action'],
                        'params': action_data.get('params', {}),
                    }
            except (json.JSONDecodeError, KeyError):
                pass
        
        # Try inline JSON pattern
        inline_pattern = r'\{"action"\s*:\s*"(\w+)"(?:\s*,\s*"params"\s*:\s*(\{[^}]*\}))?\}'
        match = re.search(inline_pattern, response_text)
        if match:
            action_name = match.group(1)
            if action_name in self.PIPELINE_TOOLS:
                params = {}
                if match.group(2):
                    try:
                        params = json.loads(match.group(2))
                    except json.JSONDecodeError:
                        pass
                return {'action': action_name, 'params': params}
        
        return None

    # ── Rule-Based Fallback Handlers ─────────────────────────

    def _classify_intent(self, message):
        message_lower = message.lower().strip()
        scores = {}
        for intent, config in self.intents.items():
            score = sum(1 for p in config['patterns'] if re.search(p, message_lower))
            if score > 0:
                scores[intent] = score
        return max(scores, key=scores.get) if scores else 'unknown'

    def _handle_explore(self, message, session_data):
        if not session_data or 'df' not in session_data:
            return {'text': '📊 Please upload a dataset first before exploring.'}

        df = session_data['df']
        message_lower = message.lower()
        col = self._extract_column(message, df)

        if 'correlation' in message_lower:
            numeric = df.select_dtypes(include='number')
            if numeric.shape[1] < 2:
                return {'text': 'Not enough numeric columns for correlation analysis.'}
            if col and col in numeric.columns:
                corrs = numeric.corr()[col].sort_values(ascending=False)
                top = corrs.head(6).to_dict()
                text = f"🔗 **Correlations with '{col}':**\n"
                for k, v in top.items():
                    if k != col:
                        bar = '🟢' if abs(v) > 0.5 else '🟡' if abs(v) > 0.3 else '⚪'
                        text += f"\n{bar} {k}: **{v:.3f}**"
                return {'text': text, 'data': top}
            else:
                return {'text': '📊 Correlation matrix computed. Try the **AutoEDA** panel for full visualization!',
                        'chart': {'type': 'heatmap'}}

        if any(w in message_lower for w in ['missing', 'null', 'na']):
            missing = df.isnull().sum()
            missing = missing[missing > 0].sort_values(ascending=False)
            if len(missing) == 0:
                return {'text': '✅ **No missing values found!** Your dataset is complete.'}
            text = f"🩹 **Missing Values ({missing.sum()} total):**\n"
            for col_name, count in missing.head(10).items():
                text += f"\n• {col_name}: **{count}** ({count/len(df)*100:.1f}%)"
            return {'text': text}

        if col and col in df.columns:
            series = df[col]
            if pd.api.types.is_numeric_dtype(series):
                text = f"📊 **Stats for '{col}':**\n"
                text += f"\n• Count: **{series.count():,}**\n• Mean: **{series.mean():.4f}**\n• Median: **{series.median():.4f}**"
                text += f"\n• Std: **{series.std():.4f}**\n• Min: **{series.min():.4f}**\n• Max: **{series.max():.4f}**"
                text += f"\n• Missing: **{series.isnull().sum()}** ({series.isnull().mean()*100:.1f}%)"
                return {'text': text}
            else:
                vc = series.value_counts().head(10)
                text = f"📊 **Value counts for '{col}' (top 10):**\n"
                for v, c in vc.items():
                    text += f"\n• {v}: **{c}** ({c/len(df)*100:.1f}%)"
                return {'text': text}

        text = f"📊 **Dataset Overview:**\n"
        text += f"\n• Rows: **{len(df):,}** | Columns: **{len(df.columns)}**"
        text += f"\n• Numeric: {len(df.select_dtypes(include='number').columns)} | Categorical: {len(df.select_dtypes(include='object').columns)}"
        text += f"\n• Missing: {df.isnull().sum().sum():,} cells ({df.isnull().mean().mean()*100:.1f}%)"
        text += f"\n\n💡 *Try the **AutoEDA** panel for a comprehensive AI-powered analysis!*"
        return {'text': text}

    def _handle_visualize(self, message, session_data):
        return {'text': '📈 Visualization tip: Use the **AutoEDA** panel for automatic charts, or check **Unsupervised → Dimensionality Reduction** for scatter plots!', 'chart': {'type': 'auto'}}

    def _handle_explain(self, message, session_data):
        msg_lower = message.lower()
        if any(w in msg_lower for w in ['counterfactual', 'flip', 'change to', 'what would']):
            return {'text': '🔄 **Counterfactual Explanations:**\n\nAfter training, go to the **Explainability** panel and click **"Counterfactuals"** to see what input changes would flip the prediction.\n\n_Example: "If tenure increased from 2 to 12 months, the prediction would flip from Churn → No Churn."_'}
        return {'text': '🔍 **Model Explanation:**\n\nAfter training, check the **Explainability** tab for:\n• 🌍 SHAP global feature importance\n• 🎯 Per-prediction local explanations\n• 🔄 Counterfactual "what-if" analysis\n• 📈 Partial Dependence Plots\n\n_Train a model first, then the SHAP section will appear automatically!_'}

    def _handle_predict(self, message, session_data):
        return {'text': '🔮 To make predictions:\n\n1. Complete the pipeline (Upload → Clean → Train → Tune)\n2. Go to the **Deploy** panel\n3. Use the **Prediction API** section\n4. Enter feature values and click Predict\n\n_You can also export as a Docker container or standalone script!_'}

    def _handle_compare(self, message, session_data):
        return {'text': '⚖️ Model comparison options:\n\n• **Leaderboard** — See all models ranked after training\n• **Experiments** panel — Compare across different runs with structured diffs\n• **Diagnostics** — ROC curves, confusion matrices per model\n\n_Click the **📚 Experiments** button in the sidebar!_'}

    def _handle_recommend(self, message, session_data):
        return {'text': '💡 **Ways to improve your model:**\n\n1. 🔧 **Feature engineering** — Use the Feature Studio to create new features\n2. ⚖️ **Class imbalance** — Use synthetic data generation\n3. 🎯 **Hyperparameter tuning** — Run optimization from the Tune step\n4. 🧹 **Data quality** — Check the Cleaning Advisor for impact-scored suggestions\n5. 🤖 **Autonomous Agent** — Let the AI try 5 strategies automatically\n6. 📊 **Calibration** — Check prediction confidence calibration\n\n_The platform handles most of this automatically in the Train → Retrain steps!_'}

    def _handle_clean(self, message, session_data):
        return {'text': '🧹 Data cleaning is automatic in Step 2:\n\n• Missing values → smart imputation (mean/median/mode)\n• Duplicates → auto-removed\n• Outliers → detected and handled\n• Encoding → label/one-hot encoding\n\n💡 **New:** Check the **Cleaning Advisor** — it now benchmarks each suggestion\'s impact on model accuracy!\n\n_Just click **Clean & Transform** and the platform handles it!_'}

    def _handle_drift(self, message, session_data):
        return {'text': '📡 **Data Drift Monitor:**\n\nThe Drift Monitor now tracks both **numerical** and **categorical** features:\n• PSI + KS-test for numeric columns\n• Chi-square test for categorical columns\n• Automatic retraining recommendation when drift exceeds thresholds\n\n_Upload new data in the **Drift Monitor** panel to compare against training distributions!_'}

    def _handle_help(self, message, session_data):
        powered = {
            'gemini': '🧠 **Powered by Google Gemini 2.5 Flash** (with conversation memory)',
            'openai': '🧠 **Powered by OpenAI GPT-4o-mini** (with conversation memory)',
            'local': '🤖 **Powered by Local LLM (Flan-T5)**',
            'rules': '💬 **Rule-based assistant**',
        }.get(self.llm_provider, '💬 Assistant')

        return {'text': f'{powered}\n\n**I can help you with:**\n\n• 📊 **Explore data**: "Show me the distribution of age"\n• 🔗 **Correlations**: "What correlates with sales?"\n• 🩹 **Missing data**: "How many missing values?"\n• 🔍 **Explain**: "Why did the model predict churn?"\n• 🔄 **Counterfactuals**: "What would flip this prediction?"\n• 🔮 **Predict**: "How do I make predictions?"\n• ⚖️ **Compare**: "Compare my experiments"\n• 💡 **Improve**: "How can I improve accuracy?"\n• 📡 **Drift**: "Is my model degrading?"\n• 🧹 **Clean**: "How is missing data handled?"\n\n_I remember our conversation — feel free to reference previous messages!_'}

    def _handle_greeting(self, message, session_data):
        provider_msg = {
            'gemini': "I'm powered by **Google Gemini 2.5 Flash** with full context of your current session.",
            'openai': "I'm powered by **GPT-4o-mini** with full context of your current session.",
            'local': "I'm using a **local AI model** for privacy-first responses.",
            'rules': "I'm using **pattern matching** to help you navigate the platform.",
        }.get(self.llm_provider, "")

        return {'text': f'👋 **Hello!** I am the AutoML Studio AI Assistant.\n\n{provider_msg}\n\nI can help you analyze your data, find correlations, explain models, and guide your ML pipeline. Try uploading a dataset and asking me: *"Show dataset summary"* or *"What correlates with the target?"*\n\nType **"help"** to see everything I can do!'}

    def _handle_unknown(self, message, session_data):
        return {'text': '🤔 I\'m not sure how to handle that. Try asking about data exploration, predictions, or model explanations.\n\nType **"help"** for available commands.'}

    def _extract_column(self, message, df):
        message_lower = message.lower()
        for col in df.columns:
            if col.lower() in message_lower:
                return col
        return None