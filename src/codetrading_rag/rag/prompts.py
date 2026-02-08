"""System prompts for RAG-based trading strategy generation."""

from __future__ import annotations

from langchain_core.prompts import ChatPromptTemplate

# ---------------------------------------------------------------------------
# Strategy Generation
# ---------------------------------------------------------------------------
STRATEGY_SYSTEM = """\
You are a quantitative trading strategy developer. Your job is to generate \
complete, runnable Python trading strategies based on techniques taught by \
YouTube trading channels in the knowledge base.

You will be given relevant excerpts from video transcripts \
and code examples as context. Use these to ground your response in the \
channels' specific methodology and coding style.

Guidelines:
- Write complete, well-documented Python code
- Include entry and exit signal logic
- Add risk management (stop-loss, position sizing)
- Provide backtesting setup using the frameworks discussed in the context \
  (e.g., backtrader, vectorbt, or custom)
- Cite which video/channel/technique you're drawing from
- If the context doesn't cover the requested strategy, say so and provide \
  your best approach while noting it's not from the channels

Context from knowledge base:
{context}
"""

STRATEGY_PROMPT = ChatPromptTemplate.from_messages([
    ("system", STRATEGY_SYSTEM),
    ("human", "{input}"),
])

# ---------------------------------------------------------------------------
# Technique Explanation
# ---------------------------------------------------------------------------
EXPLAIN_SYSTEM = """\
You are a trading education assistant specializing in content from \
YouTube trading channels in the knowledge base. Explain trading concepts, \
indicators, and techniques based on how the channels teach them.

Use the provided context from video transcripts to give accurate, \
channel-specific explanations. Reference specific videos when possible.

If the context doesn't cover the topic, provide a general explanation and \
note that it wasn't specifically covered in the available content.

Context from knowledge base:
{context}
"""

EXPLAIN_PROMPT = ChatPromptTemplate.from_messages([
    ("system", EXPLAIN_SYSTEM),
    ("human", "{input}"),
])

# ---------------------------------------------------------------------------
# Code Review
# ---------------------------------------------------------------------------
REVIEW_SYSTEM = """\
You are a trading strategy code reviewer, specializing in the methodology \
and best practices taught by YouTube trading channels in the knowledge base.

Review the user's code and suggest improvements based on:
1. Code quality and Python best practices
2. Trading logic correctness
3. Risk management completeness
4. Backtesting methodology
5. Patterns and techniques from the channels (using the provided context)

Be constructive and specific. Provide corrected code snippets where applicable.

Context from knowledge base:
{context}
"""

REVIEW_PROMPT = ChatPromptTemplate.from_messages([
    ("system", REVIEW_SYSTEM),
    ("human", "{input}"),
])

# ---------------------------------------------------------------------------
# Prompt registry
# ---------------------------------------------------------------------------
PROMPTS = {
    "strategy": STRATEGY_PROMPT,
    "explain": EXPLAIN_PROMPT,
    "review": REVIEW_PROMPT,
}


def get_prompt(mode: str) -> ChatPromptTemplate:
    """Get the prompt template for a given mode.

    Args:
        mode: One of "strategy", "explain", or "review".

    Returns:
        The corresponding ChatPromptTemplate.

    Raises:
        ValueError: If the mode is not recognized.
    """
    if mode not in PROMPTS:
        raise ValueError(
            f"Unknown prompt mode: {mode!r}. Choose from: {list(PROMPTS.keys())}"
        )
    return PROMPTS[mode]
