import pandas as pd
import streamlit as st

import scorer

st.set_page_config(page_title="Essay Scorer", page_icon="📝")
st.title("📝 Essay Scorer")
st.caption(
    "Dual-process scoring (Xiao et al., 2025) with LLaMA-3-8B. "
    "Trained on ASAP set 1 only: letters about the effects of computers on people."
)


@st.cache_resource(show_spinner="Loading model (first time only)...")
def get_model():
    return scorer.load_model()


essay = st.text_area("Paste an English essay", height=300)
words = len(essay.split())
st.caption(f"{words} words")

if st.button("Score essay", type="primary"):
    if words < 50:
        st.warning("Please paste at least 50 words.")
    else:
        model, tokenizer = get_model()
        with st.spinner("Scoring..."):
            result = scorer.score_essay(essay, model, tokenizer)

        lo, hi = scorer.RANGES["overall"]
        st.metric("Overall score", f"{result['overall']['score']} / {hi}")

        rows = []
        for trait, r in result.items():
            if trait == "overall":
                continue
            rows.append({
                "Trait": trait.replace("_", " ").title(),
                "Score": f"{r['score']} / {scorer.RANGES[trait][1]}",
                "Confidence": r["confidence"],
                "Module": r["module"],
            })
        st.dataframe(
            pd.DataFrame(rows),
            hide_index=True,
            column_config={"Confidence": st.column_config.ProgressColumn(min_value=0, max_value=1)},
        )
        st.caption("Fast = classifier on embeddings. Slow = model writes the scores (used when confidence < 0.2).")
