import pandas as pd

SOURCE = "arxiv_data.csv"   # your file with titles,summaries,terms
OUTPUT = "papers.csv"
MAX_ROWS = 500              # or any small number for demo

def main():
    df = pd.read_csv(SOURCE)

    # Expect exactly these columns
    required = {"titles", "summaries", "terms"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing expected columns in source CSV: {missing}")

    # Optional: filter by keyword in summaries, e.g. "quantum"
    df_filtered = df[df["summaries"].astype(str).str.contains("quantum", case=False, na=False)]

    if df_filtered.empty:
        print("Filter produced no rows, using first MAX_ROWS rows instead.")
        df_filtered = df

    # Map to the format RAG script expects
    df_out = pd.DataFrame({
        "id": range(1, len(df_filtered) + 1),
        "title": df_filtered["titles"],
        "abstract": df_filtered["summaries"],
    }).head(MAX_ROWS)

    df_out.to_csv(OUTPUT, index=False)
    print(f"Saved {len(df_out)} rows to {OUTPUT}")

if __name__ == "__main__":
    main()
