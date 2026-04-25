import pandas as pd

def recommend(df, selected_cuisines):

    df = df.copy()
    df["Cuisines"] = df["Cuisines"].fillna("")

    results = []

    for _, row in df.iterrows():
        cuisines = row["Cuisines"]

        score = sum([1 for c in selected_cuisines if c in cuisines])

        if score > 0:
            results.append((score, row))

    results = sorted(results, key=lambda x: x[0], reverse=True)

    return [r[1] for r in results[:5]]