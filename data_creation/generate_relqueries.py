import pandas as pd
from datasets import load_dataset


SHUFFLE_SEED = 42
SHUFFLE_BUFFER = 10_000
N_ROWS = 10_000


def generate_relqueries():
    configs = {
        "rotten_tomatoes": {
            "path": "cornell-movie-review-data/rotten_tomatoes",
            "name": None,
            "split": "train",
            "columns": ["text"],
            "templates": {
                "filtering": (
                    "Decide whether this movie is suitable for children "
                    "based on the synopsis: {text}\n"
                    "Answer with exactly one word: True or False."
                ),
                "classification": (
                    "Categorize the sentiment of the review {text}."
                    "\nRespond with exactly one label: Negative, Positive, "
                    "or Neutral. Do not explain your answer."
                ),
                "summarization": (
                    "Summarize the user's movie review {text} in 20 words "
                    "or less.\nRespond with the summary only -- no "
                    "preamble, no explanation."
                ),
                "qa": (
                    "What genre is this movie likely to be given the "
                    "review: {text}?\nAnswer with a short phrase only -- "
                    "not a full sentence."
                ),
            },
        },
        "amazon_reviews": {
            "path": "json",
            "name": None,
            "split": "train",
            "data_files": "hf://datasets/McAuley-Lab/Amazon-Reviews-2023/raw/review_categories/All_Beauty.jsonl",
            "columns": ["title", "text"],
            "templates": {
                "filtering": (
                    "Decide whether this product is suitable for daily "
                    "use based on the review: {text}\n"
                    "Answer with exactly one word: True or False."
                ),
                "classification": (
                    "Categorize the sentiment of the product review "
                    "{text}.\nRespond with exactly one label: Negative, "
                    "Positive, or Neutral. Do not explain your answer."
                ),
                "summarization": (
                    "Summarize the user's product review {text} in 20 "
                    "words or less.\nRespond with the summary only -- no "
                    "preamble, no explanation."
                ),
                "qa": (
                    "What are the main benefits of the product given the "
                    "review: {text}?\nAnswer with a short phrase only -- "
                    "not a full sentence."
                ),
            },
        },
        "amazon_meta": {
            "path": "parquet",
            "name": None,
            "split": "train",
            # Handmade_Products: ~35% missing price (vs. All_Beauty's ~82%), so the
            # price-dependent filtering/qa templates below actually exercise real
            # price values most of the time instead of "price: None" almost always.
            "data_files": "hf://datasets/McAuley-Lab/Amazon-Reviews-2023/raw_meta_Handmade_Products/full-00000-of-00001.parquet",
            "columns": ["title", "average_rating", "rating_number", "price", "store"],
            "templates": {
                "filtering": (
                    "Decide whether this product is premium-priced based "
                    "on its price: {price} and average rating: "
                    "{average_rating}.\n"
                    "Answer with exactly one word: True or False."
                ),
                "classification": (
                    "Categorize this product's popularity given its "
                    "rating count: {rating_number} and average rating: "
                    "{average_rating}.\nRespond with exactly one label: "
                    "Low, Medium, or High. Do not explain your answer."
                ),
                "summarization": (
                    "Summarize this product listing in one sentence given "
                    "its title: {title} and store: {store}.\nRespond with "
                    "the summary only -- no preamble, no explanation."
                ),
                "qa": (
                    "What price range would you expect for a product "
                    "titled {title} sold by {store}?\nAnswer with a short "
                    "phrase only -- not a full sentence."
                ),
            },
        },
        "wine_reviews": {
            "path": "spawn99/wine-reviews",
            "name": None,
            "split": "train",
            "columns": ["description", "winery"],
            "templates": {
                "filtering": (
                    "Determine whether this wine is suitable for a "
                    "beginner given the description: {description}\n"
                    "Answer with exactly one word: True or False."
                ),
                "classification": (
                    "Categorize the sentiment of the wine review "
                    "{description}.\nRespond with exactly one label: "
                    "Negative, Positive, or Neutral. Do not explain your "
                    "answer."
                ),
                "summarization": (
                    "Summarize the wine review {description} in 20 words "
                    "or less.\nRespond with the summary only -- no "
                    "preamble, no explanation."
                ),
                "qa": (
                    "What foods pair best with this wine given its "
                    "description: {description}?\nAnswer with a short "
                    "phrase only -- not a full sentence."
                ),
            },
        },
    }

    for dataset_key, config in configs.items():
        print(f"Processing dataset: {dataset_key}...")

        try:
            load_kwargs = {"split": config["split"], "streaming": True}
            if config["name"]:
                load_kwargs["name"] = config["name"]
            if "data_files" in config:
                load_kwargs["data_files"] = config["data_files"]
            dataset = load_dataset(config["path"], **load_kwargs)
            dataset = dataset.shuffle(seed=SHUFFLE_SEED, buffer_size=SHUFFLE_BUFFER)
            rows = list(dataset.take(N_ROWS))
            df = pd.DataFrame(rows)
        except Exception as e:
            print(f"Failed to load {dataset_key}: {e}")
            continue

        for col in config["columns"]:
            if col in df.columns:
                # Amazon metadata stores missing values as the literal string
                # "None", not NaN, so fillna alone won't catch them.
                df[col] = df[col].fillna("Unknown").replace("None", "Unknown")

        for query_type, template_str in config["templates"].items():
            col_name = f"prompt_{query_type}"
            df[col_name] = df.apply(
                lambda row: template_str.format(
                    **{k: row[k] for k in config["columns"] if k in df.columns}
                ),
                axis=1,
            )

        prompt_columns = [f"prompt_{qt}" for qt in config["templates"].keys()]
        output_file = f"{dataset_key}_10k_relqueries.jsonl"
        df[prompt_columns].to_json(output_file, orient="records", lines=True)
        print(f"-> Saved {len(df)} rows with {len(prompt_columns)} prompt variations to {output_file}\n")


if __name__ == "__main__":
    generate_relqueries()