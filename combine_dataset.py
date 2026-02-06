from datasets import load_dataset

ds = load_dataset("wikimedia/wikipedia", "20231101.en", split="train")
print(ds.keys())        # see fields, usually "text", "title", etc.
print(ds["text"])       # first article text
ds.to_csv("wikipedia_en.csv")  # or iterate and write your own .txt
