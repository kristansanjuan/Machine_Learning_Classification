import os

def load_texts_and_labels():
    folder_paths = {
        "left": "C:\\Users\\A Gaming\\Downloads\\archive\\Data\\Left Data",
        "right": "C:\\Users\\A Gaming\\Downloads\\archive\\Data\\Right Data",
        "center": "C:\\Users\\A Gaming\\Downloads\\archive\\Data\\Center Data"
    }

    texts = []
    labels = []

    for label, folder in folder_paths.items():
        for filename in os.listdir(folder):
            if filename.endswith(".txt"):
                path = os.path.join(folder, filename)
                with open(path, "r", encoding="utf-8") as f:
                    texts.append(f.read())
                    labels.append(label)

    print(f"Loaded {len(texts)} texts from {len(folder_paths)} folders.")
    return texts, labels