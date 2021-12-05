import argparse
import ast
import json
import os
import string


def add_start_end(tokens, start_word="<START>", end_word="<END>"):
    """
    Add start and end words for a caption
    """
    tokens_processed = [start_word]
    tokens_processed.extend(tokens)
    tokens_processed.append(end_word)
    return tokens_processed


# preprocess all the caption
def prepro_captions(json_ann):
    print("example processed tokens:")
    for i, anno in enumerate(json_ann):
        anno["processed_tokens"] = []
        for j, s in enumerate(anno["captions"]):
            txt = (
                str(s)
                .lower()
                .translate(str.maketrans("", "", string.punctuation))
                .strip()
                .split()
            )
            anno["processed_tokens"].append(txt)


def build_vocab(args, json_ann):
    count_thr = args.word_count_threshold
    # count up the number of words
    counts = {}
    for anno in json_ann:
        for txt in anno["processed_tokens"]:
            for w in txt:
                counts[w] = counts.get(w, 0) + 1
    cw = sorted([(count, w) for w, count in counts.items()], reverse=True)
    print("most words and their counts:")
    print("\n".join(map(str, cw[:20])))
    print("least words and their counts:")
    print("\n".join(map(str, cw[-20:])))

    # print some stats
    total_words = sum(counts.values())
    print("total words:", total_words)
    bad_words = [w for w, n in counts.items() if n <= count_thr]
    vocab = [w for w, n in counts.items() if n > count_thr]
    bad_count = sum(counts[w] for w in bad_words)
    print(
        "number of bad words: %d/%d = %.2f%%"
        % (len(bad_words), len(counts), len(bad_words) * 100.0 / len(counts))
    )
    print("number of words in vocab would be %d" % (len(vocab),))
    print(
        "number of UNKs: %d/%d = %.2f%%"
        % (bad_count, total_words, bad_count * 100.0 / total_words)
    )

    # lets look at the distribution of lengths as well
    sent_lengths = {}
    for anno in json_ann:
        for txt in anno["processed_tokens"]:
            nw = len(txt)
            sent_lengths[nw] = sent_lengths.get(nw, 0) + 1
    max_len = max(sent_lengths.keys())
    print("max length sentence in raw data: ", max_len)
    print("sentence length distribution (count, number of words):")
    sum_len = sum(sent_lengths.values())
    for i in range(max_len + 1):
        print(
            "%2d: %10d   %f%%"
            % (i, sent_lengths.get(i, 0), sent_lengths.get(i, 0) * 100.0 / sum_len)
        )

    # lets now produce the final annotations
    if bad_count > 0:
        # additional special UNK token we will use below to map infrequent words to
        print("inserting the special UNK token")
        vocab.append("UNK")

    for anno in json_ann:
        anno["final_captions"] = []
        for txt in anno["processed_tokens"]:
            caption = [w if counts.get(w, 0) > count_thr else "UNK" for w in txt]
            anno["final_captions"].append(caption)
    return vocab


# Parse the attribute json file
def parse_att_json(att_list, dictionary):
    att_dict = {
        "head": [],
        "upperbody": [],
        "lowerbody": [],
        "shoe": [],
        "backpack": [],
    }
    for attribute in att_list:
        key = list(attribute.keys())[0]
        values = attribute[key]
        phrase = []

        for word in values:
            if key == "person" and values == ["person"]:
                continue
            if not word.isalpha():
                words = word.replace("/", " ")
                for word in words.split(" "):
                    word = word.lower().translate(
                        str.maketrans("", "", string.punctuation)
                    )
                    if word in dictionary.keys():
                        phrase.append(dictionary[word])
                continue

            word = word.lower().translate(str.maketrans("", "", string.punctuation))
            if word in dictionary.keys():
                phrase.append(dictionary[word])

        if key == "hair" or key == "hat" or key == "person":
            key = "head"
        if key == "other":
            key = "upperbody"
        att_dict[key] += phrase
    return att_dict


def main(args):
    splits = ["train", "val", "test"]
    json_name = "%s.json"
    anno_file = "reid_raw.json"
    att_dir = os.path.join(args.datadir, "text_attribute_graph")
    json_ann = json.load(open(os.path.join(args.datadir, anno_file)))

    # tokenization and preprocessing
    prepro_captions(json_ann)
    # create the vocab
    vocab = build_vocab(args, json_ann)
    wtoi = {w: i + 1 for i, w in enumerate(vocab)}  # inverse table

    image_id = 0
    for split in splits:
        print("Starting %s" % split)
        ann_dict = {}
        annotations = []
        id_collect = {}
        img_collect = {}
        for anno in json_ann:
            if anno["split"] != split:
                continue

            n = len(anno["final_captions"])
            assert n > 0, "error: some image has no captions"

            image_id += 1
            for cap_idx, cap in enumerate(anno["final_captions"]):
                ann = {}
                ann["image_id"] = image_id
                ann["id"] = anno["id"] - 1
                id_collect[ann["id"]] = id_collect.get(ann["id"], 0) + 1
                ann["file_path"] = anno["file_path"]
                img_collect[ann["file_path"]] = img_collect.get(ann["file_path"], 0) + 1
                ann["sentence"] = anno["captions"][cap_idx]
                ann["onehot"] = []
                for k, w in enumerate(cap):
                    if k < args.max_length:
                        ann["onehot"].append(wtoi[w])

                # Load the parsed attribute and write to the processed files - Jacob
                att_json_file = anno["file_path"].replace("/", "-")
                with open(
                    os.path.join(att_dir, att_json_file + "-" + str(cap_idx) + ".json"),
                    "r",
                ) as f:
                    att_dict = parse_att_json(ast.literal_eval(f.read()), wtoi)
                ann["att_onehot"] = att_dict
                annotations.append(ann)

        categories = [k for k, v in id_collect.items()]
        ann_dict["categories"] = categories
        ann_dict["annotations"] = annotations
        print("Num id-persons: %s" % len(id_collect))
        print("Num images: %s" % len(img_collect))
        print("Num annotations: %s" % len(annotations))
        with open(os.path.join(args.outdir, json_name % split), "w") as outfile:
            outfile.write(json.dumps(ann_dict))
    return wtoi


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--datadir", default="", type=str, help="CUHK-PEDES dataset root directory"
    )
    parser.add_argument(
        "--outdir", default="", type=str, help="Root saving path for annotation files"
    )
    # options
    parser.add_argument(
        "--max_length",
        default=100,
        type=int,
        help="max length of a caption, in number of words. captions longer than this get clipped.",
    )
    parser.add_argument(
        "--word_count_threshold",
        default=2,
        type=int,
        help="only words that occur more than this number of times will be put in vocab",
    )

    args = parser.parse_args()
    dict = main(args)
