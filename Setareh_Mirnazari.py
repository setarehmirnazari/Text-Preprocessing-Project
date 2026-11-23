import json
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

DEFAULT_STOP_WORDS: Set[str] = {
    "a", "an", "and", "are", "as", "at", "be", "been", "being", "but", "by",
    "for", "from", "had", "has", "have", "he", "her", "hers", "herself", "him",
    "himself", "his", "i", "if", "in", "into", "is", "it", "its", "itself",
    "just", "me", "more", "most", "my", "myself", "no", "nor", "not", "of",
    "off", "on", "once", "only", "or", "other", "our", "ours", "ourselves",
    "out", "over", "own", "s", "same", "she", "should", "so", "some", "such",
    "t", "than", "that", "the", "their", "theirs", "them", "themselves", "then",
    "there", "these", "they", "this", "those", "through", "to", "too", "under",
    "until", "up", "very", "was", "we", "were", "what", "when", "where", "which",
    "while", "who", "whom", "why", "will", "with", "you", "your", "yours",
    "yourself", "yourselves",
}


class TextPreprocessor:
    TOKEN_PATTERN = re.compile(
        r"""
        (?:[A-Za-z]{1,3}\.){2,}[A-Za-z]?\.?
        | [A-Za-z0-9]+(?:-[A-Za-z0-9]+)+
        | [A-Za-z0-9]+(?:['’][sS])?
        """,
        re.VERBOSE,
    )

    def __init__(self, stop_words: Optional[Set[str]] = None) -> None:
        self.stop_words: Set[str] = {
            w.lower()
            for w in (
                stop_words if stop_words is not None else DEFAULT_STOP_WORDS
            )
        }

    def tokenize(self, text: str) -> List[str]:
        raw_tokens = self.TOKEN_PATTERN.findall(text)
        cleaned_tokens: List[str] = []

        for tok in raw_tokens:
            tok = re.sub(r"['’][sS]$", "", tok)

            if re.fullmatch(r"(?:[A-Za-z]{1,3}\.){2,}[A-Za-z]?\.?", tok):
                tok = tok.rstrip(".")
                if tok:
                    cleaned_tokens.append(tok)
                continue

            tok = tok.strip(".,!?;:\"'()[]{}<>")
            if tok:
                cleaned_tokens.append(tok)

        return cleaned_tokens

    def normalize(self, tokens: List[str]) -> List[str]:
        normalized: List[str] = []

        for tok in tokens:
            lower_tok = tok.lower()

            if re.fullmatch(r"(?:[a-z]{1,3}\.){1,}[a-z]+", lower_tok):
                lower_tok = lower_tok.replace(".", "")

            normalized.append(lower_tok)

        return normalized

    def remove_stop_words(self, tokens: List[str]) -> List[str]:
        return [tok for tok in tokens if tok not in self.stop_words]

    def stem_word(self, word: str) -> str:
        if len(word) <= 2:
            return word

        if "-" in word:
            parts = [
                self._stem_single(part) if len(part) > 2 else part
                for part in word.split("-")
            ]
            return "-".join(parts)

        return self._stem_single(word)

    def _stem_single(self, word: str) -> str:
        """
        Porter Stemming Algorithm
        Improved rule-based English stemmer.
        """

        w = word.lower()

        if len(w) <= 2:
            return w

        vowels = "aeiou"

        def is_consonant(i: int) -> bool:
            if w[i] in vowels:
                return False
            if w[i] == "y":
                return i == 0 or not is_consonant(i - 1)
            return True

        def measure(stem: str) -> int:
            m = 0
            in_vowel = False

            for i in range(len(stem)):
                if not is_consonant(i):
                    in_vowel = True
                elif in_vowel:
                    m += 1
                    in_vowel = False

            return m

        def has_vowel(stem: str) -> bool:
            return any(not is_consonant(i) for i in range(len(stem)))

        def double_consonant(stem: str) -> bool:
            return (
                len(stem) >= 2
                and stem[-1] == stem[-2]
                and is_consonant(len(stem) - 1)
            )

        def cvc(stem: str) -> bool:
            if len(stem) < 3:
                return False

            return (
                is_consonant(len(stem) - 3)
                and not is_consonant(len(stem) - 2)
                and is_consonant(len(stem) - 1)
                and stem[-1] not in "wxy"
            )

        def replace_suffix(
            suffix: str,
            replacement: str,
            min_m: int = 0
        ) -> bool:
            nonlocal w

            if not w.endswith(suffix):
                return False

            stem = w[:-len(suffix)]

            if measure(stem) > min_m:
                w = stem + replacement
                return True

            return False

        # STEP 1a: Plurals
        if w.endswith("sses"):
            w = w[:-2]
        elif w.endswith("ies"):
            w = w[:-2]
        elif w.endswith("ss"):
            pass
        elif w.endswith("s"):
            w = w[:-1]

        # STEP 1b: Past tense and gerunds
        step1b = False

        if w.endswith("eed"):
            stem = w[:-3]
            if measure(stem) > 0:
                w = stem + "ee"

        elif w.endswith("ed") and has_vowel(w[:-2]):
            w = w[:-2]
            step1b = True

        elif w.endswith("ing") and has_vowel(w[:-3]):
            w = w[:-3]
            step1b = True

        if step1b:
            if w.endswith(("at", "bl", "iz")):
                w += "e"
            elif double_consonant(w) and w[-1] not in "lsz":
                w = w[:-1]
            elif measure(w) == 1 and cvc(w):
                w += "e"

        # STEP 1c: Replace y with i
        if w.endswith("y") and has_vowel(w[:-1]):
            w = w[:-1] + "i"

        # STEP 2: Derivational suffixes
        step2_rules = [
            ("ational", "ate"),
            ("tional", "tion"),
            ("enci", "ence"),
            ("anci", "ance"),
            ("izer", "ize"),
            ("abli", "able"),
            ("alli", "al"),
            ("entli", "ent"),
            ("eli", "e"),
            ("ousli", "ous"),
            ("ization", "ize"),
            ("ation", "ate"),
            ("ator", "ate"),
            ("alism", "al"),
            ("iveness", "ive"),
            ("fulness", "ful"),
            ("ousness", "ous"),
            ("aliti", "al"),
            ("iviti", "ive"),
            ("biliti", "ble"),
            ("logi", "log"),
        ]

        for suffix, replacement in step2_rules:
            if replace_suffix(suffix, replacement, 0):
                break

        # STEP 3: Additional derivational suffixes
        step3_rules = [
            ("icate", "ic"),
            ("ative", ""),
            ("alize", "al"),
            ("iciti", "ic"),
            ("ical", "ic"),
            ("ful", ""),
            ("ness", ""),
        ]

        for suffix, replacement in step3_rules:
            if replace_suffix(suffix, replacement, 0):
                break

        # STEP 4: Remove remaining suffixes
        step4_suffixes = [
            "al", "ance", "ence", "er", "ic", "able",
            "ible", "ant", "ement", "ment", "ent",
            "ion", "ou", "ism", "ate", "iti",
            "ous", "ive", "ize"
        ]

        for suffix in step4_suffixes:
            if w.endswith(suffix):
                stem = w[:-len(suffix)]

                if measure(stem) > 1:
                    if suffix == "ion":
                        if stem.endswith(("s", "t")):
                            w = stem
                    else:
                        w = stem

                break

        # STEP 5a: Remove final e
        if w.endswith("e"):
            stem = w[:-1]

            if measure(stem) > 1 or (
                measure(stem) == 1 and not cvc(stem)
            ):
                w = stem

        # STEP 5b: Reduce double l
        if w.endswith("ll") and measure(w) > 1:
            w = w[:-1]

        return w

    def stem(self, tokens: List[str]) -> List[str]:
        return [self.stem_word(tok) for tok in tokens]

    def process_text(self, text: str) -> List[str]:
        tokens = self.tokenize(text)
        normalized = self.normalize(tokens)
        filtered = self.remove_stop_words(normalized)
        stemmed = self.stem(filtered)

        return stemmed


def evaluate_terms(
    predicted_terms: List[str],
    gold_terms: List[str]
) -> Dict[str, object]:

    o_set: Set[str] = set(predicted_terms)
    g_set: Set[str] = set(gold_terms)

    if not o_set or not g_set:
        precision = 0.0
        recall = 0.0
    else:
        intersection_size = len(o_set & g_set)
        precision = intersection_size / len(o_set)
        recall = intersection_size / len(g_set)

    extra_terms = sorted(o_set - g_set)
    missing_terms = sorted(g_set - o_set)

    return {
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "extra_terms_in_output": extra_terms,
        "missing_terms_from_gold": missing_terms,
    }


def read_inputs_from_file(file_path: Path) -> Dict[str, str]:
    content = file_path.read_text(encoding="utf-8").strip()

    if not content:
        return {}

    if file_path.suffix.lower() == ".json":
        data = json.loads(content)

        if isinstance(data, dict):
            return {str(k): str(v) for k, v in data.items()}

        if isinstance(data, list):
            return {
                f"text{idx+1}": str(item)
                for idx, item in enumerate(data)
            }

        raise ValueError("Invalid JSON structure.")

    lines = [
        line.strip()
        for line in content.splitlines()
        if line.strip()
    ]

    return {
        f"text{idx+1}": line
        for idx, line in enumerate(lines)
    }


def read_inputs_interactive() -> Tuple[
    Dict[str, str],
    Dict[str, List[str]]
]:
    texts: Dict[str, str] = {}
    gold_data: Dict[str, List[str]] = {}

    print("=== دریافت متون ورودی برای پیش پردازش ===")

    while True:
        count_str = input(
            "تعداد متن های ورودی را وارد کنید: "
        ).strip()

        if count_str.isdigit() and int(count_str) > 0:
            count = int(count_str)
            break

        print("لطفا یک عدد صحیح بزرگ تر از صفر وارد کنید.")

    for i in range(1, count + 1):
        print(f"\nمتن شماره {i} را وارد کنید:")
        text = input(">").strip()

        key = f"text{i}"
        texts[key] = text

        print(
            f"کلمات کلیدی استاندارد طلایی (Gold) برای متن شماره {i} "
            "را وارد کنید (با ویرگول جدا کنید، یا اینتر بزنید تا خالی بماند):"
        )

        gold_input = input("Gold>").strip()

        if gold_input:
            gold_terms = [
                t.strip().lower()
                for t in gold_input.split(",")
                if t.strip()
            ]

            gold_data[key] = gold_terms
        else:
            gold_data[key] = []

    return texts, gold_data


def main() -> None:
    preprocessor = TextPreprocessor()

    args = [
        arg for arg in sys.argv
        if not arg.startswith("-")
        and "jupyter" not in arg
        and "ipykernel" not in arg
    ]

    gold_data: Dict[str, List[str]] = {}

    if len(args) >= 2:
        input_path = Path(args[1])

        if not input_path.exists():
            print(f"خطا: فایل ورودی '{input_path}' یافت نشد.")
            sys.exit(1)

        texts = read_inputs_from_file(input_path)

        if len(args) >= 3:
            gold_path = Path(args[2])

            if gold_path.exists() and gold_path.suffix.lower() == ".json":
                gold_data = json.loads(
                    gold_path.read_text(encoding="utf-8")
                )

    else:
        texts, gold_data = read_inputs_interactive()

    outputs: Dict[str, List[str]] = {}

    for key, raw_text in texts.items():
        outputs[key] = preprocessor.process_text(raw_text)

    output_file = Path("outputs.json")

    output_file.write_text(
        json.dumps(outputs, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("\n==========================================")
    print("خروجی پیش پردازش (ذخیره شده در outputs.json)")
    print("==========================================")

    print(
        json.dumps(
            outputs,
            ensure_ascii=False,
            indent=2
        )
    )

    # Perform evaluation if we have any gold data
    if gold_data:
        print("\n==========================================")
        print("ارزیابی معیارهای Precision و Recall")
        print("==========================================")

        for key, pred_terms in outputs.items():
            gold_terms = gold_data.get(key, [])

            if isinstance(gold_terms, list):
                metrics = evaluate_terms(
                    pred_terms,
                    [str(t) for t in gold_terms]
                )

                print(f"[{key}]:")
                print(f" - Precision: {metrics['precision']}")
                print(f" - Recall: {metrics['recall']}")
                print(
                    f" - کلمات اضافی (O - G): "
                    f"{metrics['extra_terms_in_output']}"
                )
                print(
                    f" - کلمات جا افتاده (G - O): "
                    f"{metrics['missing_terms_from_gold']}\n"
                )


if __name__ == "__main__":
    main()
