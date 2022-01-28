import random
from typing import List

from flair.data import Sentence, SpanLabel


class SentenceShortener:

    def __init__(self, augmentation_probability: float = 0.5, tag: str = "ner"):
        self.augmentation_probability: float = augmentation_probability
        self.tag: str = tag

    def augment(self, sentences: List[Sentence]) -> List[Sentence]:

        if isinstance(sentences, Sentence):
            sentences = [sentences]

        shortened_sentences: List[Sentence] = []
        for sentence in sentences:
            if random.uniform(0, 1) < self.augmentation_probability and len(sentence.get_labels(self.tag)) > 0:

                # randomly select a label
                center_label: SpanLabel = random.choice(sentence.get_labels(self.tag))

                start_idx = center_label.span.tokens[0].idx
                stop_idx = center_label.span.tokens[-1].idx

                before_tokens = random.randint(0, 2)
                if random.uniform(0, 1) < 0.2: before_tokens = 0
                after_tokens = random.randint(0, 2)
                if random.uniform(0, 1) < 0.2: after_tokens = 0

                offset = 0
                new_text = []
                for i in range(start_idx - before_tokens - 1, start_idx - 1):
                    if i > 0:
                        new_text.append(sentence[i].text)
                        offset += 1

                for token in center_label.span.tokens:
                    new_text.append(token.text)

                for i in range(stop_idx, stop_idx + after_tokens):
                    if i < len(sentence) - 1:
                        new_text.append(sentence[i].text)

                shortened = Sentence(new_text, use_tokenizer=False)
                shortened[offset:offset + len(center_label)].add_tag(self.tag, center_label.value)

                shortened_sentences.append(shortened)
            else:
                shortened_sentences.append(sentence)

        return shortened_sentences
