"""

Basic data structure and functions for manipulating spans

author: kylel@allenai.org

"""

from typing import *

class Span:
    """When comparing `self` to another span `other`, there are cases:

            self            other
    (a)     (0, 3)    <     (3, 5)          disjoint   (i.e. < and > operators)
    (b)     (0, 3)    <=    (2, 5)          partial overlap    (i.e. <= and >= operators)
            (0, 3)    <=    (1, 5)
            (0, 3)    <=    (0, 5)
    (c)     (1, 2)    in    (0, 3)          (strict) subset   (i.e. `in` operator)
    (d)     (0, 3)    ==    (0, 3)          equal

    Notice that the `stop` index is non-inclusive.
    """
    def __init__(self, start: int, stop: int):
        if start >= stop:
            raise ValueError('Strictly start < stop')
        self.start = start
        self.stop = stop

    def __eq__(self, other):
        return self.start == other.start and self.stop == other.stop

    def __lt__(self, other):
        return self.stop <= other.start

    def __gt__(self, other):
        return other.__lt__(self)

    def __le__(self, other):
        # self is left of other, but not disjoint
        return self.start <= other.start and self.stop < other.stop and not self < other

    def __ge__(self, other):
        return other.__le__(self)

    def __repr__(self):
        return str((self.start, self.stop))

    def __len__(self):
        return self.stop - self.start + 1

    def __contains__(self, item):
        """Only for strict subset, doesn't include equality"""
        return item.start > self.start and item.stop < self.stop

    @classmethod
    def sort_cluster_spans(cls, spans: Set['Span']) -> List[List['Span']]:
        """Iterate over spans; accumulate each span in same group"""
        spans: List['Span'] = sorted(spans, key=lambda s: (s.start, s.stop))
        clusters: List[Dict] = [{
            'proxy': Span(start=spans[0].start, stop=spans[0].stop),
            'spans': [spans[0]]
        }]
        for span in spans[1:]:
            # if this span is disjoint from the previous spans, start new cluster
            if span > clusters[-1]['proxy']:
                clusters.append({
                    'proxy': Span(start=span.start, stop=span.stop),
                    'spans': [span]
                })
            # otherwise, add to previous group
            else:
                clusters[-1]['spans'].append(span)
                clusters[-1]['proxy'] = Span(start=clusters[-1]['proxy'].start,
                                             stop=max(clusters[-1]['proxy'].stop, span.stop))
        return [cluster['spans'] for cluster in clusters]

    def to_json(self) -> Dict:
        return {
            'start': self.start,
            'stop': self.stop
        }


class TokenSpan(Span):
    def __init__(self, start: int, stop: int, text: str):
        super().__init__(start, stop)
        self.text = text

    def __repr__(self):
        return str((self.start, self.stop, self.text))

    def to_json(self) -> Dict:
        return {
            'start': self.start,
            'stop': self.stop,
            'text': self.text
        }

    @classmethod
    def find_sent_token_spans(cls, text: str, sent_tokens: List[List[str]]) -> List[List['TokenSpan']]:
        """
        Given text and its tokenization, associate with each token a span
        that indexes characters from the original text

        text before tokenization:
            'Hi, this is.'
        tokens and their associated char-level spans:
            Hi      -> (0,2)
            ,       -> (2,3)
            this    -> (4,8)
            is      -> (9,11)
            .       -> (11,12)
        where span ends are non-inclusive

        This should work for arbitrary tokenization (even sub-word tokenization),
        as long as non-whitespace characters never disappear after tokenization.
        """

        # assertion should fail if any tokens are whitespace, which can happen like:
        #   text = "This is     too    much white    space ."
        assert ''.join([char.strip() for char in text.strip()]) == ''.join([token for tokens in sent_tokens for token in tokens])
        sent_spans = []
        index_char_in_text = 0
        for tokens in sent_tokens:
            token_spans, index_char_in_text = TokenSpan._find_token_spans(text, tokens, index_char_in_text)
            sent_spans.append(token_spans)
        return sent_spans

    @classmethod
    def _find_token_spans(cls, text: str, tokens: List[str], index_char_in_text: int) -> Tuple[List['TokenSpan'], int]:
        """Private method to process each sentence in `_find_sent_token_spans()`"""
        token_spans = []
        for token in tokens:
            # skip whitespace
            while text[index_char_in_text].strip() == '':
                index_char_in_text += 1

            # remember start of span
            start = index_char_in_text

            # iterate over token characters
            for char in token:
                index_char_in_text += 1

            # save span when match all characters in token
            assert token == text[start:index_char_in_text]
            token_span = TokenSpan(start=start, stop=index_char_in_text, text=token)
            token_spans.append(token_span)

        return token_spans, index_char_in_text


class MentionSpan(Span):
    def __init__(self, start: int, stop: int, text: str, entity_types: List[str], entity_id: str):
        super().__init__(start, stop)
        self.text = text
        self.entity_types = entity_types
        self.entity_id = entity_id

    def __repr__(self):
        return str((self.start, self.stop, self.text))

    def to_json(self) -> Dict:
        return {
            'start': self.start,
            'stop': self.stop,
            'text': self.text,
            'entity_types': self.entity_types,
            'entity_id': self.entity_id
        }

    def __hash__(self):
        """This is more strict than equality, which is inhereted from `Span.__eq__`

        e.g.
            MentionSpan(0, 1, '', [], '') == MentionSpan(0, 1, 'abc', [], 'abc') is True
            hash(MentionSpan(0, 1, '', [], '')) == hash(MentionSpan(0, 1, 'abc', [], 'abc')) is False
        """
        return hash((self.start, self.stop, self.text, tuple(self.entity_types), self.entity_id))


def label_sent_token_spans(sent_token_spans: List[List[TokenSpan]],
                           mention_spans: List[MentionSpan]) -> List[List[str]]:
    """
    `sent_token_spans` is a list of sentences, where each sentence is a list of token spans
    `mention_spans` is a single list of mention spans

    Assumes both of these are properly sorted (sentences & tokens) & disjoint in their tokens.

    Returns BIO labels for each sentence, for each token, matching structure of `sent_token_spans`
    """
    assert _is_proper_sents(sent_token_spans)

    # align mention spans with sentences
    sent_mention_spans = _match_mention_spans_to_sentences(sent_token_spans, mention_spans)
    assert _is_proper_sents([s for s in sent_mention_spans if len(s) > 0])

    # create labels
    sent_token_labels = []
    for token_spans, mention_spans in zip(sent_token_spans, sent_mention_spans):
        token_labels = _label_token_spans(token_spans, mention_spans)
        sent_token_labels.append(token_labels)
    return sent_token_labels


def _is_proper_sents(sent_spans: List[List[Span]]) -> bool:
    # order of sentences
    for i in range(len(sent_spans) - 1):
        if not sent_spans[i][-1] < sent_spans[i + 1][0]:
            return False
    # proper tokens within sentences
    for token_spans in sent_spans:
        if not _is_proper_sent(token_spans):
            return False
    return True


def _is_proper_sent(spans: List[Span]) -> bool:
    # check for sorted & disjoint tokens
    return all(spans[i] < spans[i + 1] for i in range(len(spans) - 1))


def _label_token_spans(token_spans: List[TokenSpan],
                       mention_spans: List[MentionSpan]) -> List[str]:
    """Private method to process each sentence in `label_sent_token_spans()`"""
    num_tokens, num_mentions = len(token_spans), len(mention_spans)
    # no tokens
    if num_tokens == 0:
        return []

    # no mentions
    if num_mentions == 0:
        return ['O'] * num_tokens

    # check mentions should be within range of tokens
    assert mention_spans[0].start >= token_spans[0].start
    assert mention_spans[-1].stop <= token_spans[-1].stop

    token_labels = []
    index_token, index_mention = 0, 0
    while index_token < num_tokens and index_mention < num_mentions:
        token_span = token_spans[index_token]
        mention_span = mention_spans[index_mention]
        entity_type = mention_span.entity_types[0]
        # case 1: token is left of mention (no overlap)
        if token_span < mention_span:
            token_labels.append('O')
            index_token += 1
        # case 2: token is right of mention (no overlap)
        elif token_span > mention_span:
            index_mention += 1
        # case 3: token captures start of mention
        elif token_span.start <= mention_span.start:
            token_labels.append(f'B-{entity_type}')
            index_token += 1
        # case 4: token within mention
        elif token_span in mention_span:
            token_labels.append(f'I-{entity_type}')
            index_token += 1
        # case 5: token captures end of mention
        elif token_span.stop >= mention_span.stop:
            token_labels.append(f'I-{entity_type}')
            index_token += 1
            index_mention += 1

    # ran out of mentions, but label remaining tokens
    while index_token < num_tokens:
        token_labels.append('O')
        index_token += 1

    assert len(token_labels) == len(token_spans)
    return token_labels


def _match_mention_spans_to_sentences(sent_token_spans: List[List[TokenSpan]],
                                      mention_spans: List[MentionSpan]) -> List[List[MentionSpan]]:
    """Private method to process `mention_spans` into sentences in `label_sent_token_spans()`"""
    num_sents, num_mentions = len(sent_token_spans), len(mention_spans)

    # check mentions should all be match-able to sentences
    assert mention_spans[0].start >= sent_token_spans[0][0].start
    assert mention_spans[-1].stop <= sent_token_spans[-1][-1].stop

    sent_mention_spans = []
    temp = []
    index_sent, index_mention = 0, 0
    while index_sent < num_sents - 1 and index_mention < num_mentions:
        mention_span = mention_spans[index_mention]
        this_sent_start = sent_token_spans[index_sent][0].start
        this_sent_stop = sent_token_spans[index_sent][-1].stop
        next_sent_start = sent_token_spans[index_sent + 1][0].start
        # if mention within this sentence, keep it
        if mention_span.start >= this_sent_start and mention_span.stop <= this_sent_stop:
            temp.append(mention_span)
            index_mention += 1
        # if cross-sentence mention, skip it
        elif mention_span.start < this_sent_stop and mention_span.stop > next_sent_start:
            print(f'Mention {mention_span} crosses sentence boundary')
            index_mention += 1
        # if mention not within this sentence, go to next sentence
        else:
            sent_mention_spans.append(temp)
            temp = []
            index_sent += 1

    # previous loop should conclude either:
    #   (1) sentence n-2 with mentions remaining.
    #   (2) earlier sentence but with no mentions remaining.

    # (1) handle final sentence's mentions
    while index_mention < num_mentions:
        mention_span = mention_spans[index_mention]
        temp.append(mention_span)
        index_mention += 1
    sent_mention_spans.append(temp)
    index_sent += 1

    # (2) handle remaining sentences without mentions
    while index_sent < num_sents:
        sent_mention_spans.append([])
        index_sent += 1

    assert len(sent_mention_spans) == len(sent_token_spans)
    return sent_mention_spans
