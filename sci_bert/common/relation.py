"""

Basic data structure and functions for relation mentions

author: kylel@allenai.org

"""

from typing import *

from sci_bert.common.span import MentionSpan, TokenSpan

class RelationMention:
    """
    A relation mention corresponds to an (ordered) tuple of entity mentions
    with a label(s) and a direction implied by the order of the tuple.
    """
    def __init__(self, e1: MentionSpan, e2: MentionSpan, labels: List[str],
                 is_symmetric: Optional[bool] = None):
        self.e1 = e1
        self.e2 = e2
        self.labels = labels
        self.is_symmetric = is_symmetric

    def __repr__(self):
        return str((self.e1.text, tuple(self.labels), self.e2.text))
