class SentenceReadingAgent:
    def __init__(self):
        # The 20 names from the assignment
        self.names = {
            "Serena", "Andrew", "Bobbie", "Cason", "David", "Farzana", "Frank",
            "Hannah", "Ida", "Irene", "Jim", "Jose", "Keith", "Laura", "Lucy",
            "Meredith", "Nick", "Ada", "Yeeling", "Yan",
        }

        # Usually the word i want is right after a preposition
        self.prepositions = {
            "to", "from", "with", "in", "on", "at", "by", "for", "of", "into",
            "near", "over", "under", "behind", "above", "below", "between",
            "through", "inside", "beside", "around",
        }

        # Articles and "my/your/his" all work the same when I hunt for the noun
        self.articles = {"a", "an", "the", "some", "this", "that", "these",
                         "those", "my", "your", "his", "her", "its", "our",
                         "their"}

        self.colors = {
            "red", "orange", "yellow", "green", "blue", "purple", "pink",
            "black", "white", "brown", "gray", "gold",
        }

        self.number_words = {
            "one", "two", "three", "four", "five", "six", "seven", "eight",
            "nine", "ten", "eleven", "twelve", "hundred", "thousand",
        }

        self.distance_units = {"mile", "miles", "foot", "feet", "inch", "inches",
                               "yard", "yards", "block", "blocks"}
        self.time_units = {"second", "seconds", "minute", "minutes", "hour",
                           "hours", "day", "days", "week", "weeks", "month",
                           "months", "year", "years", "morning", "night"}
        self.weight_units = {"pound", "pounds", "ounce", "ounces", "ton", "tons"}

        # If the sentence has one of these, "where" usually means destination
        self.motion_verbs = {"go", "goes", "went", "walk", "walks", "walked",
                             "run", "runs", "ran", "drive", "drives", "drove",
                             "ride", "rides", "rode", "travel", "come", "comes",
                             "came", "return", "returns", "returned"}

        self.days = {"monday", "tuesday", "wednesday", "thursday", "friday",
                     "saturday", "sunday", "today", "tomorrow", "yesterday"}
        self.months = {"january", "february", "march", "april", "may", "june",
                       "july", "august", "september", "october", "november",
                       "december"}

        # Question says "bring" but sentence says "brought"
        self.verb_forms = {
            "bring": {"bring", "brings", "brought"},
            "give": {"give", "gives", "gave"},
            "take": {"take", "takes", "took"},
            "buy": {"buy", "buys", "bought"},
            "write": {"write", "writes", "wrote"},
            "eat": {"eat", "eats", "ate"},
            "see": {"see", "sees", "saw"},
            "find": {"find", "finds", "found"},
            "get": {"get", "gets", "got"},
            "drive": {"drive", "drives", "drove"},
            "ride": {"ride", "rides", "rode"},
            "paint": {"paint", "paints", "painted"},
            "plant": {"plant", "plants", "planted"},
            "carry": {"carry", "carries", "carried"},
            "have": {"have", "has", "had"},
        }

        # Adjectives I skip to get to the noun, also used for size questions
        self.size_adjectives = {"long", "short", "tall", "big", "small", "large",
                                "tiny", "little", "wide", "narrow", "old", "new"}

        # Other describing words I skip when looking for the noun
        self.skip_adjectives = {"other", "same", "next", "last", "first", "good",
                                "bad", "hot", "cold", "warm", "fast", "slow"}

    def tokenize(self, text):
        # Keep ' and : so "don't" and "9:00AM" stay as one word
        tokens = []
        word = ""
        for ch in text:
            if ch.isalnum() or ch in ("'", ":"):
                word += ch
            else:
                if word:
                    tokens.append(word)
                    word = ""
        if word:
            tokens.append(word)
        return tokens

    def is_time(self, token):
        # Looks like 9:00AM / 11:00 / 12:34PM
        core = token.lower()
        for suffix in ("am", "pm"):
            if core.endswith(suffix):
                core = core[:-2]
                break
        if ":" not in core:
            return False
        parts = core.split(":")
        return len(parts) == 2 and parts[0].isdigit() and parts[1].isdigit()

    def is_name(self, token):
        bare = token.strip(".,!?")
        return bare in self.names or bare.capitalize() in self.names

    def solve(self, sentence, question):
        s_tokens = self.tokenize(sentence)
        q_lower = [t.lower() for t in self.tokenize(question)]
        q_set = set(q_lower)
        s_lower = [t.lower() for t in s_tokens]
        first = q_lower[0] if q_lower else ""

        # Who / whom / whose -> a person
        if first in ("who", "whom") or "whose" in q_set:
            return self.find_person(s_tokens, s_lower, q_lower)

        # When / what time / what day -> a time
        if first == "when" or "time" in q_set or \
           (first == "what" and ("day" in q_set or "month" in q_set or "year" in q_set)):
            ans = self.find_time(s_tokens, s_lower)
            if ans:
                return ans

        # Where -> a place
        if "where" in q_set:
            ans = self.find_place(s_tokens, s_lower, q_set)
            if ans:
                return ans

        if first == "how":
            head = q_lower[:3]
            # How many / how much -> a number
            if "many" in head or "much" in head:
                for i, t in enumerate(s_lower):
                    if t.isdigit() or t in self.number_words:
                        return s_tokens[i].strip(".,!?")
            if "far" in head:
                ans = self.find_measure(s_tokens, s_lower, self.distance_units)
                if ans:
                    return ans
            if "old" in head:
                ans = self.find_measure(s_tokens, s_lower, {"year", "years"})
                if ans:
                    return ans
            # "how long" could be time or size
            if "long" in head:
                ans = self.find_long(s_tokens, s_lower)
                if ans:
                    return ans
            if "heavy" in head:
                ans = self.find_measure(s_tokens, s_lower, self.weight_units)
                if ans:
                    return ans
            # "how big / tall / wide" -> a size word in the sentence
            if "big" in head or "tall" in head or "wide" in head or \
               "small" in head or "deep" in head:
                for i, t in enumerate(s_lower):
                    if t in self.size_adjectives:
                        return s_tokens[i].strip(".,!?")
            # Plain "how do they get there" -> the verb they use
            for i, t in enumerate(s_lower):
                if t in self.motion_verbs:
                    return s_tokens[i].strip(".,!?")

        # "What is X's name" -> a known name, else a capitalized word like "Red"
        if first == "what" and "name" in q_set:
            for t in s_tokens:
                if self.is_name(t):
                    return t.strip(".,!?")
            for i, t in enumerate(s_tokens):
                bare = t.strip(".,!?")
                if i > 0 and bare[:1].isupper():
                    return bare

        # What color -> the color word
        if first == "what" and "color" in q_set:
            # If the question names a noun, use the color next to that noun
            noun = None
            for t in q_lower:
                if t in s_lower and t not in self.articles and t != "color":
                    noun = t
            if noun and noun in s_lower:
                idx = s_lower.index(noun)
                for i in range(idx - 1, -1, -1):
                    if s_lower[i] in self.colors:
                        return s_tokens[i]
            # Otherwise just the first color word
            for i, t in enumerate(s_lower):
                if t in self.colors:
                    return s_tokens[i]

        # What / which -> the object
        if first in ("what", "which"):
            ans = self.find_object(s_tokens, s_lower, q_lower)
            if ans:
                return ans

        # Nothing matched, make a guess
        return self.guess(s_tokens, s_lower, q_lower)

    def names_in(self, s_tokens):
        return [t.strip(".,!?") for t in s_tokens if self.is_name(t)]

    def find_person(self, s_tokens, s_lower, q_lower):
        q_set = set(q_lower)

        # Whose -> the name right before the 's
        if "whose" in q_set:
            for t in s_tokens:
                bare = t.strip(".,!?")
                if bare.endswith("'s") and self.is_name(bare[:-2]):
                    return bare[:-2]

        people = self.names_in(s_tokens)
        if not people:
            return self.guess(s_tokens, s_lower, q_lower)

        # "who ... with" -> the person after "with"
        if "with" in q_set:
            comp = self.name_after(s_tokens, s_lower, "with")
            if comp:
                return comp
            return self.other_name(people, q_lower)

        # "who did X give it to" -> the person after "to"
        if "to" in q_set:
            recip = self.name_after(s_tokens, s_lower, "to")
            if recip:
                return recip
            return self.other_name(people, q_lower)

        # Normal case, the first name is the subject
        return people[0]

    def name_after(self, s_tokens, s_lower, prep):
        for i, t in enumerate(s_lower):
            if t == prep:
                for j in range(i + 1, len(s_tokens)):
                    if self.is_name(s_tokens[j]):
                        return s_tokens[j].strip(".,!?")
        return None

    def other_name(self, people, q_lower):
        # The companion is the name the question did not already say
        q_names = {t.capitalize() for t in q_lower}
        for name in people:
            if name not in q_names:
                return name
        return people[0]

    def find_time(self, s_tokens, s_lower):
        # A clock time is the best answer
        for t in s_tokens:
            if self.is_time(t):
                return t.strip(".,!?")
        for i, t in enumerate(s_lower):
            if t in self.days or t in self.months:
                return s_tokens[i].strip(".,!?")
        # A time word right after at / on / in
        for i, t in enumerate(s_lower):
            if t in ("at", "on", "in") and i + 1 < len(s_tokens):
                nxt = s_lower[i + 1]
                if nxt in self.time_units or nxt.isdigit():
                    return s_tokens[i + 1].strip(".,!?")
        for i, t in enumerate(s_lower):
            if t in self.time_units:
                return s_tokens[i].strip(".,!?")
        return None

    def find_place(self, s_tokens, s_lower, q_set):
        # "where from" -> where they started
        if "from" in q_set:
            loc = self.noun_after(s_tokens, s_lower, "from")
            if loc:
                return loc
        # If they are moving, the place after "to" is where they went
        loc_to = self.noun_after(s_tokens, s_lower, "to")
        moving = any(v in s_lower for v in self.motion_verbs)
        if ("to" in q_set or moving) and loc_to:
            return loc_to
        # Otherwise look after a normal place word
        for prep in ("in", "on", "at", "near", "inside", "behind", "under",
                     "over", "above", "below", "beside", "between"):
            loc = self.noun_after(s_tokens, s_lower, prep)
            if loc:
                return loc
        if loc_to:
            return loc_to
        return None

    def noun_after(self, s_tokens, s_lower, prep):
        for i, t in enumerate(s_lower):
            if t == prep:
                j = i + 1
                # "to go to school" -> skip the verb and jump to the next "to"
                while j < len(s_tokens) and (s_lower[j] in self.articles or
                                             s_lower[j] in self.motion_verbs):
                    if s_lower[j] in self.motion_verbs:
                        nxt = None
                        for k in range(j + 1, len(s_tokens)):
                            if s_lower[k] == prep:
                                nxt = k
                                break
                        if nxt is not None:
                            j = nxt + 1
                            continue
                    j += 1
                if j < len(s_tokens) and s_lower[j] not in self.prepositions:
                    # Skip adjectives like "other" to reach the real noun
                    return self.head_noun(s_tokens, s_lower, j)
        return None

    def find_measure(self, s_tokens, s_lower, units):
        for i, t in enumerate(s_lower):
            if t in units:
                # Grab the number in front of it too if there is one
                if i > 0 and (s_lower[i - 1].isdigit() or
                              s_lower[i - 1] in self.number_words):
                    return s_tokens[i - 1] + " " + s_tokens[i].strip(".,!?")
                return s_tokens[i].strip(".,!?")
        for i, t in enumerate(s_lower):
            if t.isdigit() or t in self.number_words:
                return s_tokens[i].strip(".,!?")
        return None

    def find_long(self, s_tokens, s_lower):
        # Try a time word first
        for i, t in enumerate(s_lower):
            if t in self.time_units:
                if i > 0 and (s_lower[i - 1].isdigit() or
                              s_lower[i - 1] in self.number_words):
                    return s_tokens[i - 1] + " " + s_tokens[i].strip(".,!?")
                return s_tokens[i].strip(".,!?")
        dist = self.find_measure(s_tokens, s_lower, self.distance_units)
        if dist:
            return dist
        # If no distance, "how long" is asking about size like "short"
        for i, t in enumerate(s_lower):
            if t in self.size_adjectives:
                return s_tokens[i].strip(".,!?")
        return None

    def find_object(self, s_tokens, s_lower, q_lower):
        # Content words of the question, minus names and filler words
        q_content = [t for t in q_lower
                     if t not in self.articles and
                     t.capitalize() not in self.names and
                     t not in ("what", "which", "did", "does", "do", "is",
                               "are", "was", "were", "to", "of", "with", "for")]

        # Find the verb in the sentence, the object comes right after it
        verb_idx = None
        for qc in q_content:
            forms = self.verb_forms.get(qc)
            if forms:
                for i, t in enumerate(s_lower):
                    if t in forms:
                        verb_idx = i
                        break
            if verb_idx is not None:
                break
        # Verb not in my table, so just look for the same word (not a preposition)
        if verb_idx is None:
            for qc in q_content:
                if qc in s_lower and qc not in self.prepositions:
                    verb_idx = s_lower.index(qc)
                    break

        if verb_idx is not None:
            obj = self.head_noun(s_tokens, s_lower, verb_idx + 1)
            if obj:
                return obj

        # No verb match, guess the object as the noun after the subject
        start = 0
        for i, t in enumerate(s_tokens):
            if self.is_name(t):
                start = i + 1
                break
        return self.head_noun(s_tokens, s_lower, start + 1)

    def head_noun(self, s_tokens, s_lower, start):
        # Skip "a" / "the" / adjectives to get to the noun
        j = start
        last = None
        while j < len(s_tokens):
            low = s_lower[j]
            if low in self.articles:
                j += 1
                continue
            if low in self.prepositions or self.is_name(s_tokens[j]):
                break
            w = s_tokens[j].strip(".,!?")
            if not w:
                break
            last = w
            if low in self.size_adjectives or low in self.colors or \
               low in self.skip_adjectives:
                j += 1
                continue
            # "top of the mountain" -> keep going past "of" to the real noun
            if j + 1 < len(s_tokens) and s_lower[j + 1] == "of":
                j += 2
                continue
            return w
        return last

    def guess(self, s_tokens, s_lower, q_lower):
        # Give back a sentence word the question did not use
        q_set = set(q_lower)
        stop = self.articles | self.prepositions | {
            "is", "are", "was", "were", "and", "or", "but", "do", "does",
            "did", "no", "not", "there", "when", "every",
        }
        for i, t in enumerate(s_lower):
            if t not in q_set and t not in stop and not self.is_name(s_tokens[i]):
                return s_tokens[i].strip(".,!?")
        for t in s_tokens:
            if self.is_name(t):
                return t.strip(".,!?")
        return s_tokens[0].strip(".,!?") if s_tokens else ""
