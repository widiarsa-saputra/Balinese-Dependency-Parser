### UPPOS
this is the base Part of Speech tag that used in Indonesian Language, but also commonly used in Balinese language.
- Proper Noun (propn) a word that usually refers a person name, location name, etc
- Preposition (adp) a word that is usually used to show the relationship between other words in a sentence, such as place, time, direction, or manner
- Noun (noun) a word that usually refers to a person, place, thing, or concept
- Verb (verb) typically action words or state words that serve as the predicate of a sentence, expressing actions, events, or states of being.
- Adverb (adv) typically express how, when, where, or to what degree an action or state occurs.
- Punctuation (punct) is used to mark punctuation marks in the sentence, such as commas, periods, question marks, exclamation marks, quotation marks, etc
- Deteminers (det) are words placed before nouns to indicate which or how many things are being referred to
- Numerals (num) can be cardinal (e.g., "one", "two") or ordinal (e.g., "first", "second"), and they typically modify nouns to indicate quantity or order
- Adjective (adj) typically describe or modify nouns, providing more information about their attributes, such as size, color, shape, etc.
- Pronouns (pron) typically refer to people or things mentioned earlier in the conversation or text

### XPOS
this is the base Part of Speech tag that used for treebank specific.
- Singular Proper Noun (nnp) are specific names of people, places, or things
- Singular or Mass Noun (nn) are general nouns that represent people, places, things, or concepts, but unlike proper nouns
- Preposition (in) are words that show relationships between nouns and other elements in a sentence
- Verb (vb) for base-form verbs
- Adverb (rb) is used for adverbs, which modify verbs, adjectives, or other adverbs to provide more information about the action, state, or quality.
- Determiner (dt) are words placed before nouns to indicate which or how many things are being referred to
- Modal Verb (md) is used for modal verbs, which express necessity, possibility, ability, permission, or obligation
- Personal Pronoun (prp) is used for personal pronouns, which refer to specific people or things and are used in place of a noun
- Adjective (JJ) is sometimes used as a specific form of the ADJ tag in the context

### FEATS
- Polarity=Neg is used to mark words that have a negative polarity, typically indicating negation
- Poss=Yes indicates that the word or token shows possession
- Voice=Passive is used to indicate that the verb is in the passive voice
- Degree=Neg is used to indicate that an adjective or adverb has a negative degree. This could be used for comparative or superlative forms with a negating meaning
- PronType=Art  marks a pronoun that functions as an article. This can refer to a word that modifies a noun and indicates definiteness or indefiniteness
- PronType=Dem marks a demonstrative pronoun. Demonstrative pronouns indicate specific items or people, usually in relation to their distance

### DEPREL
- Nominal Subject (nsubj) is typically the noun or noun phrase that performs or is involved in the action of the verb
- Object (obj) is typically the noun or noun phrase that receives the action of the verb
- Indirect Object (iobj) usually refers to the entity that benefits from or is affected by the action, typically introduced by prepositions
- Adverbial Modifier (advmod) describes how, when, where, or to what extent an action is performed
- Adjectival Modifier (amod) relation links an adjective or adjective phrase to a noun that it modifies
- Nominal Modifier (nmod) is a noun or noun phrase that modifies another noun. This often represents relationships like possession or description
- Clausal Complement (ccomp) refers to a clause that functions as the direct object or complement of a verb, often in sentences with verbs
- Open Clausal Complement (xcomp) is similar to a clausal complement (ccomp), but it typically lacks a subject and can be more "open-ended" (like infinitival clauses)
- Root of the Sentence (root) the central word or head of the sentence structure
- Punctuation (punct) Marks punctuation marks in the sentence, such as commas, periods, question marks, etc
- Determiner (dt) Marks a determiner, which is a word that introduces a noun and gives more information about its reference
- Appositional Modifier (appos) is a noun or noun phrase that provides additional information about another noun it is next to
- Compound (compound) that consist of two or more words that function together as a single unit, often used in noun phrases or multi-word expressions
- Case prepotition (case) is used to specify the grammatical case of a noun or pronoun. Grammatical case indicates the syntactic role a noun or pronoun plays in a sentence, such as subject, object, or possession

We refers all the tag that we used by translating the balinese form to indonesian form, helps by [online balinese dictionary](https://www.kamusbali.net/). For each word UPPOS tag we get the appropriate tags from [online indonesian dictionary](https://kbbi.web.id/).

For Dependency Relations tagging purpose we define set of rule to tags the data, such as:
1. nsubj - is noun/propn/pron that used for subject of the root(head), there's a chance that nsubj is used more than one for multiclausal sentence case, the other nsubj head's would be the verb of the clausal
2. obj - is noun/propn/pron that used for object of the root(head), there's a chance that obj is used more than one for multiclausal sentence case, the other obj head's would be the verb of the clausal
3. iobj - is noun/propn/pron that used for Indirect Object that usually get benefit from the object, the head would be the action form (verb) on the clausal
4. advmod - is adverb that used for describes the adverbial for the action/nomina form on the clausal
5. amod - is adjective that used for describes the characters, nature, etc for the noun. Usually modify the nsubj or obj as head
6. nmod - is noun/propn/pron that used for describes the extra information of the action form like time, situations, or place, etc.
7. ccomp - is verb that used on the other clausal that has different subject, give extra action information for the action form (head)
8. xcomp - is verb that used on the other clausal that has same subject on the previous clausal, give extra action information for the action form (head)
9. root - is the root of the sentences or the main action of the sentences, it can be verb(usually), noun, or adjective.
10. punct - is punct that used as the punctuation of the clausal. the used is depends for the punctuations type, such as:
    - ( . ) would be the end of the sentence, the head is the main action
    - ( , ) would be the connector of sentences to the next clausal, the head is the form before this punctuations
    - ( " ) would be the open/close dialog insides sentence, the head is the next form of the first quotation mark
    - ( ! ) would be the action amplifier of the clausal, the head is the form before this punctuations
11. det - is det that used to complete the other form, the head is depends on the PronTypes
    - PronType=Dem, the head would be the form before this det
    - PronType=Art, the head would be the form after this det
13. appos - is noun/propn/pron that used to give extra information, the head is the noun/propn/pron before/after that refers to this information
14. compound - is noun/propn/adp/det/num that used to complete the information of the noun phrase, the head is the noun/propn after this tag
15. case - is adp that used to grammatical function, the head would be the form after this tag


credits: https://universaldependencies.org/u/dep/index.html
