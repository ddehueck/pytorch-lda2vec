import re
import spacy
from spacy.util import filter_spans

class Tokenizer:
    
    def __init__(self, custom_stop=set(), merge_noun_chunks=False):
        self.custom_stop = custom_stop
        # Define pipeline - use different nlp is using pretrained
        self.nlp = spacy.load("en_core_web_sm", disable=[])
        # Use for unique vocabularies - i.e. Hacker News Experiement
        if merge_noun_chunks:
            self.nlp.add_pipe(self.moodys_merge_noun_chunks, name="mnc", last=True)


    def tokenize_doc(self, doc_str):
        """
        Tokenize a document string
        Modified version of Moody's Tokenization in:
        https://github.com/cemoody/lda2vec/blob/master/lda2vec/preprocess.py

        :params doc_str: String
        :returns: list of Strings, i.e. tokens
        """

        """# Send doc_str through pipeline
        spacy_doc = self.nlp(doc_str)
        # Filter 
        filtered_doc = filter(self.is_valid_token, spacy_doc)
        # Convert to text make lowercase
        clean_doc = [t.text.lower() for t in filtered_doc]
        # Only allow characters in the alphabet and '_'
        clean_doc = [re.sub('[^a-zA-Z]', '', t) for t in clean_doc]
        # Remove any resulting empty indices
        clean_doc = [t for t in clean_doc if len(t) > 2]
        # Filter out any custom stop
        clean_doc = [t for t in clean_doc if t not in self.custom_stop]
        return clean_doc
        """
        text = ' '.join(doc_str.split())  # remove excessive spaces
        text = self.nlp(text, disable=['parse', 'entity'])
        return [t.lemma_ for t in text
                if t.is_alpha and len(t) > 2 and not t.is_stop]


    def is_valid_token(self, token):
        """
        Determines if a token is valid or not

        :params token: String
        :returns: Boolean
        """
        if token.like_url:
            return False
        if token.like_email:
            return False
        if token.is_stop:
            return False
        if token.text in self.custom_stop:
            return False

        return True


    @staticmethod
    def moodys_merge_noun_chunks(doc):
        """
        Merge noun chunks into a single token.
        
        Modified from sources of:
        - https://github.com/cemoody/lda2vec/blob/master/lda2vec/preprocess.py
        - https://spacy.io/api/pipeline-functions#merge_noun_chunks
        
        :params doc: Doc object.
        :returns: Doc object with merged noun chunks.
        """
        bad_deps = ('amod', 'compound')
        
        if not doc.is_parsed:
            return doc
        with doc.retokenize() as retokenizer:
            for np in filter_spans(list(doc.noun_chunks)):
                
                # Only keep adjectives and nouns, e.g. "good ideas"
                while len(np) > 1 and np[0].dep_ not in bad_deps:
                    np = np[1:]
                
                if len(np) > 1:
                    # Merge NPs
                    attrs = {"tag": np.root.tag, "dep": np.root.dep}
                    retokenizer.merge(np, attrs=attrs)
                
        return doc