import math

class Retrieve:
    
    # Create new Retrieve object storing index and term weighting 
    # scheme. (You can extend this method, as required.)
    def __init__(self,index, term_weighting):
        self.index = index
        self.term_weighting = term_weighting
        self.num_docs = self.compute_number_of_documents()
       
             
    def compute_number_of_documents(self):
        self.doc_ids = set()
        for term in self.index:
            self.doc_ids.update(self.index[term])
        return len(self.doc_ids)

    # creates a 2d array of terms(present in query)*document(term_document_map) for following cases:
    # binary: stores 1 if word is present else 0
    # tf: stores the frequency of the term in that document else 0
    # tfidf: stores the tf.idf value for the term
    # term_idf_map is a dictionary which maintains term as key and idf as value 
    def compute_term_weights(self, term, document_frequency, row):        
        
        if(self.term_weighting == 'tfidf'):                
            self.term_idf_map.update({term: math.log10(self.num_docs/ len(document_frequency.keys()))})
        
        for document in document_frequency.keys():
            if(self.term_weighting == 'binary'):
                self.term_document_map[row][document] = 1
            elif(self.term_weighting == 'tf'):
                self.term_document_map[row][document] = document_frequency[document]
            else:
                self.term_document_map[row][document] = document_frequency[document] * self.term_idf_map.get(term)
    
    # computes the query vector for all cases
    def compute_query_vector(self, term, query):
        query_word_freq = query.count(term)
                
        if(self.term_weighting == 'tfidf'):
            self.query_vector.append(self.term_idf_map.get(term) * query_word_freq)              
        elif(self.term_weighting == 'tf'):
            self.query_vector.append(query_word_freq)
        else:
            self.query_vector.append(1)

    # sorts the cosine scores is descending order and fetches the top 10 documents based on score        
    def fetch_top_matching_docs(self, docid_score):
        self.sorted_doclist = sorted(docid_score, key=docid_score.get, reverse=True)[:10]
        return self.sorted_doclist
        
    # calculates the cosine score with the help of term_document_map for all the words in the query
    def compute_cosine_score(self, query):       
        docid_score = {}
        
        unique_query_words = list(dict.fromkeys(query))
        self.term_document_map = [[ 0 for x in range(self.num_docs + 1)] for y in range(len(unique_query_words))]   
        
        if(self.term_weighting == 'tfidf'):
            self.term_idf_map = {}
          
        self.query_vector = []
        for row, term in enumerate(unique_query_words):
            if term in self.index: 
                document_frequency = self.index[term]
                self.compute_term_weights(term, document_frequency, row)                
                self.compute_query_vector(term, query)
     
               
        for docid, col in enumerate(zip(*(self.term_document_map))):
            if(docid!=0 and sum(col)> 0):                
                numerator = sum(x * y for x, y in zip(self.query_vector, col))
                denominator = math.sqrt(sum(map(lambda x:x*x,col)))          
                docid_score.update({docid: (numerator/denominator)})
                
        return self.fetch_top_matching_docs(docid_score)
    
  

    # Method performing retrieval for a single query (which is 
    # represented as a list of preprocessed terms). Returns list 
    # of doc ids for relevant docs (in rank order).
    def for_query(self, query):
      
        docid_list = self.compute_cosine_score(query)
        
        return docid_list
