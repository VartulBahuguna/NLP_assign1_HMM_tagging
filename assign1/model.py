import numpy as np
#splits data in k sets:
def k_splits (k, data):
    splits = {}
    n = len(data)//k
    for i in range(0, k):
        i = int(i)
        if(i+1 == k):
            splits[i]= data[n*i : ]
        else:
            splits[i]=data[n*i : n*(i+1)]
    return splits
def add_start_end_tag(sentences_original):
    sentences=[""]*len(sentences_original)
    for i in range(len(sentences_original)):
        sentences[i]=[("<start>","<start>")]+sentences_original[i]+[("<end>","<end>")]
    return sentences
def remove_tags(sentence):
    new_sentence=[]
    for i in sentence:
        new_sentence.append(i[0])

    # print(new_sentence)

    return new_sentence
class HMM_model:
    def __init__(self):
        self.confusion_matrix=None
        self.transition_matrix=None
        self.emission_matrix=None
        self.tags_to_idx=None
        self.count_of_each_tag=None
        self.idx_to_tag=None
        self.word_to_idx=None
        self.idx_to_word=None
    
    def creat_tags_meta(self,sentences):
        
        #tag_to_idx: given a tag, what is its index : tags dict {"tag" : "tag_index"}
        #idx_to_tag: given an index what is the tag : tags_reverse list of string
        #count_of_each_tag: count of each tag
        self.tag_to_idx={}
        self.idx_to_tag=[]
        cnt = 0
        self.count_of_each_tag=[]
        for i in range(len(sentences)):
            for j in sentences[i]:
                if j[1] not in self.tag_to_idx:
                    self.tag_to_idx[j[1]] = cnt
                    self.idx_to_tag.append(j[1])
                    self.count_of_each_tag.append(0)
                    cnt += 1
                self.count_of_each_tag[self.tag_to_idx[j[1]]]+=1


    def create_word_idx_translation(self,sentences):
        #word to index number
        #word_to_idx is word to index number
        #idx_to_word
        
        idx = 0
        self.word_to_idx={}
        self.idx_to_word = []
        # word_to_idx["<start>"]=idx
        for sentence in sentences:
            for tup in sentence:
                if tup[0] not in self.word_to_idx:
                    
                    self.word_to_idx[tup[0]]=idx
                    idx += 1
                    self.idx_to_word.append(tup[0])
        self.word_to_idx["<unknown>"]=idx
        self.idx_to_word.append("<unknown>")

        # print('idx--------------------------------', idx)
        # print('worddict--------------------------------', len(word_to_idx))


    def create_transition_matrix(self,sentences):
        #creating transition matrix
        #bigram_matrix : count of every bigram tags
        #transmission_matrix : probability of every bigram tags 
        self.bigram_matrix = []
        self.transition_matrix = []
        total_tags = len(self.count_of_each_tag)
        for i in range(total_tags): 
            self.bigram_matrix.append([0]*total_tags)
            self.transition_matrix.append([0]*total_tags)

        #creating a bigram matrix
        for sentence in sentences:
            for j in range(len(sentence)-1):
                self.bigram_matrix[self.tag_to_idx[sentence[j][1]]][self.tag_to_idx[sentence[j+1][1]]]+=1
        # print(bigram_matrix)
        
        #create transition matrix A
        for i in range(total_tags):
            for j in range(total_tags):
                self.transition_matrix[i][j]=(self.bigram_matrix[i][j]+1)/(self.count_of_each_tag[i] + len(self.count_of_each_tag))  #adding 1 to numerator and total unique tags in denominator for smoothing(Laplace)
        # print(transition_matrix)


    def create_emission_matrix(self,sentences):
        #emission_matrix : given a word, what is the probability of tag
        #emission_count_matrix : given a word, what is the count of each tag that it can have
        # words = list(brown.words())
        words_cnt = len(self.word_to_idx)
        total_tags = len(self.tag_to_idx)
        self.emission_matrix = []
        
        for i in range(words_cnt):
            self.emission_matrix.append([0]*total_tags)
        # print("--------------------------------------",len(word_to_idx))
        # print("------------------------", len(emission_matrix))
        # print("------------------------", word_to_idx['<unknown>'])
        
        for sentence in sentences:
            for j in sentence:
                self.emission_matrix[self.word_to_idx[j[0]]][self.tag_to_idx[j[1]]]+=1

        emission_count_matrix = self.emission_matrix

        for i in range(words_cnt):
            for j in range(total_tags):
                self.emission_matrix[i][j] = (self.emission_matrix[i][j]+1)/(self.count_of_each_tag[j] + words_cnt)
        # print("------------------------", emission_matrix[word_to_idx['unknown'])

        # handling missing case
        aux=[0]*len(self.tag_to_idx.keys())
        # print(tag_to_idx)
        aux[self.tag_to_idx['NOUN']]=self.count_of_each_tag[self.tag_to_idx['NOUN']]/total_tags
        aux[self.tag_to_idx['VERB']]=self.count_of_each_tag[self.tag_to_idx['VERB']]/total_tags
        aux[self.tag_to_idx['ADV']]=self.count_of_each_tag[self.tag_to_idx['ADV']]/total_tags
        aux[self.tag_to_idx['ADJ']]=self.count_of_each_tag[self.tag_to_idx['ADJ']]/total_tags
        # print(aux)
        self.emission_matrix[self.word_to_idx['<unknown>']]=aux


    def build_confusion_matrix(self,actual,predicted):
        #cm: confusion matrix
        for i in range(len(actual)):
            self.confusion_matrix[self.tag_to_idx[actual[i][1]]][self.tag_to_idx[predicted[i]]]+=1

    def HMM_logic(self,input_sentence):
        # print(tags_dict)
        tags_output=["<start>"]
        last_prob = 1
        viterbi=[]
        for i in range(len(self.tag_to_idx)):
            viterbi.append(([0])*len(input_sentence))
        
        viterbi[self.tag_to_idx["<start>"]][0]=1
        
        for i in range(1, len(input_sentence)):
            word_prob=0
            tags=""
            curr_ob = input_sentence[i]
            if curr_ob not in self.word_to_idx:
                curr_ob="<unknown>"
            for curr_tag in self.tag_to_idx:
                max_prob=0
                for prev_tag in self.tag_to_idx:
                    temp = self.transition_matrix[self.tag_to_idx[prev_tag]][self.tag_to_idx[curr_tag]]*self.emission_matrix[self.word_to_idx[curr_ob]][self.tag_to_idx[curr_tag]]*viterbi[self.tag_to_idx[prev_tag]][i-1]
                    if temp>max_prob:
                        max_prob=temp
                viterbi[self.tag_to_idx[curr_tag]][i]=max_prob
                if max_prob>word_prob:
                    word_prob=max_prob
                    tags=curr_tag
            if tags=='':
                tags='X'
            tags_output.append(tags)

        return [(input_sentence[i],tags_output[i]) for i in range(len(input_sentence))] 
    



    def train_model(self,sentences_original, k):

        model=HMM_model()

        sentences = add_start_end_tag(sentences_original)
        
        #spliting data into k splits
        cross_validation_set=k_splits(k,sentences)
        keys=list(cross_validation_set.keys())

        #creating master meta data i.e. over whole corpus
        self.creat_tags_meta(sentences)
        self.create_word_idx_translation(sentences)
        #instializing master transition, emission and confusion matrices
        master_transition_matrix = []
        master_emission_matrix = []
        master_confusion_matrix = []

        for i in range(len(self.count_of_each_tag)):
            master_transition_matrix.append([0]*len(self.count_of_each_tag))
            master_confusion_matrix.append([0]*len(self.count_of_each_tag))

        for i in range(len(self.word_to_idx)):
            master_emission_matrix.append([0]*len(self.count_of_each_tag))



        #model building and traing on k sets
        for i in keys:

            #**********************------------------create test and train------------------************************************
            train_data=[]
            for j in cross_validation_set.keys():
                if j!=i:
                    train_data+=cross_validation_set[j]

            test_data=cross_validation_set[i]
            #**********************---------------------------------------************************************
            

            #**********************---------------create tags and words meta data------------------------************************************
            model.creat_tags_meta(train_data)
            model.create_word_idx_translation(train_data)
            #**********************---------------------------------------************************************


            #**********************---------------build transition and emission matrix------------------------************************************
            
            #bigram_matrix : count of every bigram tags
            #transmission_matrix : probability of every bigram tags
            #emission_count_matrix : given a word, what is the count of each tag that it can have
            #emission_matrix : given a word, what is the probability of tag
            model.create_transition_matrix(train_data)
            model.create_emission_matrix(train_data)

            #**********************---------------------------------------************************************
            
            #**********************---------------initialize confustion matrix------------------------************************************
            #confusion_matrix : A matrix with rows having actual tag and columns showing predicted tag
            model.confusion_matrix=[]
            for j in range(len(model.count_of_each_tag)):
                model.confusion_matrix.append([0]*len(model.count_of_each_tag))
            #**********************---------------------------------------************************************


            #**********************---------------bulding confustion matrix------------------------************************************
            coun=0
            for sen in test_data:
                coun+=1
                tags_output = model.HMM_logic(remove_tags(sen))
                model.build_confusion_matrix(sen,tags_output)
            #**********************---------------------------------------************************************

            
            


            #**********************---------------build masters------------------------************************************
            for i in range(len(model.count_of_each_tag)):
                row = model.idx_to_tag[i]
                for j in range(len(self.count_of_each_tag)):
                    col = model.idx_to_tag[j]
                    master_transition_matrix[self.tag_to_idx[row]][self.tag_to_idx[col]] += model.transition_matrix[i][j]
                    master_confusion_matrix[self.tag_to_idx[row]][self.tag_to_idx[col]] += model.confusion_matrix[i][j]
            
            for i in range(len(model.word_to_idx)):
                row = model.idx_to_word[i]
                for j in range(len(self.count_of_each_tag)):
                    col = model.idx_to_tag[j]
                    master_emission_matrix[self.word_to_idx[row]][self.tag_to_idx[col]] += model.emission_matrix[i][j]
            
            #**********************---------------------------------------*******************************************
                

        self.confusion_matrix = np.array(master_confusion_matrix)/k
        self.transition_matrix=np.array(master_transition_matrix)/k
        self.emission_matrix=np.array(master_emission_matrix)/k


    
            

