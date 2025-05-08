import  numpy as np
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from nltk.tokenize import word_tokenize

def cosine(u, v):
    return np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v))

def doc2vec(sentences, emb_size, input_size, freq_ignore, epoch_):
    print('model called')
    tokenized_sent = []
    for s in sentences:
        tokenized_sent.append(word_tokenize(s.lower()))
        tagged_data = [TaggedDocument(d, [i]) for i, d in enumerate(tokenized_sent)]

    #model_doc2vec = Doc2Vec(vector_size=emb_size, min_count=freq_ignore, epochs=epoch_)
    #model_doc2vec.build_vocab(tagged_data)
    #model_doc2vec.train(tagged_data, total_examples=model_doc2vec.corpus_count, epochs=model_doc2vec.epochs)
    model_doc2vec = Doc2Vec(tagged_data, vector_size = emb_size, window = input_size, min_count = freq_ignore, epochs = epoch_)
    print("model training")
    return model_doc2vec

def getEmbedding_doc2vec(model_doc2vec, qinst):
    test_doc = word_tokenize(qinst.lower())
    test_doc_vector = model_doc2vec.infer_vector(test_doc)
    return test_doc_vector

lines = ["mov rsi, 16",
       "int 0x80",
       "call printf",
       "mov rsi, message",
       "add eax, ebx",
       "xor rdi, rdi"]

#sentences = [line.rstrip() for line in open('train2.txt')]

embedding_size = 10
input_size = 2
freq_ignore = 0
epoch = 20

model = Doc2Vec.load("./doc2vecmodel_single_function_e10_i2.mod")
query = "mov eax, rdx"
print("Query Instruction = ", query)
query_vec = getEmbedding_doc2vec(model, query)
for sent in lines:
    sim = cosine(query_vec, getEmbedding_doc2vec(model, sent))
    print("Instruction = ", sent, "; similarity = ", sim)
