# coding=utf-8
"""
参考
1、https://radimrehurek.com/gensim/auto_examples/tutorials/run_word2vec.html
2、https://mp.weixin.qq.com/s/j8JPMZSPoVT_hQswX5QVxA
"""

from gensim.models import word2vec

# 用生成器的方式读取文件里的句子, 适合读取大容量文件，而不用加载到内存
class MySentences(object):
    def __init__(self, fname):
        self.fname = fname

    def __iter__(self):
        for line in open(self.fname, 'r'):
            yield line.split(' ')


"""
# 数据集不够大时，停止词太多，解决方法：去除停止词
from nltk.corpus import stopwords # python2下执行该指令存在问题, 待解决
StopWords = stopwords.words('english')
# 查看部分停止词
StopWords[:20]
"""

StopWords = ["a", "the", "."]
class MySentences_removeStopWords(object):
    def __init__(self, fname):
        self.fname = fname

    def __iter__(self):
        for line in open(self.fname, 'r'):
            yield [w for w in line.split(' ') if w not in StopWords]


# 模型训练函数
def w2vTrain(f_input, model_output):
    MIN_COUNT = 4
    CPU_NUM = 2  # 需要预先安装 Cython 以支持并行
    VEC_SIZE = 20
    CONTEXT_WINDOW = 5  # 提取目标词上下文距离最长5个词

    # sentences = MySentences(f_input)
    sentences = MySentences_removeStopWords(f_input)
    w2v_model = word2vec.Word2Vec(sentences,
                                  min_count=MIN_COUNT,
                                  workers=CPU_NUM,
                                  size=VEC_SIZE,
                                  window=CONTEXT_WINDOW)
    w2v_model.save(model_output)


if __name__ == "__main__":
    # 模型训练
    f_input = "./data/bioCorpus_5000.txt"
    model_output = "./model/test_w2v_model"
    w2vTrain(f_input, model_output)

    # 模型评估
    w2v_model = word2vec.Word2Vec.load(model_output)
    print w2v_model.most_similar('body')    # 检索相似单词
    print w2v_model.wv['body']     # 返回单词的词向量




