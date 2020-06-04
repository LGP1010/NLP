import tensorflow as tf

# embeddings_matrix 初始化存储所有向量的大矩阵，留意其中多一位（首行），词向量全为 0，用于 padding补零。
# 行数 为 所有单词数+1 比如 10000+1 ； 列数为 词向量“维度”比如100。
# https://blog.csdn.net/jiangpeng59/article/details/77646186(LSTM参数详解)
class Encoder(tf.keras.layers.Layer):
    def __init__(self, vocab_size, embedding_dim, enc_units, batch_sz, embedding_matrix):
        super(Encoder, self).__init__()
        self.batch_sz = batch_sz
        # self.enc_units = enc_units
        self.enc_units = enc_units // 2
        """
        定义Embedding层，加载预训练的词向量
        your code
        """
        self.embedding = tf.keras.layers.Embedding(input_dim = vocab_size, # 字典长度
                            output_dim=embedding_dim, # 词向量长度（怎么确定）
                            weights=[embedding_matrix], # 重点：预训练的词向量系数
                            trainable=False # 是否在 训练的过程中 更新词向量
                            )

        # tf.keras.layers.GRU自动匹配cpu、gpu
        """
        定义单向的RNN、GRU、LSTM层
        your code
        """
        # self.rnn = tf.keras.layers.RNN(cell=self.enc_units)
        self.gru = tf.keras.layers.GRU(self.enc_units,
                                       return_sequences=True,
                                       return_state=True,
                                       recurrent_initializer='glorot_uniform') # glorot_uniform
        self.lstm = tf.keras.layers.LSTM(self.enc_units,
                                         return_sequences=True,
                                         return_state=True) # enc_units是输出的维度,lstm单元的hidden layer 的神经元数量
        # RNN 的双向封装器，对序列进行前向和后向计算
        self.bigru = tf.keras.layers.Bidirectional(self.gru, merge_mode='concat')

    def call(self, x, hidden):
        x = self.embedding(x) # 可降维和升维，将词语用低维向量表示，可解决one-hot编码的高维稀疏问题
        hidden = tf.split(hidden, num_or_size_splits=2, axis=1)
        output, forward_state, backward_state = self.bigru(x, initial_state=hidden)
        state = tf.concat([forward_state, backward_state], axis=1)
        # output, state = self.gru(x, initial_state=hidden)
        return output, state

    def initialize_hidden_state(self):
        return tf.zeros((self.batch_sz, 2*self.enc_units))
    
