import tensorflow as tf


class BahdanauAttention(tf.keras.layers.Layer):
    def __init__(self, units):
        super(BahdanauAttention, self).__init__()
        self.W1 = tf.keras.layers.Dense(units)
        self.W2 = tf.keras.layers.Dense(units)
        self.V = tf.keras.layers.Dense(1)

    def call(self, dec_hidden, enc_output):
        """
        :param dec_hidden: shape=(16, 256)
        :param enc_output: shape=(16, 200, 256)
        :param enc_padding_mask: shape=(16, 200)
        :param use_coverage:
        :param prev_coverage: None
        :return:
        """
        # hidden shape == (batch_size, hidden size)
        # hidden_with_time_axis shape == (batch_size, 1, hidden size)
        # we are doing this to perform addition to calculate the score
        hidden_with_time_axis = tf.expand_dims(dec_hidden, 1)  # shape=(16, 1, 256) # deco_hidden扩充一个时间维度才与encoder的输出shape一致
        # att_features = self.W1(enc_output) + self.W2(hidden_with_time_axis)

        # Calculate v^T tanh(W_h h_i + W_s s_t + b_attn)
        """
        定义score
        your code
        """
        score = self.V(tf.nn.tanh(self.W1(enc_output) + self.W2(hidden_with_time_axis))) # shape为(batch_size, max_length, 1)
        # Calculate attention distribution
        """
        归一化score，得到attn_dist
        your code
        """
        attn_dist = tf.nn.softmax(score, axis=1) # attention_weights，shape为(batch_size, max_length, 1)
        # context_vector shape after sum == (batch_size, hidden_size)
        context_vector = attn_dist * enc_output  # shape=(16, 200, 256) # context_vector的shape为(batch size, max_length, units)
        context_vector = tf.reduce_sum(context_vector, axis=1)  # shape=(16, 256) # 在length维度上求和得到最终的上下文信息context_vector，其shape=(batch_size, units)
        return context_vector, tf.squeeze(attn_dist, -1) # tf.squeeze从tensor中删除所有大小是1的维度


class Decoder(tf.keras.layers.Layer):
    def __init__(self, vocab_size, embedding_dim, dec_units, batch_sz, embedding_matrix):
        super(Decoder, self).__init__()
        self.batch_sz = batch_sz
        self.dec_units = dec_units
        """
        定义Embedding层，加载预训练的词向量
        your code
        """
        self.embedding = tf.keras.layers.Embedding(input_dim = vocab_size, # 字典长度
                            output_dim=embedding_dim, # 词向量长度（怎么确定）
                            weights=[embedding_matrix], # 重点：预训练的词向量系数
                            trainable=False # 是否在 训练的过程中 更新词向量
                            )
        """
        定义单向的RNN、GRU、LSTM层
        your code
        """
        # self.rnn = tf.keras.layers.RNN(cell=self.dec_units)
        self.gru = tf.keras.layers.GRU(self.dec_units,
                                    return_sequences=True,
                                    return_state=True) # glorot_uniform
        self.lstm = tf.keras.layers.LSTM(self.dec_units,
                                         return_sequences=True,
                                         return_state=True) # enc_units是输出的维度,lstm单元的hidden layer 的神经元数量
        # self.dropout = tf.keras.layers.Dropout(0.5)
        """
        定义最后的fc层，用于预测词的概率
        your code
        """
        self.fc = tf.keras.layers.Dense(vocab_size, activation=tf.keras.activations.softmax)

    def call(self, x, hidden, enc_output, context_vector):
        # enc_output shape == (batch_size, max_length, hidden_size)

        # x shape after passing through embedding == (batch_size, 1, embedding_dim)
        x = self.embedding(x)
        # print('x is ', x)

        # x shape after concatenation == (batch_size, 1, embedding_dim + hidden_size)
        x = tf.concat([tf.expand_dims(context_vector, 1), x], axis=-1)

        # passing the concatenated vector to the GRU
        output, state = self.gru(x)
        # output shape == (batch_size * 1, hidden_size)
        output = tf.reshape(output, (-1, output.shape[2]))

        # output = self.dropout(output)
        out = self.fc(output)
        # print('out is ', out)

        return x, out, state

