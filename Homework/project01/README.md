文本摘要生成
----
#seq2seq_tf2<br>
为baseline模型，采用基于attention机制的seq2seq模型，rouge_L得分为19分<br>
<br>
#seq2seq_pgn_tf2<br>
引入PGN和coverage策略，解决OOV和word repition问题<br>
测试结果只训练了5个epoch，效果要比baseline好一些，后续调整超参数进行训练后测试
