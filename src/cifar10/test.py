import numpy as np
import tensorflow as tf

with tf.compat.v1.variable_scope("aux_head"):
	aux_logits = tf.Variable(0)
	print(aux_logits.name)
	with tf.compat.v1.variable_scope("proj"):
		aux_logits = aux_logits+1
		print(aux_logits.name)
	with tf.compat.v1.variable_scope("avg_pool"):
		aux_logits = tf.Variable(0)
		print(aux_logits.name)
