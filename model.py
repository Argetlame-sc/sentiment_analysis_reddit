import tensorflow as tf
import keras
from keras import layers as kl
from keras.models import Model
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from keras.losses import categorical_crossentropy
from keras.metrics import categorical_accuracy
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR) # Error with logging with tensorflow 1.14 and python 3.8

import gensim
from gensim.models import Word2Vec
from gensim.models.keyedvectors import KeyedVectors
import numpy as np

NUM_WORDS=25000
PERTURB_NORM_LENGTH = 5.0


def normalize_embedding(tokenizer, embedding_matrix):

    freq_vector = np.zeros((embedding_matrix.shape[0], 1))
    for i in range(1, embedding_matrix.shape[0]):
        freq_vector[i, :] = tokenizer.word_counts[tokenizer.index_word[i]]

    tot_word_occurence = np.sum(np.asarray(list(tokenizer.word_counts.values())).astype(np.float32))

    freq_vector = freq_vector / tot_word_occurence

    mean = np.sum(freq_vector * embedding_matrix, axis=0)

    var = np.sum(freq_vector * (embedding_matrix - mean)**2, axis=0)

    return (embedding_matrix - mean) / np.sqrt(var)


def embedding_matrix_word2vec(tokenizer, vocabulary_size, normalize=False):

    word_vectors = KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin', binary=True)

    print("word vectors read")
    embedding_dim=300


    word_index = tokenizer.word_index
    embedding_matrix = np.zeros((vocabulary_size, embedding_dim))

    for word, i in word_index.items():
        if i>=NUM_WORDS:
            continue
        try:
            embedding_vector = word_vectors[word]
            embedding_matrix[i] = embedding_vector
        except KeyError:
            embedding_matrix[i]=np.random.normal(0,np.sqrt(0.25),embedding_dim)

    del(word_vectors)

    if normalize:
        embedding_matrix = normalize_embedding(tokenizer, embedding_matrix)

    return embedding_matrix


class KFold_LSTM_Model():

    def __init__(self, num_split, *model_args, **model_kwargs):

        assert isinstance(num_split, int) and num_split > 0
        self.num_split = num_split

        self.models = [LSTM_Model(*model_args, **model_kwargs) for _ in range(num_split)]


    def __getitem__(self, idx):

        if idx < 0 or idx >= self.num_split:
            raise ValueError("Index {} is out of bound".format(idx))

        return self.models[idx]

    def predict(self, *args, **kwargs):

        results =  [self.models[i].predict(*args, **kwargs) for i in range(self.num_split)]
        if isinstance(results[0], tuple):
            results_comp = [None] * len(results[0])

            for i in range(len(results[0])):
                results_comp[i] = np.mean(np.stack([r[i] for r in results], axis=0), axis=0)

            return tuple(results_comp)

        return np.mean(np.stack(results, axis=0), axis=0)

    def evaluate(self, *args, **kwargs):

        results =  [self.models[i].evaluate(*args, **kwargs) for i in range(self.num_split)]
        results_comp = [None] * len(results[0])

        for i in range(len(results[0])):
            results_comp[i] = np.mean(np.stack([r[i] for r in results], axis=0), axis=0)

        return tuple(results_comp)

    def load_weights(self, filename, **kwargs):

        base_name = filename[:-5]

        for idx in range(self.num_split):

            self.models[idx].load_weights(base_name + "_fold_{}.hdf5".format(idx), **kwargs)

class LSTM_Model():

    def __init__(self, len_seq, voc_size, embedding_dim,
                    hidden_dim, num_cat, embedding_matrix=None,
                    trainable_embedding=False,
                    out_latent=False,
                    use_at=False,
                    use_vat=False,
                    use_weights=False,
                    lr=1e-3,
                    at_coeff=0.7):

        self.num_power_iteration = 1
        self.perturb_norm_length = PERTURB_NORM_LENGTH
        self.small_constant_for_finite_diff = 1e-1

        self.inp = kl.Input(shape=(len_seq,), name="text_inp")
        self.inp_unlabeled = kl.Input(shape=(len_seq,), name="unlabeled_text_inp")
        self.labels = kl.Input(shape=(num_cat,), name="label_inp")

        if embedding_matrix is None:
            self.emb_layer = kl.Embedding(voc_size, embedding_dim,
                                input_length=len_seq, mask_zero=True,
                                trainable=trainable_embedding)
        else:
            self.emb_layer = kl.Embedding(voc_size, embedding_dim,
                                weights=[embedding_matrix],
                                trainable=trainable_embedding, mask_zero=True,
                                input_length=len_seq)

        self.lstm_1 = kl.Bidirectional(kl.LSTM(hidden_dim, return_sequences=True))
        self.lstm_2 = kl.Bidirectional(kl.LSTM(hidden_dim))

        self.dense_latent = kl.Dense(hidden_dim // 2, activation="relu")
        self.dense_logits = kl.Dense(num_cat, name="dense_logits_{}".format(num_cat))

        self.dropout = kl.Dropout(0.5)
        emb = self.emb_layer(self.inp)
        emb = self.dropout(emb)
        logits, latent = self.forward_lstm(emb)

        emb_unlabeled = self.emb_layer(self.inp_unlabeled)
        emb_unlabeled = self.dropout(emb_unlabeled)
        logits_unlabeled, _ = self.forward_lstm(emb_unlabeled)

        x = kl.Activation("softmax", name="out_cat")(logits)

        if out_latent:
            if use_vat:
                self.model = Model(inputs=[self.inp, self.labels, self.inp_unlabeled], outputs=[x, latent])
            else:
                self.model = Model(inputs=[self.inp, self.labels], outputs=[x, latent])

        else:
            if use_vat:
                self.model = Model(inputs=[self.inp, self.labels, self.inp_unlabeled], outputs=x)
            else:
                self.model = Model(inputs=[self.inp, self.labels], outputs=x)

        loss = tf.nn.softmax_cross_entropy_with_logits_v2(labels=self.labels, logits=logits)
        if use_at:
            logits_perturb = self.adversial_logits(emb, loss)
            loss_at = at_coeff * tf.nn.softmax_cross_entropy_with_logits_v2(labels=self.labels, logits=logits_perturb)
        if use_vat:
            loss_vat = self.virtual_adversarial_loss(logits_unlabeled, emb_unlabeled)


        def loss_fn(y_true, y_pred):

            if use_weights:
                weight = y_true
                out_loss = tf.multiply(weight, loss)

            out_loss = loss

            if use_at:
                out_loss += loss_at
            if use_vat:
                out_loss += loss_vat

            return out_loss


        def metric_acc(y_true, y_pred):
            return categorical_accuracy(self.labels, y_pred)

        def metric_loss_cat(y_true, y_pred):
            return categorical_crossentropy(self.labels, y_pred)

        self.model.compile(loss={"out_cat" : loss_fn},
                    optimizer=keras.optimizers.Adam(lr=lr),
                    metrics={"out_cat" : [metric_acc, metric_loss_cat]})

        self.model.summary()

    def predict(self, *args, **kwargs):

        return self.model.predict(*args, **kwargs)


    def evaluate(self, *args, **kwargs):

        return self.model.evaluate(*args, **kwargs)

    def fit(self, *args, **kwargs):

        return self.model.fit(*args, **kwargs)


    def fit_generator(self, *args, **kwargs):

        return self.model.fit_generator(*args, **kwargs)


    def load_weights(self, *args, **kwargs):

        self.model.load_weights(*args, **kwargs)

    def adversial_logits(self, embedded, loss):
        loss = tf.reduce_mean(loss)
        grad = tf.gradients(loss, embedded,
                            aggregation_method=tf.AggregationMethod.EXPERIMENTAL_ACCUMULATE_N)
        grad = tf.stop_gradient(grad)[0]
        perturb = self._scale_l2(grad, self.perturb_norm_length)
        print(perturb.shape.as_list())
        embedded_perturb = kl.Lambda(lambda x: x + perturb, mask=lambda inp, mask: mask)(embedded)

        logits_perturb, _ = self.forward_lstm(embedded_perturb)
        return  logits_perturb

    """Virtual adversarial loss.
    Computes virtual adversarial perturbation by finite difference method and
    power iteration, adds it to the embedding, and computes the KL divergence
    between the new logits and the original logits.
    Args:
        logits: 3-D float Tensor, [batch_size, num_timesteps, m], m=num_classes.
        embedded: 3-D float Tensor, [batch_size, num_timesteps, embedding_dim].

    Returns:
        kl: float scalar.
    """
    def virtual_adversarial_loss(self, logits, embedded):

    # Stop gradient of logits. See https://arxiv.org/abs/1507.00677 for details.
        logits = tf.stop_gradient(logits)

    # Initialize perturbation with random noise.
    # shape(embedded) = (batch_size, num_timesteps, embedding_dim)
        d = tf.random_normal(shape=tf.shape(embedded))

    # Perform finite difference method and power iteration.
    # See Eq.(8) in the paper http://arxiv.org/pdf/1507.00677.pdf,
    # Adding small noise to input and taking gradient with respect to the noise
    # corresponds to 1 power iteration.
        for _ in range(self.num_power_iteration):
            d = self._scale_l2(d, self.small_constant_for_finite_diff)

        embedded_d = kl.Lambda(lambda x: x + d, mask=lambda inp, mask: mask)(embedded)
        d_logits, _ = self.forward_lstm(embedded_d)
        kl_div = tf.reduce_mean(self._kl_divergence_with_logits(logits, d_logits))
        d = tf.gradients(
            kl_div,
            d,
            aggregation_method=tf.AggregationMethod.EXPERIMENTAL_ACCUMULATE_N)[0]
        d = tf.stop_gradient(d)

        perturb = self._scale_l2(d, self.perturb_norm_length)
        embedded_perturb = kl.Lambda(lambda x: x + perturb, mask=lambda inp, mask: mask)(embedded)

        vadv_logits,_ = self.forward_lstm(embedded_perturb)

        return self._kl_divergence_with_logits(logits, vadv_logits)

    def forward_lstm(self, embedded):

        x = self.lstm_2(embedded)
        #x = self.lstm_2(x)
        #x = self.dropout(x)
        x = self.dense_latent(x)
        
       # x = self.dropout(x)
        latent = x
        #x = kl.Dense(num_cat, name="dense_cat_{}".format(num_cat))(x)
        x = self.dense_logits(x)

        return x, latent

    def _scale_l2(self, x, norm_length):
    # shape(x) = (batch, num_timesteps, d)
    # Divide x by max(abs(x)) for a numerically stable L2 norm.
    # 2norm(x) = a * 2norm(x/a)
    # Scale over the full sequence, dims (1, 2)
        alpha = tf.reduce_max(tf.abs(x), (1, 2), keep_dims=True) + 1e-12
        l2_norm = alpha * tf.sqrt(
            tf.reduce_sum(tf.pow(x / alpha, 2), (1, 2), keep_dims=True) + 1e-6)
        x_unit = x / l2_norm
        return norm_length * x_unit

    """Returns weighted KL divergence between distributions q and p.
    Args:
        q_logits: logits for 1st argument of KL divergence shape
              [batch_size, num_timesteps, num_classes] if num_classes > 2, and
              [batch_size, num_timesteps] if num_classes == 2.
        p_logits: logits for 2nd argument of KL divergence with same shape q_logits.
        weights: 1-D float tensor with shape [batch_size, num_timesteps].
             Elements should be 1.0 only on end of sequences
    Returns:
        KL: float scalar.
    """
    def _kl_divergence_with_logits(self, q_logits, p_logits):
    
 
        q = tf.nn.softmax(q_logits)
        p = tf.nn.softmax(p_logits)

        return keras.losses.kullback_leibler_divergence(q, p)
