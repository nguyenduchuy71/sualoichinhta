MAXLEN = 40
ENC_EMB_DIM = 256
DEC_EMB_DIM = 256
ENC_HID_DIM = 512
DEC_HID_DIM = 512
BATCH_SIZE = 128

class Vocab():
    def __init__(self, chars):
        self.pad = 0
        self.go = 1
        self.eos = 2

        self.chars = chars
        self.c2i = {c: i + 3 for i, c in enumerate(chars)}
        self.i2c = {i + 3: c for i, c in enumerate(chars)}

        self.i2c[0] = '<pad>'
        self.i2c[1] = '<sos>'
        self.i2c[2] = '<eos>'

    def encode(self, chars):
        return [self.go] + [self.c2i[c] for c in chars] + [self.eos]

    def decode(self, ids):
        ids =[i for i in ids]
        first = 1 if self.go in ids else 0
        last = ids.index(self.eos) if self.eos in ids else None
        sent = ''.join([self.i2c[i] for i in ids[first:last]])
        return sent

    def __len__(self):
        return len(self.c2i) + 3

    def decode_split(self, ids):
        ids =[i for i in ids]
        first = 1 if self.go in ids else 0
        last = ids.index(self.eos) if self.eos in ids else None
        return [self.i2c[i] for i in ids[first:last]]

    def __str__(self):
        return self.chars


with open('./vocab.pkl', 'rb') as f:
    vocab = pickle.load(f)
vocab_size = vocab.__len__()

class Encoder(tf.keras.Model):
  def __init__(self, vocab_size, embedding_dim, enc_units, batch_sz):
    super(Encoder, self).__init__()
    self.batch_sz = batch_sz
    self.enc_units = enc_units
    self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
    self.gru = tf.keras.layers.GRU(self.enc_units,
                                   return_sequences=True,
                                   return_state=True,
                                   recurrent_initializer='glorot_uniform')
  def call(self, x, hidden):
    x = self.embedding(x)
    output, state = self.gru(x, initial_state = hidden)
    return output, state
    
  # khởi tạo first state để  trả về last state
  def initialize_hidden_state(self):
    return tf.zeros((self.batch_sz, self.enc_units))

class BahdanauAttention(tf.keras.layers.Layer):
  def __init__(self, units):
    super(BahdanauAttention, self).__init__()
    self.W1 = tf.keras.layers.Dense(units)
    self.W2 = tf.keras.layers.Dense(units)
    self.V = tf.keras.layers.Dense(1)

  def call(self, query, values):
    query_with_time_axis = tf.expand_dims(query, 1)

    score = self.V(tf.nn.tanh(
        self.W1(query_with_time_axis) + self.W2(values)))

    # attention_weights shape == (batch_size, max_length, 1)
    attention_weights = tf.nn.softmax(score, axis=1)

    # context_vector shape after sum == (batch_size, hidden_size)
    context_vector = attention_weights * values
    context_vector = tf.reduce_sum(context_vector, axis=1)

    return context_vector, attention_weights

class Decoder(tf.keras.Model):
  def __init__(self, vocab_size, embedding_dim, dec_units, batch_sz):
    super(Decoder, self).__init__()
    self.batch_sz = batch_sz
    self.dec_units = dec_units
    self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
    self.gru = tf.keras.layers.GRU(self.dec_units,
                                   return_sequences=True,
                                   return_state=True,
                                   recurrent_initializer='glorot_uniform')
    self.fc = tf.keras.layers.Dense(vocab_size)

    # used for attention
    self.attention = BahdanauAttention(self.dec_units)

  def call(self, x, hidden, enc_output):
    # enc_output shape == (batch_size, max_length, hidden_size)
    # gọi attention truyền vào last state của encode
    context_vector, attention_weights = self.attention(hidden, enc_output)

    # x shape after passing through embedding == (batch_size, 1, embedding_dim)
    # embedding target (en)
    x = self.embedding(x)

    # x shape after concatenation == (batch_size, 1, embedding_dim + hidden_size)
    # nối context vector(attention) và vector embedding target
    x = tf.concat([tf.expand_dims(context_vector, 1), x], axis=-1)

    output, state = self.gru(x)

    # output shape == (batch_size * 1, hidden_size)
    output = tf.reshape(output, (-1, output.shape[2]))

    # output shape == (batch_size, vocab)
    x = self.fc(output)

    # trả về last state của decode cho lần train từ kế tiếp 
    return x, state, attention_weights

class CustomSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
  def __init__(self, d_model, warmup_steps=4000):
    super(CustomSchedule, self).__init__()
    
    self.d_model = d_model
    self.d_model = tf.cast(self.d_model, tf.float32)

    self.warmup_steps = warmup_steps
    
  def __call__(self, step):
    arg1 = tf.math.rsqrt(step)
    arg2 = step * (self.warmup_steps ** -1.5)
    
    return tf.math.rsqrt(self.d_model) * tf.math.minimum(arg1, arg2)

def evaluate(inputs):
  
  inputs = tf.keras.preprocessing.sequence.pad_sequences(inputs, maxlen=MAXLEN, padding='post')
  inputs = tf.convert_to_tensor(inputs)
  result = []

  # tạo first state
  hidden = tf.zeros((1, ENC_HID_DIM))
  
  enc_out, state_h = encoder(inputs, hidden)
  
  # dec_hidden = enc_hidden
  dec_input = tf.expand_dims([vocab.go], 0)

  for t in range(MAXLEN):
    dec_hidden = state_h
    predictions, state_h, attention_weights = decoder(dec_input,dec_hidden,enc_out)

    # storing the attention weights to plot later on
    attention_weights = tf.reshape(attention_weights, (-1, ))
    predicted_id = tf.argmax(predictions[0]).numpy()
    result.append(predicted_id)
    if predicted_id == vocab.eos:
      return vocab.decode(result)

    # the predicted ID is fed back into the model
    dec_input = tf.expand_dims([predicted_id], 0)

  return vocab.decode(result)

encoder = Encoder(vocab_size, ENC_EMB_DIM, ENC_HID_DIM, BATCH_SIZE)
decoder = Decoder(vocab_size, DEC_EMB_DIM, DEC_HID_DIM, BATCH_SIZE)
optimizer = tf.keras.optimizers.Adam(CustomSchedule(256), beta_1=0.9, beta_2=0.98, epsilon=1e-9)
checkpoint_dir = 'checkpoints-seq-old-gru-5-25/'
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
checkpoint = tf.train.Checkpoint(optimizer=optimizer,encoder=encoder,decoder=decoder)
checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))
