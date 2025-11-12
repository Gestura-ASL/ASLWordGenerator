"""
asl_transformer_model.py

Complete, self-contained script of the ASL transformer model you designed.


Notes:
- Custom MultiHeadAttention implemented because TFLite doesn't support tf.keras.layers.MultiHeadAttention.
- Masked softmax is implemented by turning mask into a (B, Tx, Tx) mask applied to attention logits.
- Uses tensorflow-addons optimizer (AdamW) as in your original snippet.
"""

import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa

# ----------------------------------------
# ---------- Configuration --------------
# ----------------------------------------
# Replace these placeholders with real values from your preprocessing script
INPUT_SIZE = 64            # number of frames (timesteps) the model expects
N_COLS = 66                # number of landmark columns (e.g. lips + left_hand + pose)
N_DIMS = 2                 # typically x,y

# Landmark slicing indices (match your data layout)
LIPS_START = 0
LIPS_COUNT = 40            # lips landmarks count
LEFT_HAND_START = 40
LEFT_HAND_COUNT = 21
POSE_START = 61
POSE_COUNT = 5

# Number of classes (sign vocabulary)
NUM_CLASSES = 250

# Statistical normalizers (placeholders). Replace with your real mean/std arrays or scalars.
LIPS_MEAN = 0.0
LIPS_STD = 1.0
LEFT_HANDS_MEAN = 0.0
LEFT_HANDS_STD = 1.0
POSE_MEAN = 0.0
POSE_STD = 1.0

# Model hyper-parameters (from your model config)
LAYER_NORM_EPS = 1e-6
LIPS_UNITS = 384
HANDS_UNITS = 384
POSE_UNITS = 384
UNITS = 512

NUM_BLOCKS = 2
MLP_RATIO = 2

EMBEDDING_DROPOUT = 0.00
MLP_DROPOUT_RATIO = 0.30
CLASSIFIER_DROPOUT_RATIO = 0.10

INIT_HE_UNIFORM = tf.keras.initializers.he_uniform
INIT_GLOROT_UNIFORM = tf.keras.initializers.glorot_uniform
INIT_ZEROS = tf.keras.initializers.constant(0.0)
GELU = tf.keras.activations.gelu

print(f'UNITS: {UNITS}')

# ----------------------------------------
# ---------- Attention Helpers ----------
# ----------------------------------------
def scaled_dot_product(q, k, v, attention_mask=None):
    """
    q: (B, Tx, depth)
    k: (B, Tx, depth)
    v: (B, Tx, depth)
    attention_mask: expected shape (B, Tx, 1) or (B, Tx) where 1 indicates present frame.
                    We'll convert to (B, Tx, Tx) to mask logits.
    Returns: (B, Tx, depth)
    """
    # Q . K^T -> (B, Tx, Tx)
    qkt = tf.matmul(q, k, transpose_b=True)

    # scale
    dk = tf.math.sqrt(tf.cast(tf.shape(q)[-1], tf.float32))
    scaled_qkt = qkt / dk


    if attention_mask is not None:
        # attention_mask expected shape (B, Tx, 1) or (B, Tx)
        # Convert to (B, Tx, Tx) by outer product: mask_i * mask_j
        a_mask = tf.squeeze(attention_mask, axis=-1) if len(attention_mask.shape) == 3 else attention_mask  # (B, Tx)
        a_mask = tf.cast(a_mask, tf.float32)
        # outer product -> (B, Tx, Tx)
        attn_mask_matrix = tf.matmul(tf.expand_dims(a_mask, axis=2), tf.expand_dims(a_mask, axis=1))
        # where attn_mask_matrix == 0 -> masked positions, add large negative to logits
        large_neg = -1e9
        scaled_qkt = scaled_qkt + (1.0 - attn_mask_matrix) * large_neg

    # softmax on last axis (Tx dimension)
    softmaxed = tf.nn.softmax(scaled_qkt, axis=-1)

    # attention output
    z = tf.matmul(softmaxed, v)  # (B, Tx, depth)
    return z

# ----------------------------------------
# ---------- MultiHeadAttention ----------
# ----------------------------------------
class MultiHeadAttention(tf.keras.layers.Layer):
    """
    A simple multi-head attention made from scratch.
    Splits the last dimension into `num_of_heads` heads, runs scaled dot-product attention per head,
    then concatenates and projects back to d_model.
    """
    def __init__(self, d_model, num_of_heads):
        super(MultiHeadAttention, self).__init__()
        assert d_model % num_of_heads == 0, "d_model must be divisible by num_of_heads"
        self.d_model = d_model
        self.num_of_heads = num_of_heads
        self.depth = d_model // num_of_heads

        # per-head linear projections for q,k,v
        self.wq = [tf.keras.layers.Dense(self.depth, use_bias=False) for _ in range(num_of_heads)]
        self.wk = [tf.keras.layers.Dense(self.depth, use_bias=False) for _ in range(num_of_heads)]
        self.wv = [tf.keras.layers.Dense(self.depth, use_bias=False) for _ in range(num_of_heads)]

        # final output projection
        self.wo = tf.keras.layers.Dense(d_model, kernel_initializer=INIT_GLOROT_UNIFORM)

    def call(self, x, attention_mask=None):
        # x: (B, Tx, d_model)
        multi_attn = []
        for i in range(self.num_of_heads):
            Q = self.wq[i](x)  # (B, Tx, depth)
            K = self.wk[i](x)
            V = self.wv[i](x)
            head = scaled_dot_product(Q, K, V, attention_mask)
            multi_attn.append(head)

        # concat heads -> (B, Tx, depth * num_heads) = (B, Tx, d_model)
        multi_head = tf.concat(multi_attn, axis=-1)
        out = self.wo(multi_head)
        return out

# ----------------------------------------
# ---------- Transformer (encoder) -------
# ----------------------------------------
class Transformer(tf.keras.Model):
    """
    A minimal transformer-style encoder built from stacking custom MHA + MLP blocks.
    Each block: x = x + MHA(layernorm(x)); x = x + MLP(layernorm(x))
    Note: For simplicity we used pre-norm additive connections similar to original snippet.
    """
    def __init__(self, num_blocks):
        super(Transformer, self).__init__(name='transformer')
        self.num_blocks = num_blocks
        self.mhas = []
        self.mlps = []

    def build(self, input_shape):
        # create lists of layers per block
        for i in range(self.num_blocks):
            self.mhas.append(MultiHeadAttention(UNITS, 8))
            self.mlps.append(tf.keras.Sequential([
                tf.keras.layers.Dense(UNITS * MLP_RATIO, activation=GELU, kernel_initializer=INIT_GLOROT_UNIFORM),
                tf.keras.layers.Dropout(MLP_DROPOUT_RATIO),
                tf.keras.layers.Dense(UNITS, kernel_initializer=INIT_HE_UNIFORM),
            ]))

    def call(self, x, attention_mask):
        # x: (B, Tx, UNITS)
        for mha, mlp in zip(self.mhas, self.mlps):
            # Residual connection around MHA
            x = x + mha(x, attention_mask)
            # Residual connection around MLP
            x = x + mlp(x)
        return x

# ----------------------------------------
# ---------- Landmark Embeddings ---------
# ----------------------------------------
class LandmarkEmbedding(tf.keras.Model):
    """
    Embed per-landmark-group (e.g., lips, left_hand, pose).
    If a frame's landmarks are all zeros, replace with learnable empty embedding vector.
    """
    def __init__(self, units, name):
        super(LandmarkEmbedding, self).__init__(name=f'{name}_embedding')
        self.units = units

    def build(self, input_shape):
        # empty embedding for missing landmark sets in a frame
        self.empty_embedding = self.add_weight(
            name=f'{self.name}_empty_embedding',
            shape=[self.units],
            initializer=INIT_ZEROS,
        )
        self.dense = tf.keras.Sequential([
            tf.keras.layers.Dense(self.units, name=f'{self.name}_dense_1', use_bias=False, kernel_initializer=INIT_GLOROT_UNIFORM),
            tf.keras.layers.Activation(GELU),
            tf.keras.layers.Dense(self.units, name=f'{self.name}_dense_2', use_bias=False, kernel_initializer=INIT_HE_UNIFORM),
        ], name=f'{self.name}_dense')

    def call(self, x):
        # x shape: (B, INPUT_SIZE, S, 2) -> sum across coords to detect empty frames
        is_empty = tf.reduce_sum(x, axis=2, keepdims=True) == 0
        # if empty -> empty_embedding (broadcasted), else -> dense embedding
        embedded = tf.where(is_empty, self.empty_embedding, self.dense(x))
        return embedded  # shape -> (B, INPUT_SIZE, units)

# ----------------------------------------
# ---------- Combined Embedding ----------
# ----------------------------------------
class Embedding(tf.keras.Model):
    def __init__(self):
        super(Embedding, self).__init__()

    def build(self, input_shape):
        # positional embedding (INPUT_SIZE + 1 for masked index)
        self.positional_embedding = tf.keras.layers.Embedding(INPUT_SIZE + 1, UNITS, embeddings_initializer=INIT_ZEROS)

        # per-landmark embeddings
        self.lips_embedding = LandmarkEmbedding(LIPS_UNITS, 'lips')
        self.left_hand_embedding = LandmarkEmbedding(HANDS_UNITS, 'left_hand')
        self.pose_embedding = LandmarkEmbedding(POSE_UNITS, 'pose')

        # learned weights for merging landmarks
        self.landmark_weights = tf.Variable(tf.zeros([3], dtype=tf.float32), name='landmark_weights')

        # final fully connected net to map combined landmarks -> UNITS
        self.fc = tf.keras.Sequential([
            tf.keras.layers.Dense(UNITS, name='fully_connected_1', use_bias=False, kernel_initializer=INIT_GLOROT_UNIFORM),
            tf.keras.layers.Activation(GELU),
            tf.keras.layers.Dense(UNITS, name='fully_connected_2', use_bias=False, kernel_initializer=INIT_HE_UNIFORM),
        ], name='fc')

    def call(self, lips0, left_hand0, pose0, non_empty_frame_idxs, training=False):
        """
        Inputs:
            lips0: (B, INPUT_SIZE, 40*2) OR (B, INPUT_SIZE, S, 2) depending on prior reshape
            non_empty_frame_idxs: (B, INPUT_SIZE), float-like indexes or -1 for empty
        Returns:
            x: (B, INPUT_SIZE, UNITS)
        """
        # If inputs are flattened (B, INPUT_SIZE, S*2), reshape back to (B, INPUT_SIZE, S, 2)
        # The calling code already uses flattened inputs; here we try to be compatible with either case.
        def _ensure_4d(t, coords):
            if len(t.shape) == 3:
                s = t.shape[-1] // 2
                return tf.reshape(t, [-1, INPUT_SIZE, s, 2])
            return t

        lips0 = _ensure_4d(lips0, LIPS_COUNT)
        left_hand0 = _ensure_4d(left_hand0, LEFT_HAND_COUNT)
        pose0 = _ensure_4d(pose0, POSE_COUNT)

        lips_embedding = self.lips_embedding(lips0)           # (B, INPUT_SIZE, LIPS_UNITS)
        left_hand_embedding = self.left_hand_embedding(left_hand0)
        pose_embedding = self.pose_embedding(pose0)

        # stack and weighted-merge across the landmark groups
        x = tf.stack((lips_embedding, left_hand_embedding, pose_embedding), axis=3)  # (B, INPUT_SIZE, UNITS?, 3)
        x = x * tf.nn.softmax(self.landmark_weights)  # broadcast landmark weights across dims
        x = tf.reduce_sum(x, axis=3)  # (B, INPUT_SIZE, UNITS')

        # fully connected mapping into UNITS dims
        x = self.fc(x)  # (B, INPUT_SIZE, UNITS)

        # Positional embedding handling:
        # Non-empty frame idxs should be scaled/truncated into [0..INPUT_SIZE], with -1 meaning empty

        max_frame_idxs = tf.clip_by_value(tf.reduce_max(non_empty_frame_idxs, axis=1, keepdims=True), 1.0, np.PINF)
        normalised_non_empty_frame_idxs = tf.where(
            tf.math.equal(non_empty_frame_idxs, -1.0),
            INPUT_SIZE,
            tf.cast(non_empty_frame_idxs / max_frame_idxs * INPUT_SIZE, tf.int32),
        )  # shape (B, INPUT_SIZE)

        # positional embedding add
        x = x + self.positional_embedding(normalised_non_empty_frame_idxs)

        return x

# ----------------------------------------
# ---------- Augmentation (optional) -----
# ----------------------------------------
class Augmentation(tf.keras.layers.Layer):
    """
    Optional augmentation that adds gaussian noise to non-zero landmarks during training.
    Not used by default.
    """
    def __init__(self, noise_std):
        super(Augmentation, self).__init__()
        self.noise_std = noise_std

    def add_noise(self, t):
        B = tf.shape(t)[0]
        return tf.where(
            t == 0.0,
            0.0,
            t + tf.random.normal([B, 1, 1, tf.shape(t)[3]], 0.0, self.noise_std),
        )

    def call(self, lips0, left_hand0, pose0, training=False):
        if training:
            lips0 = self.add_noise(lips0)
            left_hand0 = self.add_noise(left_hand0)
            pose0 = self.add_noise(pose0)
        return lips0, left_hand0, pose0

# ----------------------------------------
# ---------- Loss (SCCE + Label Smoothing)
# ----------------------------------------
def scce_with_ls(y_true, y_pred):
    """
    Sparse categorical crossentropy with label smoothing.
    y_true: (B,) integer labels
    y_pred: (B, num_classes) logits (or probabilities)
    """
    y_true = tf.cast(y_true, tf.int32)
    y_true = tf.one_hot(y_true, NUM_CLASSES, axis=1)
    y_true = tf.squeeze(y_true, axis=2)
    return tf.keras.losses.categorical_crossentropy(y_true, y_pred, label_smoothing=0.25)

# ----------------------------------------
# ---------- Model Builder ---------------
# ----------------------------------------


def get_model():
    """
    Returns compiled tf.keras Model matching your architecture.
    """
    # Inputs

    frames = tf.keras.layers.Input([INPUT_SIZE, N_COLS, N_DIMS], dtype=tf.float32, name='frames')
    non_empty_frame_idxs = tf.keras.layers.Input([INPUT_SIZE], dtype=tf.float32, name='non_empty_frame_idxs')

    # Padding mask (1 for non-empty frames, 0 for empty)

    mask0 = tf.cast(tf.math.not_equal(non_empty_frame_idxs, -1), tf.float32)     # (B, INPUT_SIZE)
    mask0 = tf.expand_dims(mask0, axis=2)                                       # (B, INPUT_SIZE, 1)

    # Random frame masking (data augmentation / denoising objective)
    mask = tf.where(
        (tf.random.uniform(tf.shape(mask0)) > 0.25) & tf.math.not_equal(mask0, 0.0),
        1.0,
        0.0,
    )
    # Ensure not all frames masked; fallback to mask0
    mask = tf.where(tf.math.equal(tf.reduce_sum(mask, axis=[1,2], keepdims=True), 0.0), mask0, mask)

    # Extract x,y coords only (first two dims)
    x = tf.slice(frames, [0, 0, 0, 0], [-1, INPUT_SIZE, N_COLS, 2])

    # Slice landmarks (these ranges must match how you assembled the frames)
    lips = tf.slice(x, [0, 0, LIPS_START, 0], [-1, INPUT_SIZE, LIPS_COUNT, 2])
    lips = tf.where(tf.math.equal(lips, 0.0), 0.0, (lips - LIPS_MEAN) / LIPS_STD)

    left_hand = tf.slice(x, [0, 0, LEFT_HAND_START, 0], [-1, INPUT_SIZE, LEFT_HAND_COUNT, 2])
    left_hand = tf.where(tf.math.equal(left_hand, 0.0), 0.0, (left_hand - LEFT_HANDS_MEAN) / LEFT_HANDS_STD)

    pose = tf.slice(x, [0, 0, POSE_START, 0], [-1, INPUT_SIZE, POSE_COUNT, 2])
    pose = tf.where(tf.math.equal(pose, 0.0), 0.0, (pose - POSE_MEAN) / POSE_STD)

    # Flatten per-landmark groups for embedding module (it handles both shapes)

    lips = tf.reshape(lips, [-1, INPUT_SIZE, LIPS_COUNT * 2])
    left_hand = tf.reshape(left_hand, [-1, INPUT_SIZE, LEFT_HAND_COUNT * 2])
    pose = tf.reshape(pose, [-1, INPUT_SIZE, POSE_COUNT * 2])

    # Embedding -> (B, INPUT_SIZE, UNITS)
    x_emb = Embedding()(lips, left_hand, pose, non_empty_frame_idxs)

    # Transformer encoder blocks (mask is used inside attention)
    x_enc = Transformer(NUM_BLOCKS)(x_emb, mask)

    # Pooling (mean over frames weighted by mask)
    # mask shape (B, INPUT_SIZE, 1) -> align dims

    x_masked = x_enc * mask
    x = tf.reduce_sum(x_masked, axis=1) / tf.reduce_sum(mask, axis=1)

    # Dropout then classification

    x = tf.keras.layers.Dropout(CLASSIFIER_DROPOUT_RATIO)(x)
    outputs = tf.keras.layers.Dense(NUM_CLASSES, activation=tf.keras.activations.softmax, kernel_initializer=INIT_GLOROT_UNIFORM)(x)

    model = tf.keras.models.Model(inputs=[frames, non_empty_frame_idxs], outputs=outputs)


    loss = scce_with_ls
    optimizer = tfa.optimizers.AdamW(learning_rate=1e-3, weight_decay=1e-5, clipnorm=1.0)
    metrics = [
        tf.keras.metrics.SparseCategoricalAccuracy(name='acc'),
        tf.keras.metrics.SparseTopKCategoricalAccuracy(k=5, name='top_5_acc'),
        tf.keras.metrics.SparseTopKCategoricalAccuracy(k=10, name='top_10_acc'),
    ]
    model.compile(loss=loss, optimizer=optimizer, metrics=metrics)

    return model


tf.keras.backend.clear_session()
model = get_model()


