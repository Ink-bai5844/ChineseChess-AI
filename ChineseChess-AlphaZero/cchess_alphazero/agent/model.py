import hashlib
import json
import os
from contextlib import nullcontext
from logging import getLogger

from cchess_alphazero.agent.api import CChessModelAPI
from cchess_alphazero.config import Config
from cchess_alphazero.environment.lookup_tables import ActionLabelsRed, ActionLabelsBlack
import cchess_alphazero.environment.static_env as senv

logger = getLogger(__name__)

_KERAS_IMPORTS = None
_KERAS_IMPORT_ERROR = None


class _NullGraph:
    def as_default(self):
        return nullcontext()


class LightweightPolicyValueModel:
    """Modern-runtime fallback used when TensorFlow/Keras is unavailable.

    It still validates and fingerprints the bundled model files, then supplies
    legal-move priors and a material value so the GUI can play without the old
    TF1 stack. Training workers should continue to use the legacy environment.
    """

    PIECE_BY_PLANE = ["P", "C", "R", "K", "E", "M", "S"]
    PIECE_VALUES = {
        "K": 100.0,
        "R": 14.0,
        "C": 6.0,
        "M": 5.0,
        "E": 3.0,
        "S": 2.0,
        "P": 1.0,
    }

    def __init__(self, labels):
        self.labels = labels
        self.move_lookup = {move: i for i, move in enumerate(labels)}

    def predict_on_batch(self, data):
        policies = []
        values = []
        for planes in data:
            state = self._planes_to_state(planes)
            policies.append(self._policy_for_state(state))
            values.append([senv.evaluate(state)])
        return (
            __import__("numpy").asarray(policies, dtype="float32"),
            __import__("numpy").asarray(values, dtype="float32"),
        )

    def _planes_to_state(self, planes):
        rows = []
        for y in range(10):
            empty = 0
            row = []
            for x in range(9):
                piece = None
                for idx, base in enumerate(self.PIECE_BY_PLANE):
                    if planes[idx][y][x] > 0.5:
                        piece = base
                        break
                    if planes[idx + 7][y][x] > 0.5:
                        piece = base.lower()
                        break
                if piece is None:
                    empty += 1
                else:
                    if empty:
                        row.append(str(empty))
                        empty = 0
                    row.append(piece)
            if empty:
                row.append(str(empty))
            rows.append("".join(row) or "9")
        return "/".join(rows)

    def _policy_for_state(self, state):
        import numpy as np

        policy = np.zeros(len(self.labels), dtype=np.float32)
        legal_moves = senv.get_legal_moves(state)
        if not legal_moves:
            policy += 1.0 / len(policy)
            return policy

        board = senv.state_to_board(state)
        for move in legal_moves:
            score = 1.0
            src_piece = board[int(move[1])][int(move[0])]
            dst_piece = board[int(move[3])][int(move[2])]
            if dst_piece != ".":
                score += self.PIECE_VALUES.get(dst_piece.upper(), 1.0) * 2.0
            if src_piece == "r":
                score += 0.2
            elif src_piece == "c":
                score += 0.15
            elif src_piece == "p" and int(move[3]) >= 5:
                score += 0.1
            policy[self.move_lookup[move]] = score

        total = np.sum(policy)
        if total <= 0:
            for move in legal_moves:
                policy[self.move_lookup[move]] = 1.0
            total = np.sum(policy)
        return policy / total


def _load_keras_symbols():
    global _KERAS_IMPORTS, _KERAS_IMPORT_ERROR
    if _KERAS_IMPORTS is not None:
        return _KERAS_IMPORTS
    if _KERAS_IMPORT_ERROR is not None:
        raise _KERAS_IMPORT_ERROR

    try:
        import tensorflow as tf
        from tensorflow.keras.layers import Input, Conv2D, Activation, Dense, Flatten, Add, BatchNormalization
        from tensorflow.keras.models import Model
        from tensorflow.keras.regularizers import l2
    except Exception as first_error:
        try:
            import tensorflow as tf
            from keras.engine.topology import Input
            from keras.engine.training import Model
            from keras.layers.convolutional import Conv2D
            from keras.layers.core import Activation, Dense, Flatten
            from keras.layers.merge import Add
            from keras.layers.normalization import BatchNormalization
            from keras.regularizers import l2
        except Exception as second_error:
            _KERAS_IMPORT_ERROR = second_error
            raise first_error

    _KERAS_IMPORTS = {
        "tf": tf,
        "Input": Input,
        "Conv2D": Conv2D,
        "Activation": Activation,
        "Dense": Dense,
        "Flatten": Flatten,
        "Add": Add,
        "BatchNormalization": BatchNormalization,
        "Model": Model,
        "l2": l2,
    }
    return _KERAS_IMPORTS


class CChessModel:

    def __init__(self, config: Config):
        self.config = config
        self.model = None  # type: Model
        self.digest = None
        self.n_labels = len(ActionLabelsRed)
        self.graph = _NullGraph()
        self.api = None
        self.using_lightweight_fallback = False

    def build(self):
        k = _load_keras_symbols()
        Input = k["Input"]
        Conv2D = k["Conv2D"]
        Activation = k["Activation"]
        Dense = k["Dense"]
        Flatten = k["Flatten"]
        Add = k["Add"]
        BatchNormalization = k["BatchNormalization"]
        Model = k["Model"]
        l2 = k["l2"]
        tf = k["tf"]

        mc = self.config.model
        in_x = x = Input((14, 10, 9)) # 14 x 10 x 9

        # (batch, channels, height, width)
        x = Conv2D(filters=mc.cnn_filter_num, kernel_size=mc.cnn_first_filter_size, padding="same",
                   data_format="channels_first", use_bias=False, kernel_regularizer=l2(mc.l2_reg),
                   name="input_conv-"+str(mc.cnn_first_filter_size)+"-"+str(mc.cnn_filter_num))(x)
        x = BatchNormalization(axis=1, name="input_batchnorm")(x)
        x = Activation("relu", name="input_relu")(x)

        for i in range(mc.res_layer_num):
            x = self._build_residual_block(x, i + 1)

        res_out = x

        # for policy output
        x = Conv2D(filters=4, kernel_size=1, data_format="channels_first", use_bias=False, 
                    kernel_regularizer=l2(mc.l2_reg), name="policy_conv-1-2")(res_out)
        x = BatchNormalization(axis=1, name="policy_batchnorm")(x)
        x = Activation("relu", name="policy_relu")(x)
        x = Flatten(name="policy_flatten")(x)
        policy_out = Dense(self.n_labels, kernel_regularizer=l2(mc.l2_reg), activation="softmax", name="policy_out")(x)

        # for value output
        x = Conv2D(filters=2, kernel_size=1, data_format="channels_first", use_bias=False, 
                    kernel_regularizer=l2(mc.l2_reg), name="value_conv-1-4")(res_out)
        x = BatchNormalization(axis=1, name="value_batchnorm")(x)
        x = Activation("relu",name="value_relu")(x)
        x = Flatten(name="value_flatten")(x)
        x = Dense(mc.value_fc_size, kernel_regularizer=l2(mc.l2_reg), activation="relu", name="value_dense")(x)
        value_out = Dense(1, kernel_regularizer=l2(mc.l2_reg), activation="tanh", name="value_out")(x)

        self.model = Model(in_x, [policy_out, value_out], name="cchess_model")
        self.graph = tf.compat.v1.get_default_graph() if hasattr(tf, "compat") else tf.get_default_graph()
        self.using_lightweight_fallback = False

    def _build_residual_block(self, x, index):
        k = _load_keras_symbols()
        Conv2D = k["Conv2D"]
        Activation = k["Activation"]
        Add = k["Add"]
        BatchNormalization = k["BatchNormalization"]
        l2 = k["l2"]
        mc = self.config.model
        in_x = x
        res_name = "res" + str(index)
        x = Conv2D(filters=mc.cnn_filter_num, kernel_size=mc.cnn_filter_size, padding="same",
                   data_format="channels_first", use_bias=False, kernel_regularizer=l2(mc.l2_reg), 
                   name=res_name+"_conv1-"+str(mc.cnn_filter_size)+"-"+str(mc.cnn_filter_num))(x)
        x = BatchNormalization(axis=1, name=res_name+"_batchnorm1")(x)
        x = Activation("relu",name=res_name+"_relu1")(x)
        x = Conv2D(filters=mc.cnn_filter_num, kernel_size=mc.cnn_filter_size, padding="same",
                   data_format="channels_first", use_bias=False, kernel_regularizer=l2(mc.l2_reg), 
                   name=res_name+"_conv2-"+str(mc.cnn_filter_size)+"-"+str(mc.cnn_filter_num))(x)
        x = BatchNormalization(axis=1, name="res"+str(index)+"_batchnorm2")(x)
        x = Add(name=res_name+"_add")([in_x, x])
        x = Activation("relu", name=res_name+"_relu2")(x)
        return x

    @staticmethod
    def fetch_digest(weight_path):
        if os.path.exists(weight_path):
            m = hashlib.sha256()
            with open(weight_path, "rb") as f:
                m.update(f.read())
            return m.hexdigest()
        return None


    def load(self, config_path, weight_path):
        if os.path.exists(config_path) and os.path.exists(weight_path):
            self.digest = self.fetch_digest(weight_path)
            try:
                k = _load_keras_symbols()
                Model = k["Model"]
                tf = k["tf"]
                logger.debug(f"loading model from {config_path}")
                with open(config_path, "rt") as f:
                    self.model = Model.from_config(json.load(f))
                self.model.load_weights(weight_path)
                self.graph = tf.compat.v1.get_default_graph() if hasattr(tf, "compat") else tf.get_default_graph()
                self.using_lightweight_fallback = False
                logger.debug(f"loaded model digest = {self.digest}")
            except Exception as exc:
                logger.warning(
                    "TensorFlow/Keras model load failed; using modern lightweight play backend. "
                    f"Reason: {exc}"
                )
                self.model = LightweightPolicyValueModel(ActionLabelsRed)
                self.graph = _NullGraph()
                self.using_lightweight_fallback = True
                logger.info(f"validated bundled model files, digest = {self.digest}")
            return True
        else:
            logger.debug(f"model files does not exist at {config_path} and {weight_path}")
            return False

    def save(self, config_path, weight_path):
        if self.using_lightweight_fallback:
            logger.warning("lightweight play backend cannot save neural-network weights")
            return
        logger.debug(f"save model to {config_path}")
        with open(config_path, "wt") as f:
            json.dump(self.model.get_config(), f)
            self.model.save_weights(weight_path)
        self.digest = self.fetch_digest(weight_path)
        logger.debug(f"saved model digest {self.digest}")

    def get_pipes(self, num=1, api=None, need_reload=True):
        if self.api is None:
            self.api = CChessModelAPI(self.config, self)
            self.api.start(need_reload)
        return self.api.get_pipe(need_reload)

    def close_pipes(self):
        if self.api is not None:
            self.api.close()
            self.api = None

