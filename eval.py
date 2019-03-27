import tensorflow as tf

from google.protobuf import text_format
from proto.eval_config_pb2 import EvalConfig
from core import evaluator_interactive as evaluator
import os
tf.logging.set_verbosity(tf.logging.INFO)

flags = tf.app.flags
flags.DEFINE_string('config_file', './model/eval.config',
                    'Path of config file')

flags.DEFINE_integer('id', 0,
                    'Path of config file')
FLAGS = flags.FLAGS

id = FLAGS.id
os.environ["CUDA_VISIBLE_DEVICES"] = '%d'%(id%4)
def get_configs():
    eval_config = EvalConfig()
    with open(FLAGS.config_file, 'r') as f:
        text_format.Merge(f.read(), eval_config)
    tf.logging.info(eval_config)
    return eval_config


def main():
    eval_config = get_configs()
    evaluator.evaluate(eval_config)



if __name__ == '__main__':
    #tf.app.run()
    main()
