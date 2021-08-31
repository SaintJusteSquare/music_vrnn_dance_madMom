"""import os
import sys
import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

import psutil
process = psutil.Process(os.getpid())
print(process.memory_info().rss)

from train_seq2seq import run

networks = os.path.join(os.getcwd(), 'networks')
sys.path.append(networks)
y_dim = 69
x_dim = 128 * 7
z_dim = 69

x2s_dim = 200
z2s_dim = 150

h_dim = 500

q_z_dim = 150
p_z_dim = 150
p_x_dim = 150

sequence = 100
batch_size = 128

from networks.seq2seqModel import Vrnn
model = Vrnn(y_dim=y_dim, x_dim=x_dim, x2s_dim=x2s_dim, h_dim=h_dim, z_dim=z_dim, z2s_dim=z2s_dim,
                     q_z_dim=q_z_dim,
                     p_z_dim=p_z_dim, p_x_dim=p_x_dim)
exp_folder = "exp"
model_name = os.path.join(exp_folder, "seq2seqModel")
run(exp=exp_folder, model_name=model_name, model=model)
print(process.memory_info().rss)
globals().clear()"""
"""
import os
import sys
import psutil
process = psutil.Process(os.getpid())
print(process.memory_info().rss)

from train_seq2seq import run

networks = os.path.join(os.getcwd(), 'networks')
sys.path.append(networks)
y_dim = 69
x_dim = 128 * 7
z_dim = 69

x2s_dim = 200
z2s_dim = 150

h_dim = 500

q_z_dim = 150
p_z_dim = 150
p_x_dim = 150

from networks.seq2seqModel0 import Vrnn
model = Vrnn(y_dim=y_dim, x_dim=x_dim, x2s_dim=x2s_dim, h_dim=h_dim, z_dim=z_dim, z2s_dim=z2s_dim,
                     q_z_dim=q_z_dim,
                     p_z_dim=p_z_dim, p_x_dim=p_x_dim)

exp_folder = "exp"
model_name = os.path.join(exp_folder, "seq2seqModel0")
run(exp=exp_folder, model_name=model_name, model=model)
print(process.memory_info().rss)
globals().clear()

import os
import sys
import psutil
process = psutil.Process(os.getpid())
print(process.memory_info().rss)

from train_seq2seq import run

networks = os.path.join(os.getcwd(), 'networks')
sys.path.append(networks)
y_dim = 69
x_dim = 128 * 7
z_dim = 69

x2s_dim = 200
z2s_dim = 150

h_dim = 500

q_z_dim = 150
p_z_dim = 150
p_x_dim = 150

from networks.seq2seqModel1 import Vrnn
model = Vrnn(y_dim=y_dim, x_dim=x_dim, x2s_dim=x2s_dim, h_dim=h_dim, z_dim=z_dim, z2s_dim=z2s_dim,
                     q_z_dim=q_z_dim,
                     p_z_dim=p_z_dim, p_x_dim=p_x_dim)
exp_folder = "exp"
model_name = os.path.join(exp_folder, "seq2seqModel1")
run(exp=exp_folder, model_name=model_name, model=model)
globals().clear()

import os
import sys
import psutil
process = psutil.Process(os.getpid())
print(process.memory_info().rss)

from train_seq2seq import run

networks = os.path.join(os.getcwd(), 'networks')
sys.path.append(networks)
y_dim = 69
x_dim = 128 * 7
z_dim = 69

x2s_dim = 200
z2s_dim = 150

h_dim = 500

q_z_dim = 150
p_z_dim = 150
p_x_dim = 150

from networks.seq2seqModel2 import Vrnn
model = Vrnn(y_dim=y_dim, x_dim=x_dim, x2s_dim=x2s_dim, h_dim=h_dim, z_dim=z_dim, z2s_dim=z2s_dim,
                     q_z_dim=q_z_dim,
                     p_z_dim=p_z_dim, p_x_dim=p_x_dim)
exp_folder = "exp"
model_name = os.path.join(exp_folder, "seq2seqModel2")
run(exp=exp_folder, model_name=model_name, model=model)
globals().clear()"""

"""
import os
import sys
import psutil
process = psutil.Process(os.getpid())
print(process.memory_info().rss)

from train_seq2seq import run

networks = os.path.join(os.getcwd(), 'networks')
sys.path.append(networks)
y_dim = 69
x_dim = 128 * 7
z_dim = 69

x2s_dim = 200
z2s_dim = 150

h_dim = 500

q_z_dim = 150
p_z_dim = 150
p_x_dim = 150

from networks.seq2seqModel3 import Vrnn
model = Vrnn(y_dim=y_dim, x_dim=x_dim, x2s_dim=x2s_dim, h_dim=h_dim, z_dim=z_dim, z2s_dim=z2s_dim,
                     q_z_dim=q_z_dim,
                     p_z_dim=p_z_dim, p_x_dim=p_x_dim)
exp_folder = "exp"
model_name = os.path.join(exp_folder, "seq2seqModel3")
run(exp=exp_folder, model_name=model_name, model=model)
globals().clear()
"""